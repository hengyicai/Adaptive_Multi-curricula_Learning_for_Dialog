#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import torch
import torch.nn as nn
from nltk.util import bigrams, trigrams
from torch.nn import functional as F

from parlai.agents.hy_lib.embedding_metrics import sentence_average_score, \
    sentence_greedy_score, sentence_extrema_score
from parlai.core.build_data import modelzoo_path
from parlai.core.torch_generator_agent import Output
from parlai.agents.dialog_evaluator.auto_evaluator import TorchGeneratorWithDialogEvalAgent
from parlai.core.utils import round_sigfigs, padded_tensor, fp16_optimizer_wrapper
from parlai.core.utils import warn_once
from .modules import Seq2seq, opt_to_kwargs


class Seq2seqAgent(TorchGeneratorWithDialogEvalAgent):
    """Agent which takes an input sequence and produces an output sequence.

    This model supports encoding the input and decoding the output via one of
    several flavors of RNN. It then uses a linear layer (whose weights can
    be shared with the embedding layer) to convert RNN output states into
    output tokens. This model supports greedy decoding, selecting the
    highest probability token at each time step, as well as beam
    search.

    For more information, see the following papers:
    - Neural Machine Translation by Jointly Learning to Align and Translate
      `(Bahdanau et al. 2014) <arxiv.org/abs/1409.0473>`_
    - Sequence to Sequence Learning with Neural Networks
      `(Sutskever et al. 2014) <arxiv.org/abs/1409.3215>`_
    - Effective Approaches to Attention-based Neural Machine Translation
      `(Luong et al. 2015) <arxiv.org/abs/1508.04025>`_
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-bi', '--bidirectional', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-att', '--attention', default='none',
                           choices=['none', 'concat', 'general', 'dot',
                                    'local'],
                           help='Choices: none, concat, general, local. '
                                'If set local, also set attention-length. '
                                '(see arxiv.org/abs/1508.04025)')
        agent.add_argument('-attl', '--attention-length', default=48, type=int,
                           help='Length of local attention.')
        agent.add_argument('--attention-time', default='post',
                           choices=['pre', 'post'],
                           help='Whether to apply attention before or after '
                                'decoding.')
        agent.add_argument('-rnn', '--rnn-class', default='lstm',
                           choices=Seq2seq.RNN_OPTS.keys(),
                           help='Choose between different types of RNNs.')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'],
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights. '
                                'Note that shared disabled some encoder '
                                'options--in particular, bidirectionality.')
        agent.add_argument('-lt', '--lookuptable', default='unique',
                           choices=['unique', 'enc_dec', 'dec_out', 'all'],
                           help='The encoder, decoder, and output modules can '
                                'share weights, or not. '
                                'Unique has independent embeddings for each. '
                                'Enc_dec shares the embedding for the encoder '
                                'and decoder. '
                                'Dec_out shares decoder embedding and output '
                                'weights. '
                                'All shares all three weights.')
        agent.add_argument('-soft', '--numsoftmax', default=1, type=int,
                           help='default 1, if greater then uses mixture of '
                                'softmax (see arxiv.org/abs/1711.03953).')
        agent.add_argument('-idr', '--input-dropout', type=float, default=0.0,
                           help='Probability of replacing tokens with UNK in training.')

        agent.add_argument('--weight_decay', type=float, default=0)
        # ---------------------- For logging ----------------------------------#
        agent.add_argument('--report_freq', type=float, default=0.1)

        super(Seq2seqAgent, cls).add_cmdline_args(argparser)
        return agent

    @staticmethod
    def model_version():
        """Return current version of this model, counting up from 0.

        Models may not be backwards-compatible with older versions.
        Version 1 split from version 0 on Aug 29, 2018.
        Version 2 split from version 1 on Nov 13, 2018
        To use version 0, use --model legacy:seq2seq:0
        To use version 1, use --model legacy:seq2seq:1
        (legacy agent code is located in parlai/agents/legacy_agents).
        """
        return 2

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr, 'weight_decay': self.opt.get('weight_decay', 3e-4)}
        if opt.get('momentum') > 0 and opt['optimizer'] in ['sgd', 'rmsprop', 'qhm']:
            # turn on momentum for optimizers that use it
            kwargs['momentum'] = opt['momentum']
            if opt['optimizer'] == 'sgd' and opt.get('nesterov', True):
                # for sgd, maybe nesterov
                kwargs['nesterov'] = opt.get('nesterov', True)
            elif opt['optimizer'] == 'qhm':
                # qhm needs a nu
                kwargs['nu'] = opt.get('nus', (0.7,))[0]
        elif opt['optimizer'] == 'adam':
            # turn on amsgrad for adam
            # amsgrad paper: https://openreview.net/forum?id=ryQu7f-RZ
            kwargs['amsgrad'] = True
        elif opt['optimizer'] == 'qhadam':
            # set nus for qhadam
            kwargs['nus'] = opt.get('nus', (0.7, 1.0))
        if opt['optimizer'] in ['adam', 'sparseadam', 'adamax', 'qhadam']:
            # set betas for optims that use it
            kwargs['betas'] = opt.get('betas', (0.9, 0.999))

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)
        if self.fp16:
            self.optimizer = fp16_optimizer_wrapper(self.optimizer)

        if optim_states:
            if saved_optim_type != opt['optimizer']:
                print('WARNING: not loading optim state since optim class '
                      'changed.')
            else:
                try:
                    self.optimizer.load_state_dict(optim_states)
                except ValueError:
                    print('WARNING: not loading optim state since model '
                          'params changed.')
                if self.use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        self.id = 'Seq2Seq'

    def build_model(self, states=None):
        """Initialize model, override to change model setup."""
        opt = self.opt
        if not states:
            states = {}

        kwargs = opt_to_kwargs(opt)
        self.model = Seq2seq(
            len(self.dict), opt['embeddingsize'], opt['hiddensize'],
            padding_idx=self.NULL_IDX, start_idx=self.START_IDX,
            end_idx=self.END_IDX, unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get('longest_label', 1),
            **kwargs)

        if (opt.get('dict_tokenizer') == 'bpe' and
                opt['embedding_type'] != 'random'):
            print('skipping preinitialization of embeddings for bpe')
        elif not states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(self.model.decoder.lt.weight,
                                  opt['embedding_type'])
            if opt['lookuptable'] in ['unique', 'dec_out']:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(self.model.encoder.lt.weight,
                                      opt['embedding_type'], log=False)

        if states:
            # set loaded states if applicable
            self.model.load_state_dict(states['model'])
            if 'longest_label' in states:
                self.model.longest_label = states['longest_label']

        if self.use_cuda:
            self.model.cuda()

        if opt['embedding_type'].endswith('fixed'):
            print('Seq2seq: fixing embedding weights.')
            self.model.decoder.lt.weight.requires_grad = False
            self.model.encoder.lt.weight.requires_grad = False
            if opt['lookuptable'] in ['dec_out', 'all']:
                self.model.decoder.e2s.weight.requires_grad = False

        if self.use_cuda:
            self.model.cuda()

        return self.model

    def build_criterion(self):
        # set up criteria
        if self.opt.get('numsoftmax', 1) > 1:
            self.criterion = nn.NLLLoss(
                ignore_index=self.NULL_IDX, size_average=False)
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, size_average=False)

        if self.use_cuda:
            self.criterion.cuda()

    def batchify(self, *args, **kwargs):
        """Override batchify options for seq2seq."""
        kwargs['sort'] = True  # need sorted for pack_padded
        return super().batchify(*args, **kwargs)

    def state_dict(self):
        """Get the model states for saving. Overriden to include longest_label"""
        states = super().state_dict()
        if hasattr(self.model, 'module'):
            states['longest_label'] = self.model.module.longest_label
        else:
            states['longest_label'] = self.model.longest_label

        return states

    def load(self, path):
        """Return opt and model states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        # set loaded states if applicable
        self.model.load_state_dict(states['model'])
        if 'longest_label' in states:
            self.model.longest_label = states['longest_label']
        return states

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        bsz = batch.text_vec.size(0)
        self.model.eval()

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss = self.compute_loss(batch)  # noqa: F841  we need the side effects
            self.metrics['loss'] += loss.item()

        preds = None
        if self.skip_generation:
            # noinspection PyTypeChecker
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning
            )
        elif self.beam_size == 1:
            # greedy decode
            _, preds, *_ = self.model(*self._model_input(batch), bsz=bsz)
        elif self.beam_size > 1:
            out = self.beam_search(
                self.model,
                batch,
                self.beam_size,
                start=self.START_IDX,
                end=self.END_IDX,
                pad=self.NULL_IDX,
                min_length=self.beam_min_length,
                min_n_best=self.beam_min_n_best,
                block_ngram=self.beam_block_ngram
            )
            beam_preds_scores, _, beams = out
            preds, scores = zip(*beam_preds_scores)

            if self.beam_dot_log is True:
                self._write_beam_dots(batch.text_vec, beams)

        cand_choices = None
        # TODO: abstract out the scoring here
        if self.rank_candidates:
            # compute roughly ppl to rank candidates
            cand_choices = []
            encoder_states = self.model.encoder(*self._model_input(batch))
            for i in range(bsz):
                num_cands = len(batch.candidate_vecs[i])
                enc = self.model.reorder_encoder_states(encoder_states, [i] * num_cands)
                cands, _ = padded_tensor(
                    batch.candidate_vecs[i], self.NULL_IDX, self.use_cuda
                )
                scores, _ = self.model.decode_forced(enc, cands)
                cand_losses = F.cross_entropy(
                    scores.view(num_cands * cands.size(1), -1),
                    cands.view(-1),
                    reduction='none',
                ).view(num_cands, cands.size(1))
                # now cand_losses is cands x seqlen size, but we still need to
                # check padding and such
                mask = (cands != self.NULL_IDX).float()
                cand_scores = (cand_losses * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
                _, ordering = cand_scores.sort()
                cand_choices.append([batch.candidates[i][o] for o in ordering])

        if batch.label_vec is not None:
            label_text = batch.labels
            # we are in the validation mode, print some generated responses for debugging
            for i in range(len(preds)):
                if random.random() > (1 - self.opt['report_freq']):
                    context_text = batch.observations[i]['text']
                    print('TEXT: ', context_text)
                    print('TARGET: ', self._v2t(batch.label_vec[i]))
                    print('PREDICTION: ', self._v2t(preds[i]), '\n~')
        else:
            label_text = None

        text = [self._v2t(p) for p in preds] if preds is not None else None
        context = [obs['text'] for obs in batch.observations]
        return Output(text, cand_choices), label_text, context

    def batch_act(self, observations):
        """
        Process a batch of observations (batchsize list of message dicts).

        These observations have been preprocessed by the observe method.

        Subclasses can override this for special functionality, but if the
        default behaviors are fine then just override the ``train_step`` and
        ``eval_step`` methods instead. The former is called when labels are
        present in the observations batch; otherwise, the latter is called.
        """
        batch_size = len(observations)
        # initialize a list of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]

        # check if there are any labels available, if so we will train on them
        is_training = any('labels' in obs for obs in observations)

        # create a batch from the vectors
        batch = self.batchify(observations)

        if is_training:
            output = self.train_step(batch)
        else:
            with torch.no_grad():
                # save memory and compute by disabling autograd.
                # use `with torch.enable_grad()` to gain back graidients.
                # noinspection PyTypeChecker
                eval_output = self.eval_step(batch)
                if eval_output is not None:
                    output = eval_output[0]
                    label_text = eval_output[1]
                    context = eval_output[2]
                    if label_text is not None:
                        # noinspection PyTypeChecker
                        self._eval_embedding_metrics(output, label_text, context)
                        # noinspection PyTypeChecker
                        self._eval_distinct_metrics(output, label_text)
                        self._eval_entropy_metrics(output, label_text)
                else:
                    output = None
        if output is None:
            self.replies['batch_reply'] = None
            return batch_reply

        # noinspection PyUnresolvedReferences
        self.match_batch(batch_reply, batch.valid_indices, output)
        self.replies['batch_reply'] = batch_reply
        self._save_history(observations, batch_reply)  # save model predictions

        return batch_reply

    def is_valid(self, obs):
        normally_valid = super().is_valid(obs)
        if not normally_valid:
            # shortcut boolean evaluation
            return normally_valid
        contains_empties = obs['text_vec'].shape[0] == 0
        if self.is_training and contains_empties:
            warn_once(
                'seq2seq got an empty input sequence (text_vec) during training. '
                'Skipping this example, but you should check your dataset and '
                'preprocessing.'
            )
        elif not self.is_training and contains_empties:
            warn_once(
                'seq2seq got an empty input sequence (text_vec) in an '
                'evaluation example! This may affect your metrics!'
            )
        return not contains_empties
