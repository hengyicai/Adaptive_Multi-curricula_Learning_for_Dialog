import random
import torch
from torch.nn import functional as F
from parlai.agents.transformer.transformer import add_common_cmdline_args, TransformerGeneratorModel
from parlai.agents.dialog_evaluator.auto_evaluator import TorchGeneratorWithDialogEvalAgent
from parlai.core.torch_generator_agent import Output
from parlai.core.utils import warn_once, padded_tensor, fp16_optimizer_wrapper
import torch.nn as nn
from .criterions import CrossEntropyLabelSmoothing, CrxEntLabelSmoothing
from .helper import compute_batch_loss, build_prob_desc, build_loss_desc


class AdaTransformerAgent(TorchGeneratorWithDialogEvalAgent):

    def build_model(self, states=None):
        self.model = TransformerGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model.cuda()
        return self.model

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Adaptive Learning Transformer Arguments')
        add_common_cmdline_args(agent)
        cls.dictionary_class().add_cmdline_args(argparser)
        agent.add_argument('--weight_decay', type=float, default=0)
        # ---------------------- For logging ----------------------------------#
        agent.add_argument('--report_freq', type=float, default=0.1)

        agent.add_argument('--label_smoothing', type=float, default=0.0)
        super(AdaTransformerAgent, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        self.id = 'AdaTransformer'

    def build_criterion(self):
        # self.criterion = nn.CrossEntropyLoss(
        #     ignore_index=self.NULL_IDX, reduction='sum'
        # )
        smoothing = self.opt.get('label_smoothing', 0.0)
        assert 0.0 <= smoothing < 1.0, '[ label smoothing value must lie in [0, 1) ! ]'
        if smoothing > 0:
            self.criterion = CrxEntLabelSmoothing(len(self.dict), self.NULL_IDX, smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.NULL_IDX, reduction='sum')
        self.batch_criterion = CrossEntropyLabelSmoothing(
            len(self.dict), self.NULL_IDX)
        if self.use_cuda:
            self.criterion.cuda()
            self.batch_criterion.cuda()

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
        kwargs = {'lr': lr, 'weight_decay': self.opt.get('weight_decay', 0)}
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

    def train_step(self, batch):
        """Train on a single batch of examples."""
        batchsize = batch.text_vec.size(0)
        # helps with memory usage
        self._init_cuda_buffer(batchsize, self.truncate or 256)
        self.model.train()
        self.zero_grad()

        try:
            loss, model_output = self.compute_loss(batch, return_output=True)
            self.metrics['loss'] += loss.item()
            self.backward(loss)
            self.update_params()
            return loss, model_output, \
                   compute_batch_loss(model_output, batch, self.batch_criterion, self.NULL_IDX)
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
                # gradients are synced on backward, now this model is going to be
                # out of sync! catch up with the other workers
                self._init_cuda_buffer(8, 8, True)
            else:
                raise e

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

        # check if there are any labels available, if so we will train on them
        is_training = any('labels' in obs for obs in observations)

        # initialize a list of replies with this agent's id
        batch_reply = [{'id': self.getID(), 'is_training': is_training}
                       for _ in range(batch_size)]

        # create a batch from the vectors
        batch = self.batchify(observations)

        if is_training:
            train_return = self.train_step(batch)
            if train_return is not None:
                _, model_output, batch_loss = train_return
                scores, *_ = model_output
                scores = scores.detach()
                batch_loss = batch_loss.detach()
            else:
                batch_loss = None
                scores = None

            self.replies['batch_reply'] = None
            # TODO: add more model state or training state for sampling the next batch
            #       (learning to teach)
            train_report = self.report()
            loss_desc = build_loss_desc(batch_loss, self.use_cuda)
            prob_desc = build_prob_desc(scores, batch.label_vec, self.use_cuda, self.NULL_IDX)
            for idx, reply in enumerate(batch_reply):
                reply['train_step'] = self._number_training_updates
                reply['train_report'] = train_report
                reply['loss_desc'] = loss_desc
                reply['prob_desc'] = prob_desc
            return batch_reply
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
            else:
                self.match_batch(batch_reply, batch.valid_indices, output)
                self.replies['batch_reply'] = batch_reply
                self._save_history(observations, batch_reply)  # save model predictions
                return batch_reply

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
            maxlen = self.label_truncate or 256
            _, preds, *_ = self.model(
                *self._model_input(batch), bsz=bsz, maxlen=maxlen
            )
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
