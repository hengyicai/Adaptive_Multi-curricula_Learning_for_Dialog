import random
import torch
import torch.nn as nn
import torch.optim as optim
from parlai.core.torch_generator_agent import Output
from parlai.core.utils import round_sigfigs, warn_once, padded_tensor, \
    padded_3d, AttrDict, fp16_optimizer_wrapper
from parlai.core.torch_agent import History
from .modules import DialogWAE_GMP, DialogWAE
from parlai.agents.dialog_evaluator.auto_evaluator import TorchGeneratorWithDialogEvalAgent, CorpusSavedDictionaryAgent
from .criterions import CrossEntropyLabelSmoothing
from .helper import build_loss_desc, build_prob_desc, compute_batch_loss


def make_floor(n):
    floor = [0 for _ in range(n)]
    for i in range(0, n, 2):
        floor[i] = 1
    return floor


class Batch(AttrDict):
    def __init__(self, text_vec=None, text_lengths=None, context_lens=None,
                 floors=None, label_vec=None, label_lengths=None, labels=None,
                 valid_indices=None, candidates=None, candidate_vecs=None,
                 image=None, observations=None, **kwargs):
        super().__init__(
            text_vec=text_vec, text_lengths=text_lengths, context_lens=context_lens,
            floors=floors, label_vec=label_vec, label_lengths=label_lengths, labels=labels,
            valid_indices=valid_indices,
            candidates=candidates, candidate_vecs=candidate_vecs,
            image=image, observations=observations,
            **kwargs)


class PersonDictionaryAgent(CorpusSavedDictionaryAgent):
    def __init__(self, opt, shared=None):
        """Initialize DictionaryAgent."""
        super().__init__(opt, shared)
        if not shared:
            delimiter = opt.get('delimiter', '\n')
            self.add_token(delimiter)
            self.freq[delimiter] = 999999999

            if DialogWaeAgent.P1_TOKEN:
                self.add_token(DialogWaeAgent.P1_TOKEN)

            if DialogWaeAgent.P2_TOKEN:
                self.add_token(DialogWaeAgent.P2_TOKEN)

            if DialogWaeAgent.P1_TOKEN:
                self.freq[DialogWaeAgent.P1_TOKEN] = 999999998

            if DialogWaeAgent.P2_TOKEN:
                self.freq[DialogWaeAgent.P2_TOKEN] = 999999997


class MultiTurnOnOneRowHistory(History):
    def update_history(self, obs, add_next=None):
        """
        Update the history with the given observation.

        :param add_next:
            string to append to history prior to updating it with the
            observation
        """
        coin_flip = 0

        if self.reset_on_next_update:
            # this is the first example in a new episode, clear the previous
            # history
            self.reset()
            self.reset_on_next_update = False

        if add_next is not None:
            if self.add_person_tokens:
                add_next = self._add_person_tokens(
                    add_next, self.p1_token if coin_flip % 2 == 0 else self.p2_token)
                coin_flip += 1
            # update history string
            self._update_strings(add_next)
            # update history vecs
            self._update_vecs(add_next)

        if self.field in obs and obs[self.field] is not None:
            if self.split_on_newln:
                next_texts = obs[self.field].split('\n')
            else:
                next_texts = [obs[self.field]]
            for text in next_texts:
                if self.add_person_tokens:
                    text = self._add_person_tokens(
                        text, self.p1_token if coin_flip % 2 == 0 else self.p2_token)
                    coin_flip += 1
                # update history string
                self._update_strings(text)
                # update history vecs
                self._update_vecs(text)

        if obs.get('episode_done', True):
            # end of this episode, clear the history when we see a new example
            self.reset_on_next_update = True

    def get_history_vec(self):
        """Returns a vectorized version of the history."""
        if len(self.history_vecs) == 0:
            return None

        # if self.vec_type == 'deque':
        #     history = deque(maxlen=self.max_len)
        #     for vec in self.history_vecs[:-1]:
        #         history.extend(vec)
        #         history.extend(self.delimiter_tok)
        #     history.extend(self.history_vecs[-1])
        # else:
        #     # vec type is a list
        #     history = []
        #     for vec in self.history_vecs[:-1]:
        #         history += vec
        #         history += self.delimiter_tok
        #     history += self.history_vecs[-1]
        history = self.history_vecs
        return history


class DialogWaeAgent(TorchGeneratorWithDialogEvalAgent):

    def _set_text_vec(self, obs, history, truncate):
        """
        Sets the 'text_vec' field in the observation.

        Useful to override to change vectorization behavior
        """
        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            # text vec is not precomputed, so we set it using the history
            obs['text'] = history.get_history_str()
            if obs['text'] is not None:
                obs['text_vec'] = history.get_history_vec()

        # check truncation
        if 'text_vec' in obs:
            for idx, vec in enumerate(obs['text_vec']):
                obs['text_vec'][idx] = torch.LongTensor(
                    self._check_truncate(vec, truncate, True)
                )

        return obs

    def batchify(self, obs_batch, sort=False):
        """
        Create a batch of valid observations from an unchecked batch.

        A valid observation is one that passes the lambda provided to the
        function, which defaults to checking if the preprocessed 'text_vec'
        field is present which would have been set by this agent's 'vectorize'
        function.

        Returns a namedtuple Batch. See original definition above for in-depth
        explanation of each field.

        If you want to include additonal fields in the batch, you can subclass
        this function and return your own "Batch" namedtuple: copy the Batch
        namedtuple at the top of this class, and then add whatever additional
        fields that you want to be able to access. You can then call
        super().batchify(...) to set up the original fields and then set up the
        additional fields in your subclass and return that batch instead.

        :param obs_batch:
            List of vectorized observations

        :param sort:
            Default False, orders the observations by length of vectors. Set to
            true when using torch.nn.utils.rnn.pack_padded_sequence.  Uses the text
            vectors if available, otherwise uses the label vectors if available.
        """
        if len(obs_batch) == 0:
            return Batch()

        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if
                     self.is_valid(ex)]

        if len(valid_obs) == 0:
            return Batch()

        valid_inds, exs = zip(*valid_obs)

        # TEXT
        xs, x_lens, context_lens, floors = None, None, None, None
        if any('text_vec' in ex for ex in exs):
            _xs = [ex.get('text_vec', [self.EMPTY]) for ex in exs]
            xs = padded_3d(
                _xs, self.NULL_IDX, self.use_cuda, fp16friendly=self.opt.get('fp16'),
            )
            x_lens = (xs != self.NULL_IDX).sum(dim=-1)  # bsz, context_len
            context_lens = (x_lens != 0).sum(dim=-1)  # bsz
            floors, _ = padded_tensor([make_floor(c_len.item()) for c_len in context_lens],
                                      use_cuda=self.use_cuda)
            # We do not sort on the xs which in the shape of [bsz, context_len, utt_len] is this agent
            # if sort:
            #     sort = False  # now we won't sort on labels
            #     xs, x_lens, valid_inds, exs = argsort(
            #         x_lens, xs, x_lens, valid_inds, exs, descending=True
            #     )

        # LABELS
        labels_avail = any('labels_vec' in ex for ex in exs)
        some_labels_avail = (labels_avail or
                             any('eval_labels_vec' in ex for ex in exs))

        ys, y_lens, labels = None, None, None
        if some_labels_avail:
            field = 'labels' if labels_avail else 'eval_labels'

            label_vecs = [ex.get(field + '_vec', self.EMPTY) for ex in exs]
            labels = [ex.get(field + '_choice') for ex in exs]
            y_lens = [y.shape[0] for y in label_vecs]

            ys, y_lens = padded_tensor(
                label_vecs, self.NULL_IDX, self.use_cuda,
                fp16friendly=self.opt.get('fp16')
            )
            y_lens = torch.LongTensor(y_lens)
            if self.use_cuda:
                y_lens = y_lens.cuda()
            # We do not sort examples in batch for this agent
            # if sort and xs is None:
            #     ys, valid_inds, label_vecs, labels, y_lens = argsort(
            #         y_lens, ys, valid_inds, label_vecs, labels, y_lens,
            #         descending=True
            #     )

        # LABEL_CANDIDATES
        cands, cand_vecs = None, None
        if any('label_candidates_vecs' in ex for ex in exs):
            cands = [ex.get('label_candidates', None) for ex in exs]
            cand_vecs = [ex.get('label_candidates_vecs', None) for ex in exs]

        # IMAGE
        imgs = None
        if any('image' in ex for ex in exs):
            imgs = [ex.get('image', None) for ex in exs]

        return Batch(text_vec=xs, text_lengths=x_lens, context_lens=context_lens,
                     floors=floors, label_vec=ys, label_lengths=y_lens, labels=labels,
                     valid_indices=valid_inds, candidates=cands,
                     candidate_vecs=cand_vecs, image=imgs,
                     observations=exs)

    @classmethod
    def history_class(cls):
        """
        Return the history class that this agent expects to use.

        Can be overriden if a more complex history is required.
        """
        return MultiTurnOnOneRowHistory

    @staticmethod
    def dictionary_class():
        """
        Return the dictionary class that this agent expects to use.

        Can be overriden if a more complex dictionary is required.
        """
        return PersonDictionaryAgent

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('DialogWAE Arguments')
        # Model Arguments
        agent.add_argument('--rnn_class', type=str, default='gru', choices=['gru', 'lstm'])
        agent.add_argument('-esz', '--emb_size', type=int, default=300,
                           help='Size of all embedding layers')
        agent.add_argument('--maxlen', type=int, default=60, help='maximum utterance length')
        agent.add_argument('--hiddensize', type=int, default=512, help='number of hidden units per layer')
        agent.add_argument('--numlayers', type=int, default=2, help='number of layers')
        agent.add_argument('--noise_radius', type=float, default=0.2,
                           help='stdev of noise for autoencoder (regularizer)')
        agent.add_argument('--z_size', type=int, default=200, help='dimension of z (300 performs worse)')
        agent.add_argument('--lambda_gp', type=int, default=10, help='Gradient penalty lambda hyperparameter.')
        agent.add_argument('--temp', type=float, default=1.0, help='softmax temperature (lower --> more discrete)')
        agent.add_argument('--dropout', type=float, default=0.2)
        agent.add_argument('--gmp', type='bool', default=False)
        # -- with the following two arguments, we have model ``DialogWAE_GMP''
        agent.add_argument('--n_prior_components', type=int, default=3)
        agent.add_argument('--gumbel_temp', type=float, default=0.1)
        # -- if hred or vhred to be true, then this model degenerate into the vanilla hred or vhred
        agent.add_argument('--hred', type='bool', default=False)
        agent.add_argument('--vhred', type='bool', default=False)

        # Training Arguments
        agent.add_argument('--n_iters_d', type=int, default=5, help='number of discriminator iterations in training')
        agent.add_argument('--lr_gan_g', type=float, default=5e-05, help='generator learning rate')
        agent.add_argument('--lr_gan_d', type=float, default=1e-05, help='critic/discriminator learning rate')
        agent.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
        agent.add_argument('--clip', type=float, default=1.0, help='gradient clipping, max norm')
        agent.add_argument('--gan_clamp', type=float, default=0.01,
                           help='WGAN clamp (Do not use clamp when you apply gradient penelty')
        agent.add_argument('--norm_z', type='bool', default=False)

        # Evaluation Arguments
        agent.add_argument('--report_freq', type=float, default=0.1)
        cls.dictionary_class().add_cmdline_args(argparser)

        super(DialogWaeAgent, cls).add_cmdline_args(argparser)
        return agent

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        self.id = 'DialogWAE'
        if opt.get('hred', False) and opt.get('vhred', False):
            raise RuntimeError('The flags hred and vhred can not set to be True simultaneously!')

        if not shared:
            self.add_metric('loss_G', 0.0)
            self.add_metric('loss_G_cnt', 0)
            self.add_metric('loss_D', 0.0)
            self.add_metric('loss_D_cnt', 0)
            self.add_metric('kl_loss', 0.0)
            self.add_metric('kl_loss_cnt', 0)
            self.add_metric('bow_loss', 0.0)
            self.add_metric('bow_loss_cnt', 0)

        if (
                # only build an optimizer if we're training
                'train' in opt.get('datatype', '') and
                # and this is the main model, or on every fork if doing hogwild
                (shared is None or self.opt.get('numthreads', 1) > 1)
        ):
            self.optimizer_G = optim.RMSprop(list(self.model.post_net.parameters())
                                             + list(self.model.post_generator.parameters())
                                             + list(self.model.prior_net.parameters())
                                             + list(self.model.prior_generator.parameters()), lr=opt['lr_gan_g'])
            self.optimizer_D = optim.RMSprop(self.model.discriminator.parameters(), lr=opt['lr_gan_d'])

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
        # params = list(self.model.context_encoder.parameters()) + \
        #          list(self.model.post_net.parameters()) + \
        #          list(self.model.post_generator.parameters()) + \
        #          list(self.model.decoder.parameters())
        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}
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

    def build_criterion(self):
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.NULL_IDX,
            reduction='sum'
        )
        self.batch_criterion = CrossEntropyLabelSmoothing(
            len(self.dict), self.NULL_IDX)
        if self.use_cuda:
            self.criterion.cuda()
            self.batch_criterion.cuda()

    def build_model(self, states=None):
        self.opt['n_hidden'] = self.opt['hiddensize']
        self.opt['n_layers'] = self.opt['numlayers']
        special_tokens = [self.START_IDX, self.END_IDX, self.NULL_IDX,
                          self.dict[self.dict.unk_token]]
        if self.opt.get('gmp', False):
            self.model = DialogWAE_GMP(self.opt, len(self.dict), use_cuda=self.use_cuda,
                                       special_tokens=special_tokens)
        else:
            self.model = DialogWAE(self.opt, len(self.dict), use_cuda=self.use_cuda,
                                   special_tokens=special_tokens)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.embedder.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model.cuda()
        return self.model

    def load_state_dict(self, state_dict):
        """
        Load the state dict into model.

        This is easily overridable to facilitate transfer of state dicts.
        """
        self.model.load_state_dict(state_dict, strict=False)

    def report(self):
        base = super().report()
        m = dict()

        if self.metrics['loss_G_cnt'] > 0:
            m['loss_G'] = self.metrics['loss_G'] / self.metrics['loss_G_cnt']
        if self.metrics['loss_D_cnt'] > 0:
            m['loss_D'] = self.metrics['loss_D'] / self.metrics['loss_D_cnt']

        if self.metrics['kl_loss_cnt'] > 0:
            m['kl_loss'] = self.metrics['kl_loss'] / self.metrics['kl_loss_cnt']

        if self.metrics['bow_loss_cnt'] > 0:
            m['bow_loss'] = self.metrics['bow_loss'] / self.metrics['bow_loss_cnt']

        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            base[k] = round_sigfigs(v, 4)
        return base

    def vectorize(self, obs, history, add_start=True, add_end=True,
                  text_truncate=None, label_truncate=None):
        """
        Make vectors out of observation fields and store in the observation.

        In particular, the 'text' and 'labels'/'eval_labels' fields are
        processed and a new field is added to the observation with the suffix
        '_vec'.

        If you want to use additional fields on your subclass, you can override
        this function, call super().vectorize(...) to process the text and
        labels, and then process the other fields in your subclass.

        Additionally, if you want to override some of these default parameters,
        then we recommend using a pattern like:

        .. code-block:: python

          def vectorize(self, *args, **kwargs):
              kwargs['add_start'] = False
              return super().vectorize(*args, **kwargs)


        :param obs:
            Single observation from observe function.

        :param add_start:
            default True, adds the start token to each label.

        :param add_end:
            default True, adds the end token to each label.

        :param text_truncate:
            default None, if set truncates text vectors to the specified
            length.

        :param label_truncate:
            default None, if set truncates label vectors to the specified
            length.

        :return:
            the input observation, with 'text_vec', 'label_vec', and
            'cands_vec' fields added.
        """
        self._set_text_vec(obs, history, text_truncate)
        self._set_label_vec(obs, True, True, label_truncate)
        self._set_label_cands_vec(obs, add_start, add_end, label_truncate)
        return obs

    def reset_metrics(self):
        super().reset_metrics()
        self.metrics['loss_G_cnt'] = 0
        self.metrics['loss_G'] = 0.0
        self.metrics['loss_D_cnt'] = 0
        self.metrics['loss_D'] = 0.0
        self.metrics['kl_loss_cnt'] = 0
        self.metrics['kl_loss'] = 0.0
        self.metrics['bow_loss_cnt'] = 0
        self.metrics['bow_loss'] = 0.0

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
            # noinspection PyTypeChecker
            train_return = self.train_step(batch)
            if train_return is not None:
                model_output, batch_loss = train_return
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
            # noinspection PyUnresolvedReferences
            prob_desc = build_prob_desc(scores, batch.label_vec[:, 1:], self.use_cuda, self.NULL_IDX)
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

    def train_step(self, batch):
        """Train on a single batch of examples."""
        # helps with memory usage
        # self._init_cuda_buffer(batchsize, self.truncate or 256)

        self.model.train()
        # self.zero_grad()  # zero_grad left be done in the model code

        try:
            # loss = self.compute_loss(batch)
            # self.metrics['loss'] += loss.item()
            # self.backward(loss)
            # self.update_params()
            # with torch.autograd.detect_anomaly():
            loss_AE = self.model.train_AE(batch.text_vec, batch.context_lens, batch.text_lengths,
                                          batch.floors, batch.label_vec, batch.label_lengths,
                                          self.optimizer, self.criterion, self._number_training_updates)
            self._number_training_updates += 1
            self.metrics['correct_tokens'] += loss_AE['correct_tokens']
            self.metrics['nll_loss'] += loss_AE['nll_loss']
            self.metrics['num_tokens'] += loss_AE['num_tokens']
            self.metrics['loss'] += loss_AE['avg_loss']

            if self.opt.get('hred', False):
                pass
            elif self.opt.get('vhred', False):
                vhred_kl_loss = loss_AE['vhred_kl_loss']
                bow_loss = loss_AE['bow_loss']
                self.metrics['kl_loss_cnt'] += 1
                self.metrics['kl_loss'] += vhred_kl_loss
                self.metrics['bow_loss_cnt'] += 1
                self.metrics['bow_loss'] += bow_loss
            else:
                loss_G = self.model.train_G(batch.text_vec, batch.context_lens, batch.text_lengths,
                                            batch.floors, batch.label_vec, batch.label_lengths,
                                            self.optimizer_G)
                self.metrics['loss_G_cnt'] += 1
                self.metrics['loss_G'] += loss_G['train_loss_G']

                for i in range(self.opt['n_iters_d']):
                    loss_D = self.model.train_D(batch.text_vec, batch.context_lens, batch.text_lengths,
                                                batch.floors, batch.label_vec, batch.label_lengths,
                                                self.optimizer_D)
                    if i == 0:
                        self.metrics['loss_D_cnt'] += 1
                        self.metrics['loss_D'] += loss_D['train_loss_D']

            model_output = (loss_AE['scores'], loss_AE['preds'])
            return model_output, compute_batch_loss(model_output, batch, self.batch_criterion, self.NULL_IDX,
                                                    drop_first=True)
        except RuntimeError as e:
            # catch out of memory exceptions during fwd/bck (skip batch)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch. '
                      'if this happens frequently, decrease batchsize or '
                      'truncate the inputs to the model.')
                self.metrics['total_skipped_batches'] += 1
                # gradients are synced on backward, now this model is going to be
                # out of sync! catch up with the other workers
                # self._init_cuda_buffer(8, 8, True)
            else:
                raise e

    def eval_step(self, batch):
        """Evaluate a single batch of examples."""
        if batch.text_vec is None:
            return
        self.model.eval()

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            # loss = self.compute_loss(batch)  # noqa: F841  we need the side effects
            # self.metrics['loss'] += loss.item()
            valid_loss = self.model.valid(
                batch.text_vec, batch.context_lens, batch.text_lengths,
                batch.floors, batch.label_vec, batch.label_lengths, self.criterion
            )
            self.metrics['correct_tokens'] += valid_loss['correct_tokens']
            self.metrics['nll_loss'] += valid_loss['nll_loss']
            self.metrics['num_tokens'] += valid_loss['num_tokens']
            self.metrics['loss'] += valid_loss['avg_loss']

            if self.opt.get('hred', False):
                pass
            elif self.opt.get('vhred', False):
                self.metrics['kl_loss_cnt'] += 1
                self.metrics['kl_loss'] += valid_loss['vhred_kl_loss']
                self.metrics['bow_loss_cnt'] += 1
                self.metrics['bow_loss'] += valid_loss['bow_loss']
            else:
                self.metrics['loss_G_cnt'] += 1
                self.metrics['loss_G'] += valid_loss['valid_loss_G']
                self.metrics['loss_D_cnt'] += 1
                self.metrics['loss_D'] += valid_loss['valid_loss_D']

        preds = None
        if self.skip_generation:
            # noinspection PyTypeChecker
            warn_once(
                "--skip-generation does not produce accurate metrics beyond ppl",
                RuntimeWarning
            )
        else:
            sample_words, sample_lens = self.model.sample(
                batch.text_vec, batch.context_lens, batch.text_lengths,
                batch.floors, self.START_IDX, self.END_IDX
            )
            preds = torch.from_numpy(sample_words)
        if batch.label_vec is not None:
            label_text = batch.labels
            # we are in the validation mode, print some generated responses for debugging
            for i in range(len(preds)):
                if random.random() > (1 - self.opt['report_freq']):
                    context_text = batch.observations[i]['text']
                    print('TEXT: ', context_text)
                    print('TARGET: ', label_text[i])
                    print('PREDICTION: ', self._v2t(preds[i]), '\n~')
        else:
            label_text = None

        context = [obs['text'] for obs in batch.observations]
        text = [self._v2t(p) for p in preds] if preds is not None else None
        return Output(text, None), label_text, context