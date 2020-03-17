import torch

from parlai.agents.seq2seq.seq2seq import Seq2seqAgent
from .criterions import LabelSmoothing, CrossEntropyLabelSmoothing
from .helper import build_loss_desc, build_prob_desc, compute_batch_loss
import torch.nn as nn


class AdaSeq2seqAgent(Seq2seqAgent):

    def build_criterion(self):
        # set up criteria
        if self.opt.get('numsoftmax', 1) > 1:
            self.criterion = nn.NLLLoss(
                ignore_index=self.NULL_IDX, size_average=False)
            self.batch_criterion = LabelSmoothing(
                len(self.dict), self.NULL_IDX)
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, size_average=False)
            self.batch_criterion = CrossEntropyLabelSmoothing(
                len(self.dict), self.NULL_IDX)

        if self.use_cuda:
            self.criterion.cuda()
            self.batch_criterion.cuda()

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