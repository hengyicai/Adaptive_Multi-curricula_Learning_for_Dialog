#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""adaptive learning task agents.

This is specified in the following way:
--task adaptive_learning:{task_name}:{score_func}
...where {task_name} is one of...
    - personachat
...where {score_func} is one of...
    - avgnidf
    - lastuttsim
"""

import os
import math
import json
import copy
import torch
import random
import numpy as np
from torch import optim
import scipy.stats as ss
import torch.nn.functional as F
from collections import OrderedDict
from .modules import PolicyNet, CriticNet
from torch.distributions import Categorical
from parlai.core.logs import TensorboardLogger
from parlai.core.utils import Timer, round_sigfigs
from parlai.core.teachers import FbDialogTeacher, DialogData


def _path(opt, task_name, score_func):
    # Build the data if it doesn't exist.
    dt = opt['datatype'].split(':')[0]
    _file_path = os.path.join(opt['datapath'], 'AdaptiveLearning', task_name, score_func, dt + '.txt')
    assert os.path.exists(_file_path), 'Your dataset {} does dot exist!'.format(_file_path)
    return _file_path


def get_init_teacher(opt, shared):
    init_teacher = None
    if not shared:
        model_file = opt.get('model_file')
        if model_file and os.path.isfile(model_file + '.teacher'):
            init_teacher = model_file + '.teacher'
    return init_teacher


class DefaultTeacher(FbDialogTeacher):

    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        super().__init__(opt, shared)
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        self.is_combine_attr = (hasattr(self, 'other_task_datafiles') and self.other_task_datafiles)
        self.random_policy = opt.get('random_policy', False)
        self.count_sample = opt.get('count_sample', False)
        self.anti = opt.get('anti', False)

        if self.random_policy:
            random.seed(17)

        if not shared:
            if not self.stream and opt.get('pace_by', 'sample') == 'bucket':
                score_list = [episode[0][2] for episode in self.data.data]
                assert score_list == sorted(score_list)
                num_buckets = opt.get('num_buckets', int(self.num_episodes() / 10))
                lb_indices = [int(len(score_list) * i / num_buckets) for i in range(num_buckets)]
                lbs = [score_list[idx] for idx in lb_indices]
                bucket_ids = [self.sort_into_bucket(ctrl_val, lbs) for ctrl_val in score_list]
                bucket_cnt = [0 for _ in range(num_buckets)]
                for i in range(num_buckets):
                    bucket_cnt[i] = bucket_ids.count(i)
                self.bucket_cnt = bucket_cnt
            self.lastYs = [None] * self.bsz
            # build multiple task data
            self.tasks = [self.data]

            if self.is_combine_attr:
                print('[ build multiple task data ... ]')
                for datafile in self.other_task_datafiles:
                    task_opt = copy.deepcopy(opt)
                    task_opt['datafile'] = datafile
                    self.tasks.append(
                        DialogData(task_opt, data_loader=self.setup_data,
                                   cands=self.label_candidates())
                    )
                print('[ build multiple task data done! ]')

                # record the selections of each subtasks
                self.subtasks = opt['subtasks'].split(':')
                self.subtask_counter = OrderedDict()
                self.p_selections = OrderedDict()
                self.c_selections = OrderedDict()
                for t in self.subtasks:
                    self.subtask_counter[t] = 0
                    self.p_selections[t] = []
                    self.c_selections[t] = []

                if self.count_sample and not self.stream:
                    self.sample_counter = OrderedDict()
                    for idx, t in enumerate(self.subtasks):
                        self.sample_counter[t] = [0 for _ in self.tasks[idx].data]

            # setup the tensorboard log
            if opt['tensorboard_log_teacher'] is True:
                opt['tensorboard_tag'] = 'task'
                teacher_metrics = 'reward,policy_loss,critic_loss,mean_advantage_reward,action_ent'.split(',')
                opt['tensorboard_metrics'] = ','.join(opt['tensorboard_metrics'].split(',') + teacher_metrics)
                self.writer = TensorboardLogger(opt)

        else:
            self.lastYs = shared['lastYs']
            self.tasks = shared['tasks']
            if not self.stream and opt.get('pace_by', 'sample') == 'bucket':
                self.bucket_cnt = shared['bucket_cnt']
            if 'writer' in shared:
                self.writer = shared['writer']
            if 'subtask_counter' in shared:
                self.subtask_counter = shared['subtask_counter']
            if 'p_selections' in shared:
                self.p_selections = shared['p_selections']
            if 'c_selections' in shared:
                self.c_selections = shared['c_selections']

        # build the policy net, criterion and optimizer here
        self.state_dim = 32 + len(self.tasks)  # hand-craft features
        self.action_dim = len(self.tasks)

        if not shared:
            self.policy = PolicyNet(self.state_dim, self.action_dim)
            self.critic = CriticNet(self.state_dim, self.action_dim)

            init_teacher = get_init_teacher(opt, shared)
            if init_teacher is not None:
                # load teacher parameters if available
                print('[ Loading existing teacher params from {} ]'
                      ''.format(init_teacher))
                states = self.load(init_teacher)
            else:
                states = {}
        else:
            self.policy = shared['policy']
            self.critic = shared['critic']
            states = shared['states']

        if (
                # only build an optimizer if we're training
                'train' in opt.get('datatype', '') and
                # and this is the main model
                shared is None
        ):
            # for policy net
            self.optimizer = self.init_optim(
                [p for p in self.policy.parameters() if p.requires_grad],
                lr=opt['learningrate_teacher'],
                optim_states=states.get('optimizer'),
                saved_optim_type=states.get('optimizer_type')
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                'min',
                factor=0.8,  # 0.5 --> 0.8
                patience=5,  # 3 -- > 5
                verbose=True
            )
            if 'lr_scheduler' in states:
                self.scheduler.load_state_dict(states['lr_scheduler'])

            # for critic net
            self.optimizer_critic = self.init_optim(
                [p for p in self.critic.parameters() if p.requires_grad],
                lr=opt['learningrate_teacher_critic'],
                optim_states=states.get('optimizer_critic'),
                saved_optim_type=states.get('optimizer_type')
            )
            self.scheduler_critic = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_critic,
                'min',
                factor=0.8,  # 0.5 --> 0.8
                patience=5,  # 3 -- > 5
                verbose=True
            )
            if 'lr_scheduler_critic' in states:
                self.scheduler_critic.load_state_dict(states['lr_scheduler_critic'])

            self.critic_criterion = torch.nn.SmoothL1Loss()

        self.reward_metric = opt.get('reward_metric', 'total_metric')
        self.reward_metric_mode = opt.get('reward_metric_mode', 'max')

        self.prev_prev_valid_report = states['prev_prev_valid_report'] if 'prev_prev_valid_report' in states else None
        self.prev_valid_report = states['prev_valid_report'] if 'prev_valid_report' in states else None
        self.current_valid_report = states['current_valid_report'] if 'current_valid_report' in states else None
        self.saved_actions = states['saved_actions'] if 'saved_actions' in states else OrderedDict()
        self.saved_state_actions = states['saved_state_actions'] if 'saved_state_actions' in states else OrderedDict()
        if self.use_cuda:
            for k, v in self.saved_actions.items():
                self.saved_actions[k] = v.cuda()
            for k, v in self.saved_state_actions.items():
                self.saved_state_actions[k] = v.cuda()
        self._number_teacher_updates = states['_number_teacher_updates'] if '_number_teacher_updates' in states else 0

        # enable the batch_act
        self.use_batch_act = self.bsz > 1

        self.T = self.opt.get('T', 1000)
        self.c0 = self.opt.get('c0', 0.01)
        self.p = self.opt.get('p', 2)

        # setup the timer
        self.log_every_n_secs = opt['log_every_n_secs'] if opt['log_every_n_secs'] > 0 \
            else float('inf')
        self.action_log_time = Timer()

        self.move_to_cuda()

    def move_to_cuda(self):
        if self.use_cuda:
            self.policy.cuda()
            self.critic.cuda()

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, lr, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with teacher parameters.

        :param params:
            parameters from the teacher

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        kwargs = {'lr': lr}
        if opt.get('momentum_teacher') > 0 and opt['optimizer_teacher'] in ['sgd', 'rmsprop', 'qhm']:
            # turn on momentum for optimizers that use it
            kwargs['momentum'] = opt['momentum_teacher']
            if opt['optimizer_teacher'] == 'sgd' and opt.get('nesterov_teacher', True):
                # for sgd, maybe nesterov
                kwargs['nesterov'] = opt.get('nesterov_teacher', True)
            elif opt['optimizer_teacher'] == 'qhm':
                # qhm needs a nu
                kwargs['nu'] = opt.get('nus_teacher', (0.7,))[0]
        elif opt['optimizer_teacher'] == 'adam':
            # turn on amsgrad for adam
            # amsgrad paper: https://openreview.net/forum?id=ryQu7f-RZ
            kwargs['amsgrad'] = True
        elif opt['optimizer_teacher'] == 'qhadam':
            # set nus for qhadam
            kwargs['nus'] = opt.get('nus_teacher', (0.7, 1.0))
        if opt['optimizer_teacher'] in ['adam', 'sparseadam', 'adamax', 'qhadam']:
            # set betas for optims that use it
            kwargs['betas'] = opt.get('betas_teacher', (0.9, 0.999))

        optim_class = self.optim_opts()[opt['optimizer_teacher']]
        optimizer = optim_class(params, **kwargs)

        if optim_states:
            if saved_optim_type != opt['optimizer_teacher']:
                print('WARNING: not loading optim state since optim class '
                      'changed.')
            else:
                try:
                    optimizer.load_state_dict(optim_states)
                except ValueError:
                    print('WARNING: not loading optim state since model '
                          'params changed.')
                if self.use_cuda:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
        return optimizer

    def load(self, path):
        """
        Return opt and teacher states.

        TODO: load behaviors should be consistent with function state_dict().
        """
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        if 'policy' in states:
            self.policy.load_state_dict(states['policy'])
        if 'critic' in states:
            self.critic.load_state_dict(states['critic'])
        if 'optimizer' in states and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(states['optimizer'])
        if 'optimizer_critic' in states and hasattr(self, 'optimizer_critic'):
            self.optimizer_critic.load_state_dict(states['optimizer_critic'])
        return states

    def share(self):
        shared = super().share()
        if hasattr(self, 'bucket_cnt'):
            shared['bucket_cnt'] = self.bucket_cnt

        shared['tasks'] = self.tasks
        shared['policy'] = self.policy
        shared['critic'] = self.critic

        shared['states'] = {
            'optimizer_type': self.opt['optimizer_teacher'],
            'prev_prev_valid_report': self.prev_prev_valid_report,
            'prev_valid_report': self.prev_valid_report,
            'current_valid_report': self.current_valid_report,
            'saved_actions': self.saved_actions,
            'saved_state_actions': self.saved_state_actions,
        }
        if hasattr(self, 'writer'):
            shared['writer'] = self.writer
        if hasattr(self, 'subtask_counter'):
            shared['subtask_counter'] = self.subtask_counter
        if hasattr(self, 'p_selections'):
            shared['p_selections'] = self.p_selections
        if hasattr(self, 'c_selections'):
            shared['c_selections'] = self.c_selections
        return shared

    @staticmethod
    def sort_into_bucket(val, bucket_lbs):
        """
        Returns the highest bucket such that val >= lower bound for that bucket.

        Inputs:
          val: float. The value to be sorted into a bucket.
          bucket_lbs: list of floats, sorted ascending.

        Returns:
          bucket_id: int in range(num_buckets); the bucket that val belongs to.
        """
        num_buckets = len(bucket_lbs)
        for bucket_id in range(num_buckets - 1, -1, -1):  # iterate descending
            lb = bucket_lbs[bucket_id]
            if val >= lb:
                return bucket_id
        raise ValueError('val %f is not >= any of the lower bounds: %s' % (val, bucket_lbs))

    def pace_function(self, states, sum_num, T=1000, c0=0.01, p=2):
        train_step = states['train_step']
        progress = self.root_p_pace(train_step, T, c0, p)
        return int(sum_num * progress)

    @staticmethod
    def root_p_pace(timestep, T=1000, c0=0.01, p=2):
        root_p = math.pow(timestep * (1 - math.pow(c0, p)) / T + math.pow(c0, p), 1.0 / p)
        return min(1.0, root_p)

    def act(self, observation=None, task_idx=0):
        """Send new dialog message."""
        if not hasattr(self, 'epochDone'):
            # reset if haven't yet
            self.reset()

        # get next example, action is episode_done dict if already out of exs
        action, self.epochDone = self.next_example(observation=observation, task_idx=task_idx)
        action['id'] = self.getID()

        # remember correct answer if available
        self.lastY = action.get('labels', action.get('eval_labels', None))
        if ((not self.datatype.startswith('train') or 'evalmode' in self.datatype) and
                'labels' in action):
            # move labels to eval field so not used for training
            # but this way the model can use the labels for perplexity or loss
            action = action.copy()
            labels = action.pop('labels')
            if not self.opt.get('hide_labels', False):
                action['eval_labels'] = labels

        return action

    def _cry_for_missing_in_obs(self, something):
        raise RuntimeError(
            "{} is needed to include in observations to build states!".format(something)
        )

    def _build_states(self, observations):
        for key in ['train_step', 'train_report', 'loss_desc', 'prob_desc']:
            if key not in observations[0]:
                self._cry_for_missing_in_obs(key)

        train_step = observations[0]['train_step']  # scala
        train_step = min(train_step / self.T, 1)
        train_report = observations[0]['train_report']
        nll_loss = train_report.get('nll_loss', 0) / 10  # scala
        loss_desc = observations[0]['loss_desc']
        loss_desc = F.normalize(loss_desc, p=2, dim=-1)

        prob_desc = observations[0]['prob_desc']
        prob_desc = F.normalize(prob_desc, p=2, dim=-1)

        if hasattr(self, 'subtask_counter'):
            subtask_progress = self.subtask_counter.values()
            max_min = max(subtask_progress) - min(subtask_progress)
            subtask_progress = [(item - min(subtask_progress)) / max_min
                                if max_min > 0 else 0 for item in subtask_progress]
        else:
            subtask_progress = [0]
        subtask_progress = torch.FloatTensor(subtask_progress)
        if self.use_cuda:
            subtask_progress = subtask_progress.cuda()

        prev_valid_report = self.prev_valid_report
        if prev_valid_report is None:
            prev_valid_report = {}

        bleu = prev_valid_report.get('bleu', 0)
        valid_nll_loss = prev_valid_report.get('nll_loss', 0) / 10
        dist_1_ratio = prev_valid_report.get('dist_1_ratio', 0)
        dist_2_ratio = prev_valid_report.get('dist_2_ratio', 0)
        dist_3_ratio = prev_valid_report.get('dist_3_ratio', 0)
        embed_avg = prev_valid_report.get('embed_avg', 0)
        embed_greedy = prev_valid_report.get('embed_greedy', 0)
        embed_extrema = prev_valid_report.get('embed_extrema', 0)
        embed_coh = prev_valid_report.get('embed_coh', 0)
        intra_dist_1 = prev_valid_report.get('intra_dist_1', 0) / 10
        intra_dist_2 = prev_valid_report.get('intra_dist_2', 0) / 10
        intra_dist_3 = prev_valid_report.get('intra_dist_3', 0) / 10
        response_length = prev_valid_report.get('response_length', 0) / self.opt.get('label_truncate', 100)
        # sent_entropy_uni = prev_valid_report.get('sent_entropy_uni', 0) / 100
        # sent_entropy_bi = prev_valid_report.get('sent_entropy_bi', 0) / 100
        # sent_entropy_tri = prev_valid_report.get('sent_entropy_tri', 0) / 100
        word_entropy_uni = prev_valid_report.get('word_entropy_uni', 0) / 100
        word_entropy_bi = prev_valid_report.get('word_entropy_bi', 0) / 100
        word_entropy_tri = prev_valid_report.get('word_entropy_tri', 0) / 100
        states = torch.FloatTensor([train_step, nll_loss, bleu, valid_nll_loss,
                                    dist_1_ratio, dist_2_ratio, dist_3_ratio,
                                    embed_avg, embed_greedy, embed_extrema, embed_coh,
                                    intra_dist_1, intra_dist_2, intra_dist_3, response_length,
                                    # sent_entropy_uni, sent_entropy_bi, sent_entropy_tri,
                                    word_entropy_uni, word_entropy_bi, word_entropy_tri])
        if self.use_cuda:
            states = states.cuda()
        states = torch.cat([states, loss_desc, prob_desc, subtask_progress], dim=-1).unsqueeze(dim=0)
        return states

    def __uniform_weights(self):
        w = 1 / len(self.tasks)
        weights = torch.FloatTensor([w] * len(self.tasks))
        if self.use_cuda:
            weights = weights.cuda()
        return weights.unsqueeze(dim=0)

    def __load_training_batch(self, observations):
        if observations and len(observations) > 0 and observations[0] and self.is_combine_attr:
            if not self.random_policy:
                with torch.no_grad():
                    current_states = self._build_states(observations)
                action_probs = self.policy(current_states)
                if self.action_log_time.time() > self.log_every_n_secs and len(self.tasks) > 1:
                    with torch.no_grad():
                        # log the action distributions
                        action_p = ','.join([str(round_sigfigs(x, 4)) for x in action_probs[0].data.tolist()])
                        log = '[ {} {} ]'.format('Action probs:', action_p)
                        print(log)
                        self.action_log_time.reset()
                sample_from = Categorical(action_probs[0])
                action = sample_from.sample()
                train_step = observations[0]['train_step']
                self.saved_actions[train_step] = sample_from.log_prob(action)
                self.saved_state_actions[train_step] = torch.cat([current_states, action_probs], dim=1)
                selected_task = action.item()
                self.subtask_counter[self.subtasks[selected_task]] += 1

                probs = action_probs[0].tolist()
                selection_report = {}
                for idx, t in enumerate(self.subtasks):
                    selection_report['p_{}'.format(t)] = probs[idx]
                    self.p_selections[t].append(probs[idx])
                    selection_report['c_{}'.format(t)] = self.subtask_counter[t]
                    self.c_selections[t].append(self.subtask_counter[t])
                self.writer.add_metrics(setting='Teacher/task_selection', step=train_step, report=selection_report)
            else:
                selected_task = random.choice(range(len(self.tasks)))
                self.subtask_counter[self.subtasks[selected_task]] += 1
        else:
            selected_task = 0

        return self.__load_batch(observations, task_idx=selected_task)

    def __load_batch(self, observations, task_idx=0):
        if observations is None:
            observations = [None] * self.bsz
        bsz = len(observations)

        batch = []
        # Sample from multiple tasks using the policy net
        for idx in range(bsz):
            batch.append(self.act(observations[idx], task_idx=task_idx))
        return batch

    def batch_act(self, observations):
        """
        Returns an entire batch of examples instead of just one.
        """
        if not hasattr(self, 'epochDone'):
            # reset if haven't yet
            self.reset()
        if self.opt['datatype'] == 'train':
            batch = self.__load_training_batch(observations)
        else:
            batch = self.__load_batch(observations)

        # pad batch
        if len(batch) < self.bsz:
            batch += [
                         {'episode_done': True, 'id': self.getID()}
                     ] * (self.bsz - len(batch))

        # remember correct answer if available (for padding, None)
        for i, ex in enumerate(batch):
            if 'labels' in ex:
                labels = ex['labels']
                self.lastYs[i] = labels
                if not self.datatype.startswith('train') or 'evalmode' in self.datatype:
                    del ex['labels']
                    if not self.opt.get('hide_labels', False):
                        ex['eval_labels'] = labels
            else:
                self.lastYs[i] = ex.get('eval_labels', None)

        return batch

    def next_example(self, observation=None, task_idx=0):
        """
        Returns the next example.

        If there are multiple examples in the same episode, returns the next
        one in that episode. If that episode is over, gets a new episode index
        and returns the first example of that episode.
        """
        if self.stream:
            action, epoch_done = self.tasks[task_idx].get()
        else:
            if self.episode_done:
                self.episode_idx = self.next_episode_idx()
                self.entry_idx = 0
            else:
                self.entry_idx += 1

            if self.episode_idx >= self.num_episodes():
                return {'episode_done': True}, True

            if observation is None or self.opt['datatype'] != 'train':
                # The first step of the training or validation mode
                sampled_episode_idx = self.episode_idx
                sampled_entry_idx = self.entry_idx
            else:
                # --------------- pick the sample according to the pace function -----------
                pace_by = self.opt.get('pace_by', 'sample')

                if pace_by == 'sample':
                    sum_num = self.num_episodes()
                elif pace_by == 'bucket':
                    sum_num = len(self.bucket_cnt)
                else:
                    raise ValueError('pace_by must be {} or {}!'.format('sample', 'bucket'))

                states4pace_func = observation
                if hasattr(self, 'subtask_counter'):
                    states4pace_func = {'train_step': self.subtask_counter[self.subtasks[task_idx]]}

                threshold = self.pace_function(states4pace_func, sum_num, self.T, self.c0, self.p)
                if pace_by == 'sample':
                    stop_step = threshold
                elif pace_by == 'bucket':
                    stop_step = sum(self.bucket_cnt[:threshold])
                else:
                    raise ValueError('pace_by must be {} or {}!'.format('sample', 'bucket'))

                stop_step = self.num_episodes() if stop_step > self.num_episodes() else stop_step
                # sampled_episode_idx = random.choice(list(range(self.num_episodes()))[:stop_step])
                sampled_episode_idx = np.random.choice(stop_step)
                sampled_entry_idx = 0  # make sure the episode only contains one entry

                if self.anti:
                    sampled_episode_idx = self.num_episodes() - 1 - sampled_episode_idx

            if self.count_sample:
                self.sample_counter[self.subtasks[task_idx]][sampled_episode_idx] += 1

            ex = self.get(sampled_episode_idx, sampled_entry_idx, task_idx=task_idx)

            if observation is None or self.opt['datatype'] != 'train':
                self.episode_done = ex.get('episode_done', False)
                if (not self.random and self.episode_done and
                        self.episode_idx + self.opt.get("batchsize", 1) >= self.num_episodes()):
                    epoch_done = True
                else:
                    epoch_done = False
            else:
                # in the setting of curriculum leaning, samples are not uniformly
                # picked from the training set, so, the epoch records here make no sense.
                epoch_done = False

            action = ex

        return action, epoch_done

    def get(self, episode_idx, entry_idx=0, task_idx=0):
        return self.tasks[task_idx].get(episode_idx, entry_idx)[0]

    def update_params(self):
        self._number_teacher_updates += 1
        if self.opt.get('gradient_clip_teacher', -1) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.opt['gradient_clip_teacher']
            )

        self.optimizer.step()

    def update_critic_params(self):
        if self.opt.get('gradient_clip_teacher', -1) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.opt['gradient_clip_teacher']
            )

        self.optimizer_critic.step()

    def receive_metrics(self, metrics_dict):
        if self.is_combine_attr and not self.random_policy:
            assert self.reward_metric in metrics_dict, '{} is not in the metrics_dict!'.format(self.reward_metric)
            self.prev_prev_valid_report = self.prev_valid_report
            self.prev_valid_report = self.current_valid_report
            self.current_valid_report = metrics_dict
            delt_reward = None
            if self.prev_prev_valid_report and self.prev_valid_report and self.current_valid_report:
                delt_reward1 = self.current_valid_report[self.reward_metric] - self.prev_valid_report[
                    self.reward_metric]
                delt_reward0 = self.prev_valid_report[self.reward_metric] - self.prev_prev_valid_report[
                    self.reward_metric]
                if self.reward_metric_mode == 'min':
                    delt_reward1 = -delt_reward1
                    delt_reward0 = -delt_reward0
                delt_reward = delt_reward1 / (delt_reward0 + 1e-6) - 1
            if delt_reward and len(self.saved_actions) > 0 and len(self.saved_state_actions) > 0:
                reward = torch.clamp(torch.FloatTensor([delt_reward]), -10, 10)
                if self.use_cuda:
                    reward = reward.cuda()

                with torch.no_grad():
                    batch_state_actions = torch.cat(list(self.saved_state_actions.values()), dim=0)
                    if self.use_cuda:
                        batch_state_actions = batch_state_actions.cuda()
                    estimate_rewards = self.critic(batch_state_actions).squeeze()
                    advantages = reward - estimate_rewards

                    # rescale the rewards by ranking
                    episode_len = len(advantages)
                    ranks = torch.FloatTensor(list(reversed(ss.rankdata(advantages.cpu(), method='dense')))).unsqueeze(
                        dim=1)
                    rescaled_rewards = torch.sigmoid(12 * (0.5 - ranks / episode_len))

                rescaled_rewards = [r.item() for r in rescaled_rewards]
                policy_loss = []
                idx = 0
                for model_train_step, log_prob in self.saved_actions.items():
                    policy_loss.append(-log_prob.unsqueeze(dim=0) * rescaled_rewards[idx])
                    idx += 1
                policy_loss = torch.cat(policy_loss).sum()

                # regularization term regarding action distribution
                bsz = batch_state_actions.size(0)
                action_probs = torch.cat(list(self.saved_state_actions.values()), dim=0).narrow(1, self.state_dim,
                                                                                                self.action_dim)
                action_ent = torch.sum(-action_probs * torch.log(action_probs)) / bsz

                self.policy.train()
                self.optimizer.zero_grad()
                policy_loss = policy_loss + self.opt.get('reg_action', 0.001) * (-action_ent)
                policy_loss.backward()
                self.update_params()

                # lr_scheduler step on teacher loss
                policy_loss_item = policy_loss.item()
                if self.opt.get('optimizer_teacher', '') == 'sgd':
                    self.scheduler.step(policy_loss_item)

                # training on the critic
                self.critic.train()
                self.optimizer_critic.zero_grad()

                batch_values = self.critic(batch_state_actions)
                critic_target = torch.FloatTensor(bsz, 1)
                critic_target = critic_target.fill_(reward.item())
                if self.use_cuda:
                    critic_target = critic_target.cuda()
                critic_loss = self.critic_criterion(batch_values, critic_target)
                critic_loss.backward()
                self.update_critic_params()
                critic_loss_item = critic_loss.item()
                if self.opt.get('optimizer_teacher', '') == 'sgd':
                    self.scheduler_critic.step(critic_loss_item)

                # log something
                print(
                    '[ reward: {}; mean_advantage_reward: {}; policy loss: {};'
                    ' critic loss: {}; action ent: {}; episode length: {} ]'.format(
                        reward.item(), np.mean(advantages.tolist()), policy_loss_item,
                        critic_loss_item, action_ent.item(), len(self.saved_actions)))

                report = {
                    'reward': reward.item(),
                    'mean_advantage_reward': np.mean(advantages.tolist()),
                    'policy_loss': policy_loss_item,
                    'critic_loss': critic_loss_item,
                    'action_ent': action_ent.item(),
                }
                self.writer.add_metrics(setting='Teacher/receive_metrics', step=self._number_teacher_updates,
                                        report=report)
                # clear history actions
                self.saved_actions.clear()
                self.saved_state_actions.clear()

    def state_dict(self):
        """
        Get the state dict for saving

        TODO: save more teacher-related states for reloading
        """
        states = {}
        if hasattr(self, 'policy'):  # save model params
            if hasattr(self.policy, 'module'):
                # did we wrap in a DistributedDataParallel
                states['policy'] = self.policy.module.state_dict()
            else:
                states['policy'] = self.policy.state_dict()

        if hasattr(self, 'critic'):  # save model params
            if hasattr(self.critic, 'module'):
                # did we wrap in a DistributedDataParallel
                states['critic'] = self.critic.module.state_dict()
            else:
                states['critic'] = self.critic.state_dict()

        if hasattr(self, 'optimizer'):  # save optimizer params
            states['optimizer'] = self.optimizer.state_dict()
            states['optimizer_type'] = self.opt['optimizer_teacher']
        if hasattr(self, 'optimizer_critic'):
            states['optimizer_critic'] = self.optimizer_critic.state_dict()

        if getattr(self, 'scheduler', None):
            states['lr_scheduler'] = self.scheduler.state_dict()
        if getattr(self, 'scheduler_critic', None):
            states['lr_scheduler_critic'] = self.scheduler_critic.state_dict()

        states['prev_prev_valid_report'] = self.prev_prev_valid_report
        states['prev_valid_report'] = self.prev_valid_report
        states['current_valid_report'] = self.current_valid_report
        states['saved_actions'] = self.saved_actions
        states['saved_state_actions'] = self.saved_state_actions

        states['_number_teacher_updates'] = self._number_teacher_updates

        return states

    def save(self, path=None):
        if path:
            teacher_path = path
        else:
            model_file = self.opt.get('model_file', None)
            if model_file:
                teacher_path = model_file + '.teacher'
            else:
                teacher_path = None

        if teacher_path:
            states = self.state_dict()
            if states:
                with open(teacher_path, 'wb') as write:
                    torch.save(states, write)
                # save opt file
                with open(teacher_path + '.opt', 'w', encoding='utf-8') as handle:
                    json.dump(self.opt, handle)
                    # for convenience of working with jq, make sure there's a newline
                    handle.write('\n')

            if self.count_sample:
                # save sample count info
                for task_name, task_val in self.sample_counter.items():
                    with open(teacher_path + '.sample_count.{}'.format(task_name), 'w', encoding='utf-8') as f:
                        f.write('\n'.join([str(item) for item in task_val]))

            self.write_selections('p_selections', teacher_path)
            self.write_selections('c_selections', teacher_path)

    def write_selections(self, selections, teacher_path):
        if hasattr(self, selections):
            with open(teacher_path + '.{}'.format(selections), 'w', encoding='utf-8') as f:
                f.write('\t'.join(self.subtasks))
                f.write('\n')
                for idx in range(len(getattr(self, selections)[self.subtasks[0]])):
                    p_line = []
                    for t in self.subtasks:
                        p_line.append(str(getattr(self, selections)[t][idx]))
                    f.write('\t'.join(p_line))
                    f.write('\n')


class PersonachatH3Teacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        assert 'subtasks' in opt, 'subtasks must be specified!'
        subtasks = opt['subtasks'].split(':')
        opt['datafile'] = _path(opt, 'personachat_history3', subtasks[0])
        other_task_datafiles = []
        for attr in subtasks[1:]:
            other_task_datafiles.append(
                _path(opt, 'personachat_history3', attr)
            )
        self.other_task_datafiles = other_task_datafiles
        super().__init__(opt, shared)


class PersonachatH3OriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'personachat_history3', 'original')
        super().__init__(opt, shared)


class PersonachatH3SparseTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        assert 'subtasks' in opt, 'subtasks must be specified!'
        subtasks = opt['subtasks'].split(':')
        opt['datafile'] = _path(opt, 'personachat_history3_sparse', subtasks[0])
        other_task_datafiles = []
        for attr in subtasks[1:]:
            other_task_datafiles.append(
                _path(opt, 'personachat_history3_sparse', attr)
            )
        self.other_task_datafiles = other_task_datafiles
        super().__init__(opt, shared)


class PersonachatH3SparseOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'personachat_history3_sparse', 'original')
        super().__init__(opt, shared)


class DailyDialogTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        assert 'subtasks' in opt, 'subtasks must be specified!'
        subtasks = opt['subtasks'].split(':')
        opt['datafile'] = _path(opt, 'data_filtering_data', subtasks[0])
        other_task_datafiles = []
        for attr in subtasks[1:]:
            other_task_datafiles.append(
                _path(opt, 'data_filtering_data', attr)
            )
        self.other_task_datafiles = other_task_datafiles
        super().__init__(opt, shared)


class DailyDialogOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'data_filtering_data', 'original')
        super().__init__(opt, shared)


class OpensubH3SparseSmallTeacher(DefaultTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        assert 'subtasks' in opt, 'subtasks must be specified!'
        subtasks = opt['subtasks'].split(':')
        opt['datafile'] = _path(opt, 'OpenSubtitles2018_history3_sparse_small', subtasks[0])
        other_task_datafiles = []
        for attr in subtasks[1:]:
            other_task_datafiles.append(
                _path(opt, 'OpenSubtitles2018_history3_sparse_small', attr)
            )
        self.other_task_datafiles = other_task_datafiles
        super().__init__(opt, shared)


class OpensubH3SparseSmallOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'OpenSubtitles2018_history3_sparse_small', 'original')
        super().__init__(opt, shared)
