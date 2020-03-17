#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import locale
from parlai.scripts.train_model import setup_args, TrainLoop
from parlai.agents.hy_lib.common_utils import override_opt
from projects.adaptive_learning.utils import set_teacher_args
from projects.adaptive_learning.utils import TENSORBOARD_METRICS

# locale.setlocale(locale.LC_ALL, 'en_US')
PARLAI_HOME = os.getenv('PARLAI_HOME')

OVERRIDE = {
    "datatype": 'train',
    "max_train_time": -1,
    "batchsize": 256,
    "dropout": 0.2,
    "batch_sort": True,
    "validation_every_n_secs": -1,
    "validation_every_n_epochs": 0.5,
    "validation_metric": 'ppl',
    "validation_metric_mode": 'min',
    "validation_patience": 12,
    "log_every_n_secs": 1,
    "shuffle": False,
    "numworkers": 40,
    "multigpu": False,
    "num_epochs": 20,
    "display_examples": False,
    "history_size": -1,
    "text_truncate": 128,
    "label_truncate": 128,
    "truncate": 128,
    "gpu": 0,
    "batch_sort_field": 'label',
    "pytorch_teacher_batch_sort": False,
    "pace_by": 'sample',
    "T": 3000,
    "c0": 0.01,
    "p": 2,
    "beam_size": 1,
}

if __name__ == '__main__':
    parser = setup_args()
    parser = set_teacher_args(parser)

    parser.set_defaults(
        task='adaptive_learning:personachat_h3_sparse',
        subtasks='avg_nidf',
        model='parlai.agents.adaptive_learning.dialog_wae:DialogWaeAgent',
        model_file=os.path.join(PARLAI_HOME, 'models/adaptive_learning/personachat_h3_sparse'),
        dict_lower=True,
        dict_tokenizer='split',
        embedding_type='glove',
        skip_generation=False,
        split_lines=True,
        person_tokens=True,
        delimiter='__EOT__',
        hred=False,
        vhred=False,
        optimizer='sgd',
        learningrate=1,
        momentum=0.95,
        nesterov=True,
        lr_scheduler="reduceonplateau",
        lr_scheduler_decay=0.8,
        lr_scheduler_patience=5,
        tensorboard_log=True,
        tensorboard_log_teacher=True,
        tensorboard_metrics=TENSORBOARD_METRICS + ',kl_loss,bow_loss',
        reward_metric='total_metric',
        reward_metric_mode='max',
        save_after_valid=False,
    )
    parser.set_defaults(**OVERRIDE)
    opt = parser.parse_args()

    opt = override_opt(opt, OVERRIDE)

    TrainLoop(opt).train()
