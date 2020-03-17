#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train model for ppl metric with pre-selected parameters.
These parameters have some variance in their final perplexity, but they were
used to achieve the pre-trained model.
"""
import os

from parlai.scripts.train_model import TrainLoop, setup_args
from parlai.agents.hy_lib.common_utils import override_opt
from projects.adaptive_learning.utils import set_teacher_args
from projects.adaptive_learning.utils import TENSORBOARD_METRICS

PARLAI_HOME = os.getenv('PARLAI_HOME')

OVERRIDE = {
    "datatype": 'train',
    "max_train_time": -1,
    "batchsize": 4,
    "learningrate": 5e-4,
    "dropout": 0.2,
    "gradient_clip": 0.1,
    "batch_sort": True,
    "validation_every_n_secs": 30,
    "validation_every_n_epochs": -1,
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
        subtasks='avg_nidf:intrep_word:lastuttsim:loss_of_seq2seq:post_sim',
        model='parlai.agents.adaptive_learning.seq2seq:AdaSeq2seqAgent',
        model_file=os.path.join(PARLAI_HOME, 'models/adaptive_learning/personachat_h3_sparse'),
        dict_lower=True,
        dict_minfreq=-1,
        hiddensize=512,
        embeddingsize=300,
        attention='general',
        attention_time='post',
        numlayers=2,
        rnn_class='lstm',
        lookuptable='enc_dec',
        optimizer='adam',
        weight_decay=0,
        embedding_type='glove',
        momentum=0.95,
        bidirectional=True,
        numsoftmax=1,
        no_cuda=False,
        dict_maxtokens=20000,
        dict_tokenizer='split',
        lr_scheduler='invsqrt',
        warmup_updates=2000,
        split_lines=True,
        delimiter='__EOT__',
        tensorboard_log=True,
        tensorboard_log_teacher=True,
        tensorboard_metrics=TENSORBOARD_METRICS,
        reward_metric='total_metric',
        reward_metric_mode='max',
        save_after_valid=False,
    )
    parser.set_defaults(**OVERRIDE)
    opt = parser.parse_args()

    opt = override_opt(opt, OVERRIDE)

    TrainLoop(opt).train()
