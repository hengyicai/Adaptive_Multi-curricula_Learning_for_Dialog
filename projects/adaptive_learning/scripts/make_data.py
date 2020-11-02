#!/usr/bin/env python3

import os
import torch
import random
import argparse
import warnings
from parlai.core.params import ParlaiParser
from parlai.core.utils import round_sigfigs
from parlai.core.agents import create_agent, _load_opt_file
from parlai.agents.hy_lib.common_utils import create_batch_from_file
from parlai.agents.adaptive_learning.helper import compute_batch_loss
from projects.adaptive_learning.scripts.dialog_scoring import initialize_control_information, eval_attr


def create_model_batch(agent, batch_lines):
    batch_obs = []
    for line in batch_lines:
        line = line.strip().replace('\\n', '\n')
        item_arr = line.split('\t')
        if len(item_arr) == 1:
            item_arr.append('__EMPTY__')
        id_context, response = item_arr[0], item_arr[1]
        context = ' '.join(id_context.split(' ')[1:])
        context = context.replace('\\n', '\n')
        batch_obs.append(agent.observe({
            'text': context.lower() if agent.opt['dict_lower'] else context,
            'labels': (response.lower() if agent.opt['dict_lower'] else response,)}))

    batch = agent.batchify(batch_obs)
    return batch


def mk_data_wrt_loss(opt):
    train_file = opt['train_file']
    assert os.path.exists(train_file), 'File {} does not exist!'.format(train_file)
    model_name = opt['model_name']
    model_opt_file = opt['model_opt_path']
    assert model_name and os.path.isfile(
        model_opt_file), 'Can not build training data without model_name or model_opt_path!'

    # Load model agent
    model_opt = _load_opt_file(model_opt_file)
    model = create_agent(model_opt, requireModelExists=True)

    batch_arr = create_batch_from_file(train_file)
    output_data = []

    if model_name == 'dialogwae' or model_name == 'hred':
        drop_first = True
    else:
        drop_first = False

    for batch_lines in batch_arr:
        batch = create_model_batch(model, batch_lines)
        with torch.no_grad():
            if model_name == 'dialogwae' or model_name == 'hred':
                loss_AE = model.model.compute_loss_AE(batch.text_vec, batch.context_lens, batch.text_lengths,
                                                      batch.floors, batch.label_vec, batch.label_lengths)
                model_output = (loss_AE['scores'], loss_AE['preds'])
            else:
                _, model_output = model.compute_loss(batch, return_output=True)
            batch_loss = compute_batch_loss(model_output, batch, model.batch_criterion, model.NULL_IDX, drop_first)
            batch_loss = torch.mean(batch_loss, dim=1)

        for line, loss in zip(batch_lines, batch_loss):
            loss = round_sigfigs(loss, 5)
            output_data.append("{}\t{}".format(line, loss))

    write2file(output_data, 'loss_of_' + opt['model_name'], opt)


def mk_data_wrt_attr(opt):
    initialize_control_information(opt)
    train_file = opt['train_file']
    assert os.path.exists(train_file), 'File {} does not exist!'.format(train_file)
    output_data = []
    with open(train_file) as f:
        for line in f.readlines():
            line = line.strip()
            fileds = line.split('\t')
            if len(fileds) == 2:
                id_context, response = fileds
                next_post = None
            elif len(fileds) == 3:
                id_context, response, next_post = fileds
            else:
                raise RuntimeError('The file {} does not have the correct format!'.format(train_file))

            item_arr = id_context.split()
            idx = item_arr[0]
            history = ' '.join(item_arr[1:]).split('\\n')
            if opt['score_func'] == 'post_sim':
                assert next_post is not None, 'Found next_post to be None, but your score_func is post_sim!'

            score = eval_attr(response, history, next_post, opt['score_func'])
            if opt['score_func'] == 'avg_nidf':
                assert 0 <= score <= 1, 'score should be in the range of [0, 1], but found score=={}'.format(score)
            if opt['score_func'] == 'lastuttsim' or opt['score_func'] == 'post_sim':
                if score is None or score == 'oov':
                    score = -1
                if score > 1:
                    warnings.warn(
                        'score should be in the range of [-1, 1], but found score=={}, cast its value to 1'.format(
                            score))
                if score < -1:
                    warnings.warn(
                        'score should be in the range of [-1, 1], but found score=={}, cast its value to -1'.format(
                            score))
                # assert -1 <= score <= 1, 'score should be in the range of [-1, 1], but found score=={}'.format(score)
                score = -score  # smaller to be eaiser
            print_next_post = (next_post is not None) and opt['print_next_post']
            if print_next_post:
                output_data.append(
                    "{} {}\t{}\t{}\t{}".format(idx, ' \\n '.join(history), response, next_post, score)
                )
            else:
                output_data.append(
                    "{} {}\t{}\t{}".format(idx, ' \\n '.join(history), response, score)
                )

    write2file(output_data, opt['score_func'], opt)


def write2file(out_data, suffix, opt):
    output_file = os.path.join(opt['datapath'], opt['task_dir'], 'train.txt.{}'.format(suffix))
    with open(output_file, 'w') as f:
        for line in out_data:
            f.write(line)
            f.write('\n')


def mk_data(opt):
    if opt['score_func'] == 'loss':
        mk_data_wrt_loss(opt)
    else:
        mk_data_wrt_attr(opt)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    random.seed(42)
    parser = ParlaiParser()
    parser.add_argument('--score_func', type=str, required=True)
    parser.add_argument('--arora_data_dir', type=str, required=True)
    parser.add_argument('--nidf_data_dir', type=str, required=True)
    parser.add_argument('--task_dir', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--print_next_post', type=str2bool, default=False)

    # model loss related
    parser.add_argument('--model_name', type=str, default=None,
                        choices=['seq2seq', 'cvae', 'transformer', 'hred', 'dialogwae'])
    parser.add_argument('--model_opt_path', type=str, default=None)
    opt_ = parser.parse_args()

    mk_data(opt_)
