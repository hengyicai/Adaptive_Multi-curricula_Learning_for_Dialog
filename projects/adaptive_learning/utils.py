def set_teacher_args(parser):
    # curriculum learning
    parser.add_argument_group('Curriculum Learning Arguments')
    parser.add_argument('--pace_by', type=str, choices=['sample', 'bucket'], default='sample')
    parser.add_argument('--T', type=int, default=3000)
    parser.add_argument('--c0', type=float, default=0.01)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--reward_metric', type=str, default='ppl')
    parser.add_argument('--reward_metric_mode', type=str, choices=['min', 'max'], default='min')
    parser.add_argument('--subtasks', type=str, default='original')
    parser.add_argument('--random_policy', type='bool', default=False)
    parser.add_argument('--count_sample', type='bool', default=False)
    parser.add_argument('--anti', type='bool', default=False)

    parser.add_argument('--cutoff_metric_name', type=str, default='none')
    parser.add_argument('--cutoff_metric_val', type=float, default=-1.0)

    # teacher optimizer
    parser.add_argument_group('Teacher Optimizer Arguments')
    parser.add_argument(
        '--optimizer_teacher', default='adam',
        help='Choose between pytorch optimizers. Any member of torch.optim'
             ' should be valid.'
    )
    parser.add_argument(
        '--learningrate_teacher', type=float, default=1e-4,
        help='learning rate for policy net.'
    )
    parser.add_argument(
        '--learningrate_teacher_critic', type=float, default=1e-3,
        help='learning rate for critic net.'
    )
    parser.add_argument(
        '--gradient_clip_teacher', type=float, default=0.1,
        help='gradient clipping using l2 norm'
    )
    parser.add_argument(
        '--momentum_teacher', default=0.95, type=float,
        help='if applicable, momentum value for optimizer.'
    )
    parser.add_argument(
        '--nesterov_teacher', default=True, type='bool',
        help='if applicable, whether to use nesterov momentum.'
    )
    parser.add_argument(
        '--nus_teacher', default='0.7', type='floats',
        help='if applicable, nu value(s) for optimizer. can use a single '
             'value like 0.7 or a comma-separated tuple like 0.7,1.0'
    )
    parser.add_argument(
        '--betas_teacher', default='0.9,0.999', type='floats',
        help='if applicable, beta value(s) for optimizer. can use a single '
             'value like 0.9 or a comma-separated tuple like 0.9,0.999'
    )

    parser.add_argument(
        '--reg_action', default=0.001, type=float, help='regularization weight regarding action entropy'
    )
    # logs
    parser.add_argument('--tensorboard_log_teacher', type='bool', default=False)
    parser.add_argument('--run_test_after_validation', type='bool', default=False)
    return parser


MODELS = ['seq2seq', 'cvae', 'transformer', 'hred', 'dialogwae']
LOSS_NAMES = ['loss_of_{}'.format(model_name) for model_name in MODELS]
SUB_TASKS = ['avg_nidf', 'intrep_word', 'lastuttsim', 'post_sim'] + LOSS_NAMES
TENSORBOARD_METRICS = 'ppl,loss,bleu,dist_1_ratio,dist_2_ratio,dist_3_ratio,embed_avg,' \
                      'embed_extrema,embed_greedy,embed_coh,word_entropy_uni,word_entropy_bi,' \
                      'word_entropy_tri,intra_dist_1,intra_dist_2,intra_dist_3,' + \
                      ','.join(['p_{}'.format(t) for t in SUB_TASKS]) + ',' + \
                      ','.join(['c_{}'.format(t) for t in SUB_TASKS])
