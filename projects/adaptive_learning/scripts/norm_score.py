import sys
import numpy as np
import argparse


def norm_score(reverse=False):
    data = []
    for line in sys.stdin:
        line = line.strip()
        score = float(line)
        if reverse:
            score = -score
        data.append(score)

    arr = np.asarray(data)
    norm_arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    for num in norm_arr:
        print(num)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--reverse', type=str2bool, default=False)
    opt = parser.parse_args()
    norm_score(reverse=opt.reverse)
