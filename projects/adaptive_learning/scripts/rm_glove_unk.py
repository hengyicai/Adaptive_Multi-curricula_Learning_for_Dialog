#!/bin/bash
import sys

import torch

if __name__ == '__main__':
    path_pt = sys.argv[1]
    itos, stoi, vectors, dim = torch.load(path_pt)
    for line in sys.stdin:
        line_ = line.strip()
        item_arr = line_.split('\t')
        context = ' '.join(list(filter(lambda w: w != '\\n' or w != '\n', item_arr[0].split()[1:])))

        response = item_arr[1].strip()
        res_tok = response.split()
        context_tok = context.split()
        if all([t not in stoi for t in res_tok]) or all([t not in stoi for t in context_tok]):
            pass
        else:
            print(line_)
