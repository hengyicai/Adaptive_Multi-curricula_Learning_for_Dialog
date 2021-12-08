# /bin/bash/python

import sys

for line in sys.stdin:
    line = line.strip()
    item_arr = line.split('\t')
    if len(item_arr) < 2:
        raise RuntimeError('The input file is not in the right format!')
    print('\t'.join(item_arr[0:2]))
