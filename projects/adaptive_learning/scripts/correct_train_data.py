#!/bin/python
import argparse
import os


def main(opt_):
    train_file = opt_.train_file
    valid_file = opt_.valid_file
    test_file = opt_.test_file
    assert all(os.path.isfile(f) for f in [train_file, valid_file, test_file])
    with open(valid_file) as valid_f, open(test_file) as test_f:
        valid_data = dict([(line.strip(), 1) for line in valid_f.readlines()])
        test_data = dict([(line.strip(), 1) for line in test_f.readlines()])

    with open(train_file) as f:
        for line in f.readlines():
            line = line.strip()
            item_arr = line.split('\t')
            id_context_response = '\t'.join(item_arr[0:2])
            if id_context_response in valid_data or id_context_response in test_data:
                pass
            else:
                print(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    opt = parser.parse_args()
    main(opt)
