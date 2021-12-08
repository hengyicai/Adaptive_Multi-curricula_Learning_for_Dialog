import argparse
import os

EMPTY_RESPONSE = 'EMPTY_RESPONSE'


def print_dialogs(dialogs, history_size, sparse=True, print_next_post=False, delimeter='\\n'):
    if sparse:
        step = history_size + 1
    else:
        step = 1
    if print_next_post:
        window_sz = history_size + 1
    else:
        window_sz = history_size

    for i in range(0, len(dialogs), step):
        history = ' {} '.format(delimeter).join(dialogs[i:i + history_size])
        if i + window_sz < len(dialogs):
            response = dialogs[i + history_size]
            if print_next_post:
                next_post = dialogs[i + history_size + 1]
                print("1 {}\t{}\t{}".format(history, response, next_post))
            else:
                print("1 {}\t{}".format(history, response))
        else:
            remaining_turns = history.split(' {} '.format(delimeter))
            if not print_next_post:
                if len(remaining_turns) > 1:
                    history = ' {} '.format(delimeter).join(remaining_turns[0:-1])
                    response = remaining_turns[-1]
                    print("1 {}\t{}".format(history, response))
            else:
                if len(remaining_turns) > 2:
                    history = ' {} '.format(delimeter).join(remaining_turns[0:-2])
                    response = remaining_turns[-2]
                    next_post = remaining_turns[-1]
                    print("1 {}\t{}\t{}".format(history, response, next_post))


def handle_session(session, history_size, sparse=True, print_next_post=False):
    if not session:
        return
    dialogs = []
    for idx, post, response in session:
        dialogs.append(post)
        dialogs.append(response)

    dialogs = list(filter(lambda item: item != EMPTY_RESPONSE, dialogs))
    print_dialogs(dialogs, history_size, sparse, print_next_post)


def build_single_turn_fb_dialog(file_name, history_size, sparse=True, print_next_post=False):
    with open(file_name) as f:
        line = f.readline()
        session = []
        prev_id = -1
        while line:
            line = line.strip()
            item_arr = line.split('\t')
            id_ = int(item_arr[0].split()[0])
            post = ' '.join(item_arr[0].split()[1:]).lower()
            if len(item_arr) > 1:
                response = item_arr[1].lower()
            else:
                response = 'EMPTY_RESPONSE'
            if id_ <= prev_id:
                handle_session(session, history_size, sparse, print_next_post)
                session = [(id_, post, response)]
            else:
                session.append((id_, post, response))
            line = f.readline()
            prev_id = id_
        handle_session(session, history_size, sparse, print_next_post)


def main(fb_dialog_file, history_size, sparse=True, print_next_post=False):
    assert os.path.isfile(fb_dialog_file)
    build_single_turn_fb_dialog(fb_dialog_file, history_size, sparse, print_next_post)


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
    parser.add_argument('--fb_dialog_file', type=str, required=True)
    parser.add_argument('--history_size', type=int, required=True)
    parser.add_argument('--sparse', type=str2bool, default=True, required=True)
    parser.add_argument('--print_next_post', type=str2bool, default=False, required=True)
    opt = parser.parse_args()
    fb_dialog_file = opt.fb_dialog_file
    history_size = opt.history_size
    sparse = opt.sparse
    print_next_post = opt.print_next_post
    main(fb_dialog_file, history_size, sparse, print_next_post)
