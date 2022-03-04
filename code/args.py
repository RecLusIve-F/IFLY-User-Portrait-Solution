from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()

    # general
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index. Default: -1, using CPU.')

    # data
    parser.add_argument('--max_len', type=int, default=100, help='Length of TagIds.')

    # model
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embed_size', nargs='+', type=int)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--epoch', type=int, default=100)

    parser.set_defaults(embed_size=[103, 48, 30, 110])

    args = parser.parse_args()
    return args


args = make_args()
print(args)