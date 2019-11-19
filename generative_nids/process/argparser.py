import argparse


def get_argparser():

    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str,
                        help="dataset directory")
    parser.add_argument("-o", "--out_dir", type=str, default=None,
                        help="output directory")
    parser.add_argument("-p", "--processes", type=int, default=1,
                        help="number of processes")
    parser.add_argument("-f", "--frequency", type=str, default='T',
                        help="time window scale")
    parser.add_argument("-l", "--logging", type=str, default='INFO',
                        help="logging level")
    return parser