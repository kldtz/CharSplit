#!/usr/bin/env python3

import argparse
from kirke.utils import antutils


if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('ebdata_file', help='input file')
    parser.add_argument('ant_file', help='output file')    

    args = parser.parse_args()

    if not args.ebdata_file or not args.ant_file:
        print('usage: ebdata_to_ant.py --ebdata_file xxx.ebdata --ant_file xxx.ant')

    antutils.ebdata_to_ant_file(args.ebdata_file,
                                args.ant_file)
