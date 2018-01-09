#!/usr/bin/env python3

import argparse
from pprint import pprint
import sys
import re

from kirke.utils import regexutils, strutils
from kirke.ebrules import titles

    

if __name__ == '__main__X':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    # parser.add_argument('file', help='input file')

    args = parser.parse_args()

    # fname = args.file

    print("tag_regexes1")
    print(regexutils.tag_regexes1)

    print("tag_regexes")
    print(regexutils.tag_regexes)
    for i, x in enumerate(regexutils.tag_regexes):
        print("{}\t{}".format(i, x))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    fname = args.file

    with open(fname, 'rt') as fin:
        for line in fin:
            line = line[:-1]  # stip line
            if line:
                print("line: [{}]".format(line))
                print("plne: [{}]".format(regexutils.process_as_line(line)))
                print("awrd: [{}]".format(' '.join(strutils.get_alpha_or_num_words(line, is_lower=True))))
                print("  v2: [{}]".format(regexutils.process_as_line_2(line)))
