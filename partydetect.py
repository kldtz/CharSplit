#!/usr/bin/env python3

import argparse
import logging
import sys
import re
import fileinput

DEBUG_MODE = False

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')

st_pat_list = ['is made and entered into by and between',
               'is entered into between',
               'is entered into among',
               'is entered into',
               'by and between',
               'by and among',
               'is by and among',
               'by and between',
               'is among',
               'between',
               'confirm their agreement',
               'each confirms its agreement',
               'confirms its agreement',
               'the parties to this',
               'promises to pay to'
]

party_pat = re.compile(r'\b({})\b'.format('|'.join(st_pat_list)), re.IGNORECASE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('files', metavar='FILE', nargs='*',
                        help='files to read, if empty, stdin is used')    

    args = parser.parse_args()
    line_iter = fileinput.input(args.files)

    num_matched, num_not_matched = 0, 0
    for line in line_iter:
        cols = line.strip().split('\t')

        mat = party_pat.search(line)
        if mat:
            # print('[{}]'.format(cols[6]))
            num_matched += 1
        else:
            if len(cols) < 7:
                print("error {}".format(line))
            else:
                print('[{}]'.format(cols[6]))
            num_not_matched += 1

    print('num_matched = {}'.format(num_matched), file=sys.stderr)
    print('num_not_matched = {}'.format(num_not_matched), file=sys.stderr)
    
