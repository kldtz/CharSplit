#!/usr/bin/env python3

import argparse
import logging
from pprint import pprint
import sys
import warnings
import re

import nltk

from kirke.utils import strutils
from kirke.docstruct import secheadutils


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

dash_only_pat = re.compile(r'^\s*-+\s*$')

def test_nbsp():
    st = "3.4.   Changes to Purchase Orders. If Arrayit Diagnostics requests a change to a Purchase Order (a \"Change Order\") after such Purchase Order is accepted by Arrayit, Arrayit Diagnostics shall inform Arrayit about such Change Order as soon as possible. Arrayit shall use commercially reasonable efforts to accommodate such Change Order."

    pat = re.compile(r'4\.\s+Changes')

    m = pat.search(st)
    if m:
        print("found:")
    else:
        print("not found:")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    fname = args.file

    prev_line, prev_line_idx = '', -1
    with open(fname, 'rt') as fin:
        for line in fin:
            line = line.strip().replace('\xa0', ' ')
            split_idx = -1

            if line and not dash_only_pat.match(line):
                sechead_type, prefix_num, sec_head, split_idx = \
                    secheadutils.extract_sechead_v4(line, prev_line, prev_line_idx)

                if sechead_type:
                    if sechead_type == 'sechead-comb':
                        linetype = 'line2'
                    else:
                        linetype = 'line1'
                    if split_idx != -1:
                        print("{}\t<{}>".format(linetype, line[:min(split_idx, 40)]), end='')
                    else:
                        print("{}\t<{}>".format(linetype, line[:40]), end='')

                    print('\t{}\t[{}]\t[{}]\t{}'.format(sechead_type,
                                                        prefix_num,
                                                        sec_head[:40],
                                                        split_idx))

                # for html file, uncomment below
                # prev_line, prev_line_idx = line, split_idx

    logging.info('Done.')
