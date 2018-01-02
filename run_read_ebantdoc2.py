#!/usr/bin/env python3

import argparse
from collections import defaultdict
import os

from kirke.utils import ebantdoc2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    fname = args.file

    ebdoc = ebantdoc2.load_cached_ebantdoc2(fname)

    print('fname = {}'.format(ebdoc.file_id))
    prov_count_map = defaultdict(int)
    for prov_ant in ebdoc.prov_annotation_list:
        print("prov_ant: {}".format(prov_ant))
        prov_count_map[prov_ant.label] += 1

    for prov, count in prov_count_map.items():
        print("prov_count[{}] = {}".format(prov, count))


    for attrvec in ebdoc.attrvec_list:
        print("attrvec: {}".format(attrvec))
