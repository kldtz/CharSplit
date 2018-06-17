#!/usr/bin/env python3

import argparse
import logging
import pprint
import sys
import warnings
import re
import copy

from typing import Dict, List, Optional, Pattern, Tuple

from kirke.utils import ebsentutils
from kirke.sampleutils import regexgen


def extract_cand(alphanum: regexgen.RegexContextGenerator, line: str):
    candidates, _, _ = alphanum.get_candidates_from_text(line)
    cand_text = ' /// '.join([cand['chars'] for cand in candidates])
    return cand_text

def extract_idnum_list(alphanum: regexgen.RegexContextGenerator,
                       line: str,
                       label_ant_list_param: Optional[List[ebsentutils.ProvisionAnnotation]] = None,
                       label: str = '') -> Tuple[List[Dict],
                                                 List[int],
                                                 List[bool]]:
    candidates, group_id_list, label_list = \
        alphanum.get_candidates_from_text(line,
                                          group_id=-1,
                                          label_ant_list_param=label_ant_list_param,
                                          label=label)
    return candidates, group_id_list, label_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    # parser.add_argument('file', help='input file')

    args = parser.parse_args()

    # fname = args.file

    """
    # line = '1234567'
    regex = re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)')

    for i, line in enumerate(st_list):
        print('\nline {}: [{}]'.format(i, line))
        mat_list = extract_matches(regex, line)

        for j, mat in enumerate(mat_list):
            print('    mat #{}: {}'.format(j, mat))
            # pprint.pprint(mat)
    """



    # regex = re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)')
    # regex = re.compile(r'((\+ )?[^\s]*\d[^\s]*( {1,2}[^\s]*\d[^\s]*)*)')
    regex = re.compile(r'([^\s]*\d[^\s]*)')
    regex = re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)')

    alphanum = regexgen.RegexContextGenerator(3,
                                              3,
                                              re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)'),
                                              'idnum',
                                              join=True,
                                              length_min=2)


    st_list = ['xxxxxx1',
               'xxx1, hi xxx2',
               'xxx1, xxx2, xxx3',
               'xxx1,xxx2,xxx3',
               '63 33',
               'xxx1 xxx2 bbb',
               'xxx1 xxx2 xxx3 bbb',
               'text PHONE NUM: 191 541 754 3010 text',
               # '+ 63 3 477 4000',
               '+1 917',
               '+49',
               '8 10 64 3 477 4000'
               ]

    for i, line in enumerate(st_list):
        print('\nline {}: [{}]'.format(i, line))
        mat_list, _, _ = extract_idnum_list(alphanum, line)

        if mat_list:
            for j, mat in enumerate(mat_list):
                print('    mat #{}:'.format(j))
                pprint.pprint(mat, indent=20)

    line = 'aaaaaaaaa bbbbbbbbb ccccccccc abcd #678,012 456 text ddddddddd eeeeeeeee ffffffffff'
    print('found idnum = [{}]'.format(extract_cand(alphanum, line)))  # , '#678,901 345')

    ant_list = [ebsentutils.ProvisionAnnotation(label='purchase_order_number',
                                                start=44,
                                                end=47)]
    candidates, group_id_list, label_list = \
        extract_idnum_list(alphanum,
                           line,
                           label_ant_list_param=ant_list,
                           label='purchase_order_number')
    for i, cand in enumerate(candidates):
        print("cand_list[{}] = {}".format(i, cand))
    for i, label in enumerate(label_list):
        print("label_list[{}] = {}".format(i, label))


    line = 'aaaaaaaaa bbbbbbbbb ccccccccc abcd #678,012 456 text ddddddddd eeeeeeeee ffffffffff'
    print('found idnum = [{}]'.format(extract_cand(alphanum, line)))  # , '#678,901 345')

    ant_list = [ebsentutils.ProvisionAnnotation(label='purchase_order_number',
                                                start=35,
                                                end=43)]
    candidates, group_id_list, label_list = \
        extract_idnum_list(alphanum,
                           line,
                           label_ant_list_param=ant_list,
                           label='purchase_order_number')
    for i, cand in enumerate(candidates):
        print("cand_list[{}] = {}".format(i, cand))
    for i, label in enumerate(label_list):
        print("label_list[{}] = {}".format(i, label))
