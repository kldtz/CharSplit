#!/usr/bin/env python3

import argparse
import logging
import pprint
import sys
import warnings
import re
import copy

from typing import Dict, List, Pattern

from kirke.sampleutils import idnumgen
from kirke.utils import ebsentutils


def extract_idnum_list(atext: str) -> List[Dict]:

    idnum_word_pat = re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)')
    candidates = idnumgen.extract_idnum_list(atext,
                                             idnum_word_pat,
                                             group_num=1,
                                             is_join=True)
    return candidates

def extract_idnum_str_list(atext: str) -> List[Dict]:

    idnum_word_pat = re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)')
    candidates = idnumgen.extract_idnum_list(atext,
                                             idnum_word_pat,
                                             group_num=1,
                                             is_join=True)
    return candidates
    

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

    """
    # regex = re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)')
    # regex = re.compile(r'((\+ )?[^\s]*\d[^\s]*( {1,2}[^\s]*\d[^\s]*)*)')
    regex = re.compile(r'([^\s]*\d[^\s]*)')
    regex = re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)')    
    """
    
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
               '8 10 64 3 477 4000',
               '8'
               ]

    for i, line in enumerate(st_list):
        print('\nline {}: [{}]'.format(i, line))
        mat_list = extract_idnum_list(line)

        if mat_list:
            for j, mat in enumerate(mat_list):
                print('    mat #{}: {}'.format(j, mat))
                # pprint.pprint(mat, indent=20)


    idnum_gen = idnumgen.IdNumContextGenerator(3,
                                               3,
                                               re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)'),
                                               'idnum',
                                               is_join=True,
                                               length_min=2)
                
    line = 'abcd #678,012 456 text'
    line = 'aaaaaaaaa bbbbbbbbb ccccccccc abcd #678,012 456 text ddddddddd eeeeeeeee ffffffffff'    
    print('\nline: [{}]'.format(line))
    mat_list = extract_idnum_list(line)
    if mat_list:
        for j, mat in enumerate(mat_list):
            print('    mat #{}: {}'.format(j, mat))

    ant_list = [ebsentutils.ProvisionAnnotation(label='purchase_order_number',
                                                start=35,
                                                end=43)]
    cand_list, cand_label_list, cand_group_id_list = \
        idnum_gen.get_candidates_from_text(line,
                                           group_id=-1,
                                           label_ant_list=ant_list,
                                           label='purchase_order_number')
    for i, cand in enumerate(cand_list):
        print("cand_list[{}] = {}".format(i, cand))
    for i, label in enumerate(cand_label_list):
        print("label_list[{}] = {}".format(i, label))
                
