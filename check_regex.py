#!/usr/bin/env python3

import argparse
import logging
import pprint
import sys
import warnings
import re
import copy

from typing import Dict, List, Pattern

from kirke.sampleutils import regexgen


def extract_matches(pat: Pattern, atext: str) -> List[Dict]:
    group_num = 1
    length_min = 2

    # print('atext = [{}]'.format(atext))

    candidates = []  # type: List[Dict]
    #finds all matches in the text and adds window around each as a candidate
    matches = pat.finditer(atext)
    for match in matches:
        match_start, match_end = match.span(group_num)
        match_str = match.group(group_num)

        print("  -- match_str {} {}: {}".format(match_start,
                                                match_end,
                                                match_str))

        #update span based on window size
        new_start = match_start
        new_end = match_end

        # clean up the string if special character is at the end.  Currently
        # none of the matat_str will have nose characters except for ";" or ":"
        if match_str.endswith(',') or match_str.endswith(';') or match_str.endswith(':'):
            match_str = match_str[:-1]
            match_end -= 1
        if match_str.endswith(')') and not '(' in match_str:
            match_str = match_str[:-1]
            match_end -= 1
        if match_str.startswith('(') and not ')' in match_str:
            match_str = match_str[1:]
            match_start += 1

        a_candidate = {'start': match_start,
                       'end': match_end,
                       'chars': match_str}
        candidates.append(a_candidate)

    merge_candidates = []
    i = 0
    while i < len(candidates):
        skip = True
        new_candidate = copy.deepcopy(candidates[i])
        while skip and i+1 < len(candidates):
            diff = candidates[i+1]['start'] - new_candidate['end']
            diff_str = atext[new_candidate['end']:candidates[i+1]['start']]
            if (diff_str.isspace() or not diff_str) and diff < 3:
                new_candidate['end'] = candidates[i+1]['end']
                new_candidate['chars'] = atext[new_candidate['start']:new_candidate['end']]
                i += 1
            else:
                merge_candidates.append(new_candidate)
                i += 1
                skip = False
        if i == len(candidates) - 1:
            skip = False
            merge_candidates.append(candidates[i])
            i += 1
    candidates = merge_candidates

    filtered_candidates = []
    for candidate in candidates:
        if len(candidate['chars']) >= length_min:
            filtered_candidates.append(candidate)
    
    return filtered_candidates


def extract_matches_2(regex: Pattern, atext: str) -> List[Dict]:

    candidates = regexgen.extract_doc_candidates(regex_pat=regex,
                                                 group_num=1,
                                                 atext=atext,
                                                 candidate_type='mycant',
                                                 num_prev_words=3,
                                                 num_post_words=3,
                                                 min_length=2,
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

    # regex = re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)')
    # regex = re.compile(r'((\+ )?[^\s]*\d[^\s]*( {1,2}[^\s]*\d[^\s]*)*)')
    regex = re.compile(r'([^\s]*\d[^\s]*)')
    regex = re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)')    

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
        mat_list = extract_matches_2(regex, line)

        if mat_list:
            for j, mat in enumerate(mat_list):
                print('    mat #{}:'.format(j))
                pprint.pprint(mat, indent=20)        

