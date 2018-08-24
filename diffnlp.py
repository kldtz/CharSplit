#!/usr/bin/env python3

import argparse
import re
from typing import List

from kirke.utils import strutils


SAMPLE_TEXT_LEN = 30

IS_PRINT_MATCHED = False
IS_PRINT_MINOR_DIFF = True

ENDSWITH_NUM_PAT = re.compile(r'\s+\d+$')

MAX_NUM_DIFF = 1000

def get_next_index_2(index2: int,
                     st_list_2: List[str],
                     len_st_list_2: int) \
                     -> int:
    index2 += 1
    while index2 < len_st_list_2 and \
          not st_list_2[index2].strip():
        index2 += 1
    return index2


def diff_st_lists(fname1: str,
                  st_list_1: List[str],
                  fname2: str,
                  st_list_2: List[str]) -> None:

    fname1 = '11111'
    fname2 = '22222'
    index2 = 0
    len_st_list_2 = len(st_list_2)
    num_match = 0
    num_diff = 0
    num_minor_diff = 0
    for index1, st1 in enumerate(st_list_1):
        st1 = st1.strip()

        if st1:
            st2 = st_list_2[index2].strip()

            if st1 == st2:
                num_match += 1
                if IS_PRINT_MATCHED:
                    if len(st1) < SAMPLE_TEXT_LEN:
                        print("\nmatching #{}, line {} to {}, [{}]".format(num_match, index1+1, index2+1, st1[:SAMPLE_TEXT_LEN]))
                    else:
                        print("\nmatching #{}, line {} to {}, [{}...]".format(num_match, index1+1, index2+1, st1[:SAMPLE_TEXT_LEN]))
                index2 = get_next_index_2(index2, st_list_2, len_st_list_2)
            else:
                mat = ENDSWITH_NUM_PAT.search(st1)
                
                if mat:
                    st1_v2 = st1[:mat.start()]
                    if st1_v2 == st2:
                        if IS_PRINT_MINOR_DIFF:
                            num_minor_diff += 1                            
                            print('\n  --minor diff, extra suffix_num #{}:'.format(num_minor_diff))
                            print('     {} line {}: [{}]'.format(fname1, index1+1, st1))
                            print('     {} line {}: [{}]'.format(fname2, index2+1, st2))

                        index2 = get_next_index_2(index2, st_list_2, len_st_list_2)
                        continue

                if st1.startswith(st2):
                    is_finished = False
                    matched_st2_list = []  # type: List[Tuple[int, str]]
                    matched_st2_list.append((index2, st2))

                    st1_v2 = st1[len(st2):].strip()
                    # check if the extra stuff in st1 is in the next line
                    if len(st1_v2) > 10:
                        index2 = get_next_index_2(index2, st_list_2, len_st_list_2)

                        while st1_v2.startswith(st_list_2[index2]):
                            st2 = st_list_2[index2]
                            matched_st2_list.append((index2, st2))
                            st1_v2 = st1_v2[len(st2):].strip()
                            index2 = get_next_index_2(index2, st_list_2, len_st_list_2)

                            if not st1_v2:  # finished matched
                                if IS_PRINT_MATCHED:
                                    num_match += 1                                    
                                    print('\nmatching #{}:'.format(num_match))
                                    print('   {} line {}: [{}]'.format(fname1, index1+1, st1))
                                    for idx2, matched_st2 in matched_st2_list:
                                        print('   {} line {}: [{}...]'.format(fname2, idx2+1, matched_st2[:SAMPLE_TEXT_LEN]))
                                is_finished = True
                                break
                            
                            
                        # if st1_v2 == st_list_2[index2]:
                        #     index2 = get_next_index_2(index2, st_list_2, len_st_list_2)
                        #    continue

                    if is_finished:
                        continue

                    if IS_PRINT_MINOR_DIFF:
                        num_minor_diff += 1
                        print('\n  --minor diff, prefix only #{}:'.format(num_minor_diff))
                        print('     {} line {}: [{}]'.format(fname1, index1+1, st1))
                        print('     {} line {}: [{}]'.format(fname2, index2+1, st2))
                        
                    index2 = get_next_index_2(index2, st_list_2, len_st_list_2)
                    continue

                num_diff += 1
                print('\ndiff #{}:'.format(num_diff))
                print('   {} extra line {}: [{}]'.format(fname1, index1+1, st1))
                print('   {} line {}: [{}]'.format(fname2, index2+1, st2))
                
                
                if num_diff > MAX_NUM_DIFF:
                    return
    print('\nDone.')
    
                  
def main():
    global IS_PRINT_MATCHED, IS_PRINT_MINOR_DIFF
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument("-m", "--matched", action="store_true", help="print matched in addition to diff")
    parser.add_argument("-n", "--not_minor", action="store_true", help="not print minor diff in addition to diff")
    parser.add_argument('--max', help="maximum of diff to print")
    parser.add_argument('file1', help='input file 1')
    parser.add_argument('file2', help='input file 2')    

    # pylint: disable=invalid-name
    args = parser.parse_args()

    if args.matched:
        IS_PRINT_MATCHED = True
    if args.not_minor:
        IS_PRINT_MINOR_DIFF = False
    if args.max:
        MAX_NUM_DIFF = int(args.max)

    stlist1 = strutils.load_str_list(args.file1)
    stlist2 = strutils.load_str_list(args.file2)

    diff_st_lists(args.file1, stlist1,
                  args.file2, stlist2)
    

if __name__ == '__main__':
    main()
