#!/usr/bin/env python3

import argparse
from collections import defaultdict
# pylint: disable=unused-import
from typing import DefaultDict, List, Tuple

from kirke.utils import strutils


def diff_word_lists(fname1: str,
                    fname2: str) \
                    -> Tuple[List[Tuple[str, int, int]],
                             List[Tuple[str, int, int]]]:

    st_list_1 = strutils.load_str_list(fname1)
    st_list_2 = strutils.load_str_list(fname2)

    word_freq_map1 = defaultdict(int)  # type: DefaultDict[str, int]
    for st1 in st_list_1:
        words1 = st1.split()
        for word in words1:
            word_freq_map1[word] += 1

    word_freq_map2 = defaultdict(int)  # type: DefaultDict[str, int]
    for st2 in st_list_2:
        words2 = st2.split()
        for word in words2:
            word_freq_map2[word] += 1

    same_list = []  # type: List[Tuple[str, int, int]]
    diff_list = []  # type: List[Tuple[str, int, int]]

    for word in sorted(word_freq_map1):
        freq1 = word_freq_map1[word]
        freq2 = word_freq_map2.get(word, 0)

        if freq1 == freq2:
            # print("SAME: [{}]\t{}\t{}".format(word, freq1, freq2))
            same_list.append((word, freq1, freq2))
        else:
            # print("DIFF: [{}]\t{}\t{}".format(word, freq1, freq2))
            diff_list.append((word, freq1, freq2))

    for word in sorted(word_freq_map2):

        if word not in word_freq_map1:
            freq1 = 0
            freq2 = word_freq_map2[word]

            # print("DIFF_W2: [{}]\t{}\t{}".format(word, freq1, freq2))
            diff_list.append((word, freq1, freq2))

    return same_list, diff_list


def main():
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file1', help='input file 1')
    parser.add_argument('file2', help='input file 2')

    # pylint: disable=invalid-name
    args = parser.parse_args()

    same_list, diff_list = diff_word_lists(args.file1, args.file2)

    print("same: {}, diff: {}".format(len(same_list), len(diff_list)))
    for word, freq1, freq2 in diff_list:
        print("diff\t{}\t{}\t{}".format(word, freq1, freq2))

if __name__ == '__main__':
    main()
