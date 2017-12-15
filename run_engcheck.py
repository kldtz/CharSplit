#!/usr/bin/env python

import argparse
import logging
from pprint import pprint
import sys
import warnings

import nltk

from kirke.utils import strutils

from kirke.utils.engutils import (files2ngram, load_ngram_score, check_english,
                                  classify_english_line,
                                  classify_english_lines,
                                  classify_english_sentence)


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')

    args = parser.parse_args()
    work_dir = args.dir

    # word_dir = 'data-test-tmp'
    # classify_english_lines(word_dir)

    st = 'viii. Air temperature measurement uncertainty'
    is_english = classify_english_sentence(st, debug_mode=True)
    print("is_english = {}, {}".format(is_english, st))

    st2 = 'Permits (O&M Facilities] 12/10/2011, construction (Likely 2/15/2015)'
    num_dates = strutils.count_date(st2)
    print("num_dates should be 2, got", num_dates)

    st3 = '3/9/1920 Permits (O&M Facilities] 12/10/2011, construction (Likely 2/15/2015)'
    num_dates = strutils.count_date(st3)
    print("num_dates should be 3, got", num_dates)

    logging.info('Done.')
