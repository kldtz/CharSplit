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
from kirke.ebrules import dates

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    # parser.add_argument('file', help='input file')

    args = parser.parse_args()

    line = 'entered into as of __________, 2011 (the “Effective Date”)'
    line = 'made this 22nd day of April, 2014'
    result = dates.extract_std_dates(line)

    print("line = [{}]".format(line))
    for start, end in result:
        print('date = [{}]'.format(line[start:end]))

    words = strutils.get_simple_words(line)
    for word in words:
        print("word = [{}]".format(word))

    print("divider chars: [{}]".format(line[19:21]))
    
    words = strutils.get_prev_n_words(line, 19, 2)
    print("prev words: [{}]".format(words))

    words = strutils.get_post_n_words(line, 21, 2)
    print("post words: [{}]".format(words))
    
        
    
    """
    line = '[ ], 2012'
    line = 'dated as of March     , 2009'

    line = 'dated as of the 1 st day of April, 2005'
    xxx = dates.extract_dates_from_party_line(line)
    print("{}\t{}".format(line, xxx))
    """


