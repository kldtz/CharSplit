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
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    fname = args.file

    line = 'entered into as of __________, 2011 (the “Effective Date”)'
    line = 'made this 22nd day of April, 2014'
    line = '[ ], 2012'
    line = 'dated as of March     , 2009'

    line = 'dated as of the 1 st day of April, 2005'
    xxx = dates.extract_dates_from_party_line(line)
    print("{}\t{}".format(line, xxx))

    """
    with open(fname, 'rt') as fin:
        for line in fin:
            line = line.strip()
            xxx = dates.extract_dates_from_party_line(line)
            #if not xxx:
            #    print("{}\t{}".format(line, xxx))
            print("{}\t{}".format(line, xxx))
            # if len(xxx) != 1:
            #    print("{}\t{}".format(line, xxx))
    """

    # print(dates.extract_dates(fname))
