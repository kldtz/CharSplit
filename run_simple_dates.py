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
from kirke.ebrules import simple_dates

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    fname = args.file

    print(simple_dates.extract_dates(fname))