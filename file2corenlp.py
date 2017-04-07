#!/usr/bin/env python

import argparse
import logging
from pprint import pprint
import sys
from pprint import pprint
import json

from kirke.utils.corenlputils import annotate

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file')

    args = parser.parse_args()

    with open(args.file, 'rt') as fin:
        text = fin.read()
        print('len(text) = {}'.format(len(text)))

        print("found 30? {}".format(chr(30) in text))
        text = text.replace(chr(30), ' ')

        print("found 16? {}".format(chr(16) in text))
        text = text.replace(chr(16), ' ')

        print("found 1? {}".format(chr(1) in text))
        text = text.replace(chr(1), ' ')
        
        print('after replace, len(text) = {}'.format(len(text)))
        result = annotate(text)
        with open('debug.json', 'wt') as fout:
            print(result, file=fout)

        pprint(result)
