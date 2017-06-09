#!/usr/bin/env python

import argparse
import logging
from pprint import pprint
import sys
from pprint import pprint
import json
import re

from kirke.utils.corenlputils import annotate

from kirke.utils import strutils

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')


def test_same(orig_text):
    text = orig_text
    print('len(text) = {}'.format(len(text)))

    print("found 30? {}".format(chr(30) in text))
    text = text.replace(chr(30), ' ')

    print("found 16? {}".format(chr(16) in text))
    text = text.replace(chr(16), ' ')

    print("found 1? {}".format(chr(1) in text))
    text = text.replace(chr(1), ' ')

    print("found 2? {}".format(chr(2) in text))
    text = text.replace(chr(2), ' ')

    # print('after replace, len(text) = {}'.format(len(text)))
    text2 = replace_ignorable_ctrl_chars(orig_text)

    if text == text2:
        print("good")
    else:
        print("bad")

    print(len(orig_text))
    print(len(text))
    print(len(text2))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file')

    args = parser.parse_args()

    with open(args.file, 'rt') as fin:
        orig_text = fin.read()

        text2 = strutils.replace_ignorable_json_ctrl_chars(orig_text)

        print("len(orig_text) = ", len(orig_text))
        print("len(text2)     = ", len(text2))
        if orig_text != text2:
            print("good")
        else:
            print("bad")

        print("orig_text = [{}]".format(orig_text))
        print("text2     = [{}]".format(text2))

        result = annotate(text2)
        with open('tmp/debug.json', 'wt') as fout:
            print(result, file=fout)

        pprint(result)

