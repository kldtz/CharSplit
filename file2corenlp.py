#!/usr/bin/env python

import argparse
import logging
from pprint import pprint
import sys
from pprint import pprint
import json
import re

from kirke.utils.corenlputils import annotate

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# We encountered characters, 1, 2, 16, 31 in input to corenlp before.
# These characters passed through the processing and appeared as they are
# in the JSON output.  Unfortunately, these are not valid characters in JSON.
# Replace them for now
BAD_JSON_CTRL_CHARS = set([0, 1, 2, 3, 4, 5, 6, 7, 8,
                      # 9, \t
                      # 10 \n
                      11, 12,
                      # 13 \r
                      14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                      31,
                      127])
IGNORABLE_JSON_CTRL_CHARS = ''.join([chr(chx) for chx in BAD_JSON_CTRL_CHARS])
IGNORABLE_JSON_CTRL_PAT = re.compile(r'[' + IGNORABLE_JSON_CTRL_CHARS + ']')

UUENCODE_REMOVE_PAT = re.compile(r'%(\d)')

def replace_ignorable_ctrl_chars(line: str) -> str:
    line = re.sub(IGNORABLE_JSON_CTRL_PAT, ' ', line)
    # TODO, in future, regex these
    line = line.replace('\u201c', '"')
    line = line.replace('\u201d', '"')
    line = line.replace('\u2019', "'")
    line = re.sub(UUENCODE_REMOVE_PAT, r' \1', line)
    return line

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

        text2 = replace_ignorable_ctrl_chars(orig_text)

        with open('bkkk.json', 'wt') as fout:
            print(text2, file=fout)

        print("len(orig_text) = ", len(orig_text))
        print("len(text2)     = ", len(text2))
        if orig_text != text2:
            print("good")
        else:
            print("bad")


        print("orig_text = [{}]".format(orig_text))
        print("text2     = [{}]".format(text2))

        result = annotate(text2)
        with open('debug.json', 'wt') as fout:
            print(result, file=fout)

        pprint(result)

