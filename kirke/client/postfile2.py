#!/usr/bin/env python

import argparse
import logging
import os
from pathlib import Path
import sys
import requests


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# pylint: disable=C0103
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='identify the language')
    parser.add_argument('-v', '--verbosity', help='increase output verbosity')
    parser.add_argument('-d', '--debug', action='store_true', help='print debug information')
    parser.add_argument('--url', help='url to post the file')
    parser.add_argument('filename')

    args = parser.parse_args()
    if args.verbosity:
        print('verbosity turned on')
    if args.debug:
        isDebug = True

    url = 'http://127.0.0.1:8000/detect-lang'
    # use url='http://127.0.0.1:8000/detect-langs' to detect top langs with probabilities
    if args.url:
        url = args.url

    txt_file = Path(args.filename)
    if txt_file.is_file():
        if args.filename.endswith('.txt'):
            offset_filename = args.filename.replace('.txt', '.offsets.json')
            if os.path.exists(offset_filename):
                files = [('file', open(args.filename, 'rt', encoding='utf-8')),
                         ('file', open(offset_filename, 'rt', encoding='utf-8'))]
            else:
                files = {'file': open(args.filename, 'rt', encoding='utf-8')}

        # payload = {'types': 'party'}
        # payload = {'types': 'change_control'}
        # payload = {'types': 'termination,term,confidentiality,cust_3566'}
        payload = {'types': 'term'}
        # payload = {}
        req = requests.post(url, files=files, data=payload)
        print(req.text)
    else:
        print("file '{}' is not a valid file".format(args.filename), file=sys.stderr)
