#!/usr/bin/env python

import argparse
import logging
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
    parser.add_argument('-u', '--url', help='url to post the file')
    parser.add_argument('-l', '--lang', action='store_true', help='to detect lang')
    parser.add_argument('--noprovs', action='store_true', help='to detect lang')    
    parser.add_argument('--doccat', action='store_true', help='to classify document')
    
    parser.add_argument('filename')

    args = parser.parse_args()
    if args.verbosity:
        print('verbosity turned on')
    if args.debug:
        isDebug = True

    url = 'http://127.0.0.1:8000/annotate-doc'

    if args.url:
        url = args.url

    payload = {}
    if args.lang:
        payload['detect-lang'] = True
    if args.doccat:
        payload['classify-doc'] = True

    if not args.noprovs:
        # payload['types'] = 'all-provs'
        pass

    txt_file = Path(args.filename)
    if txt_file.is_file():
        files = {'file': open(args.filename, 'rt', encoding='utf-8')}
        # payload = {'types': 'party'}
        # payload = {'types': 'change_control'}
        # payload = {'types': 'party,change_control'}
        req = requests.post(url, files=files, data=payload)
        print(req.text)
    else:
        print("file '{}' is not a valid file".format(args.filename), file=sys.stderr)
