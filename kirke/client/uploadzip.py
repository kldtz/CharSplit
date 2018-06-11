#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import requests
import sys


# pylint: disable=C0103
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='identify the language')
    parser.add_argument('-v', '--verbosity', help='increase output verbosity')
    parser.add_argument('-d', '--debug', action='store_true', help='print debug information')
    parser.add_argument('-u', '--url', help='url to post the file')
    parser.add_argument('-l', '--lang', action='store_true', help='to detect lang')
    parser.add_argument('--doccat', action='store_true', help='to classify document')

    parser.add_argument('filename')

    args = parser.parse_args()
    if args.verbosity:
        print('verbosity turned on')
    if args.debug:
        isDebug = True

    url = 'http://127.0.0.1:8000/custom-train-import'
    # use url='http://127.0.0.1:8000/detect-langs' to detect top langs with probabilities
    if args.url:
        url = args.url

    payload = {}
    # payload = {'types': 'party'}
    # payload = {'types': 'party,change_control'}
    # payload = {'types': 'termination,term,confidentiality,cust_3566'}
    # payload = {'types': 'term,cust_2253'}
    zip_file = Path(args.filename)
    if zip_file.is_file():
        files = {'file': open(args.filename, 'rb')}

        req = requests.post(url, files=files, data={})
        print(req.text)
    else:
        print("file '{}' is not a valid file".format(args.filename), file=sys.stderr)
