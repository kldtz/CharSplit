#!/usr/bin/env python

import argparse
import logging
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='identify the language')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    parser.add_argument("filename")

    args = parser.parse_args()
    if args.verbosity:
        print("verbosity turned on")
    if args.debug:
        isDebug= True
    if args.filename:
        url = 'http://127.0.0.1:8000/detect-lang'
        files = {'file': open(args.filename, 'rt', encoding='utf-8')}
        r = requests.post(url, files=files)
        print(r.text)
