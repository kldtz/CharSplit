#!/usr/bin/env python3

import argparse
import requests


def main():
    parser = argparse.ArgumentParser(description='identify the language')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument("-u", "--url", help="url to post the file")
    parser.add_argument("filename")

    args = parser.parse_args()
    if args.verbosity:
        print("verbosity turned on")

    payload = {'dev-mode': True}

    url = 'http://127.0.0.1:8000/detect-lang'
    if args.url:
        url = args.url

    if args.filename:
        files = {'file': open(args.filename, 'rt')}
        req = requests.post(url, files=files, data=payload)
        print(req.text)

if __name__ == '__main__':
    main()
