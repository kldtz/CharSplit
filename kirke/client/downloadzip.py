#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import re
import requests
import sys
import zipfile, io


def download_file(url: str, local_filename: str) -> str:
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    print("finished saving '{}".format(local_filename))
    return local_filename


# pylint: disable=C0103
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='identify the language')
    parser.add_argument('-v', '--verbosity', help='increase output verbosity')
    parser.add_argument('-d', '--debug', action='store_true', help='print debug information')
    parser.add_argument('-u', '--url', help='url to post the file')
    parser.add_argument('-l', '--lang', action='store_true', help='to detect lang')
    # parser.add_argument('--doccat', action='store_true', help='to classify document')

    # parser.add_argument('filename')

    args = parser.parse_args()
    if args.verbosity:
        print('verbosity turned on')
    if args.debug:
        isDebug = True

    url = 'http://127.0.0.1:8000/custom-train-export/cust_12345.1003'
    # use url='http://127.0.0.1:8000/detect-langs' to detect top langs with probabilities
    if args.url:
        url = args.url

    # req = requests.get(url)
    # z = zipfile.ZipFile(io.BytesIO(req.content))
    # z.extractall()
    # z.write('cust_12345.aaa.zip')

    # zipfile = urllib2.urlopen(url)

    mat = re.search(r'(cust_[\d\.]+)', url)
    cust_id_ver = mat.group(1)

    local_filename = '{}.custom_models'.format(cust_id_ver)

    download_file(url, local_filename)
