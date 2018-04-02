#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path
import re
import requests
import sys


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

eb_files = os.environ['EB_FILES']
eb_models = os.environ['EB_MODELS']
logging.info("eb files is: '%s'", eb_files)
logging.info("eb models is: '%s'", eb_models)

CUSTOM_MODEL_DIR = eb_files + 'pymodel'

# pylint: disable=C0103
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='identify the language')
    parser.add_argument('-v', '--verbosity', help='increase output verbosity')
    parser.add_argument('-d', '--debug', action='store_true', help='print debug information')
    parser.add_argument('-u', '--url', help='url to post the file')
    parser.add_argument('-l', '--lang', action='store_true', help='to detect lang')
    parser.add_argument('--doccat', action='store_true', help='to classify document')

    args = parser.parse_args()
    if args.verbosity:
        print('verbosity turned on')
    if args.debug:
        isDebug = True

    dir_name = CUSTOM_MODEL_DIR
    maybe_fnames = [f for f in os.listdir(dir_name)
                    if (os.path.isfile(os.path.join(dir_name, f))
                        and 'classifier' in f and f.endswith('.pkl'))]

    for fname in maybe_fnames:
        # print("fname = [{}]".format(fname))
        mat = re.match(r'(cust_\d+)(_.*)', fname)
        if mat:
            ofname = '{}/{}'.format(dir_name, fname)
            vfname = '{}/{}.1{}'.format(dir_name, mat.group(1), mat.group(2))
            print("rename %s %s" % (ofname, vfname))
            os.rename(ofname, vfname)
        else:
            print("skip file '{}/{}'".format(dir_name, fname))


    maybe_fnames = [f for f in os.listdir(dir_name)
                    if (os.path.isfile(os.path.join(dir_name, f))
                        and f.startswith('cust_') and f.endswith('.status'))]

    for fname in maybe_fnames:
        # print("fname = [{}]".format(fname))
        mat = re.match(r'(cust_\d+)(\.status)', fname)
        if mat:
            ofname = '{}/{}'.format(dir_name, fname)
            vfname = '{}/{}.1{}'.format(dir_name, mat.group(1), mat.group(2))
            print("rename %s %s" % (ofname, vfname))
            os.rename(ofname, vfname)
        else:
            print("skip file '{}/{}'".format(dir_name, fname))            
