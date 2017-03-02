#!/usr/bin/env python3

import logging
import argparse
from pprint import pprint

from eblearn import ebrunner
from utils import osutils


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    parser.add_argument('--docs', help='a file containing list of .txt files')
    parser.add_argument('--provisions', help='a comma separate list of provisions')
    parser.add_argument('--work_dir', help='output directory for .arff and .npz')
    parser.add_argument('--model_dir', help='output directory for trained models')
    # no need for this, depends on 'model_dir'
    # parser.add_argument('--scut', action='store_true', help='build short-cut trained models')        

    args = parser.parse_args()
    custom_model_dir = args.model_dir.replace('.model', '.custmodel')
    model_dir = args.model_dir

    provision_list = None
    if args.provisions:
        provision_list = args.provisions.split(',')

    txt_fn_list_fn = args.docs
        
    eb_runner = ebrunner.EbRunner(model_dir, args.work_dir, custom_model_dir)
    eval_status = eb_runner.test_annotators(txt_fn_list_fn, provision_list)

    # return some json accuracy info
    pprint(eval_status)

    logging.info('Done.')
