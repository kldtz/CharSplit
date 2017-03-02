#!/usr/bin/env python

import logging
import argparse
import json
from pprint import pprint

from eblearn import ebrunner


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Annotate a document in parallel.')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    parser.add_argument('--doc', help='a .txt files')
    parser.add_argument('--work_dir', help='output directory for .arff and .npz', default='sample_data2.feat')
    parser.add_argument('--model_dir', help='output directory for trained models', default='sample_data2.model')

    args = parser.parse_args()
    print("args.doc = [{}]".format(args.doc))
    print("args.work_dir = [{}]".format(args.work_dir))
    print("args.model_dir = [{}]".format(args.model_dir))
    custom_model_dir = args.model_dir.replace('.model', 'custmodel')

    eb_runner = ebrunner.EbRunner(args.model_dir, args.work_dir, custom_model_dir)
    prov_labels_map = eb_runner.annotate_document(args.doc)
    pprint(prov_labels_map)
    
    logging.info('Done.')




