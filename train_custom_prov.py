#!/usr/bin/env python

import argparse
import logging

from eblearn import ebrunner, scutclassifier 
from utils import osutils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    parser.add_argument('--docs', help='a file containing list of .txt files')
    parser.add_argument('--work_dir', required=True, help='output directory for .corenlp.json')
    parser.add_argument('--model_dir', required=True, help='output directory for trained models')
    parser.add_argument('--custom_model_dir', required=True, help='output directory for custom trained models')    

    args = parser.parse_args()
    txt_fn_list_fn = args.docs
    work_dir = args.work_dir
    model_dir = args.model_dir
    custom_model_dir = args.custom_model_dir
    osutils.mkpath(custom_model_dir)

    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)
    
    # cust_id = '12345'
    provision = 'cust_12345'

    eb_classifier = scutclassifier.ShortcutClassifier(provision)
    # model_file_name = custom_model_dir + '/' +  provision + "_scutclassifier.pkl"
    eval_status = eb_runner.custom_train_provision_and_evaluate(txt_fn_list_fn,
                                                                provision,
                                                                custom_model_dir)

    logging.info('Done.')
