#!/usr/bin/env python3

import argparse
import json
import logging
import os.path

from eblearn import ebtrainer, provclassifier, scutclassifier 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    parser.add_argument('--docs', help='a file containing list of .txt files')
    parser.add_argument('--provision', help='the provisio to train')
    parser.add_argument('--work_dir', help='output directory for .corenlp.json')
    parser.add_argument('--model_dir', help='output directory for trained models')
    parser.add_argument('--scut', action='store_true', help='build short-cut trained models')    

    args = parser.parse_args()
    provision = args.provision
    txt_fn_list = args.docs
    work_dir = args.work_dir
    model_dir = args.model_dir

    if args.scut:
        eb_classifier = scutclassifier.ShortcutClassifier(provision)
        model_file_name = model_dir + '/' +  provision + "_scutclassifier.pkl"
    else:
        eb_classifier = provclassifier.ProvisionClassifier(provision)
        model_file_name = model_dir + '/' +  provision + "_provclassifier.pkl"
    
    ebtrainer._train_classifier(provision,
                                txt_fn_list,
                                work_dir,
                                model_file_name,
                                eb_classifier)

    logging.info('Done.')
