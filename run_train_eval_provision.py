#!/usr/bin/env python3

import json
import logging
import argparse
import os.path

from sklearn.externals import joblib

from utils import strutils, corenlputils, csrutils, osutils

from collections import defaultdict

from eblearn import provclassifier, scutclassifier
from eblearn import ebannotator, ebtext2antdoc, ebtrainer
from pprint import pprint

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

    ebtrainer.train_eval_annotator(provision,
                                   txt_fn_list,
                                   work_dir,
                                   model_dir,
                                   model_file_name,
                                   eb_classifier)
    
    logging.info('Done.')
