#!/usr/bin/env python3

import logging
import argparse

from sklearn.externals import joblib

from utils import osutils

from eblearn import ebrunner, ebtext2antdoc, ebannotator
from pprint import pprint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    parser.add_argument('--docs', help='a file containing list of .txt files')
    parser.add_argument('--provisions', help='a comma separate list of provisions')
    parser.add_argument('--work_dir', help='output directory for .arff and .npz')
    parser.add_argument('--model_file', help='trained model to test')


    args = parser.parse_args()
    print("args.docs = [{}]".format(args.docs))
    print("args.work_dir = [{}]".format(args.work_dir))
    print("args.model_file = [{}]".format(args.model_file))

    txt_fn_list = args.docs
    work_dir = args.work_dir

    eb_classifier = joblib.load(args.model_file)
    provision = eb_classifier.provision

    print("provision = {}".format(provision))

    ebantdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))
    
    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    # update the hashmap of annotators
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)

    ant_status = prov_annotator.test_antdoc_list(ebantdoc_list, work_dir)

    ant_status['provision'] = provision
    ant_status['pred_status'] = pred_status

    pprint(ant_status)

    logging.info('Done.')
