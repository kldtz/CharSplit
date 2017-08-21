#!/usr/bin/env python

import argparse
import logging
from pprint import pprint
import sys
import warnings
import re
import json

from collections import defaultdict
import os

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from kirke.eblearn import ebrunner, ebtrainer, provclassifier, scutclassifierv1_2
from kirke.eblearn import ebtext2antdoc, ebannotator
from kirke.utils import osutils, splittrte, strutils

from kirke.docstruct import docreader

from kirke.ebrules import rateclassifier

SCUT_CLF_VERSION = '1.2'
PROV_CLF_VERSION = '1.0'

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# This trains on ALL data, no separate testing
def train_classifier(provision, txt_fn_list_fn, work_dir, model_dir, is_scut):
    if is_scut:
        eb_classifier = scutclassifierv1_2.ShortcutClassifierV1_2(provision)
        model_file_name = '{}/{}'.format(model_dir, provision + '_scutclassifier.v{}.pkl'.format(SCUT_CLF_VERSION))
    else:
        eb_classifier = provclassifier.ProvisionClassifier(provision)
        model_file_name = '{}/{}'.format(model_dir, provision + '_provclassifier.v{}.pkl'.format(PROV_CLF_VERSION))

    ebtrainer._train_classifier(txt_fn_list_fn,
                                work_dir,
                                model_file_name,
                                eb_classifier)


# This separates out training and testing data, trains only on training data.
def train_annotator(provision, txt_fn_list_fn, work_dir, model_dir, is_scut, is_doc_structure=True):
    if is_scut:
        eb_classifier = scutclassifierv1_2.ShortcutClassifierV1_2(provision)
        model_file_name = '{}/{}'.format(model_dir, provision + '_scutclassifier.v{}.pkl'.format(SCUT_CLF_VERSION))
    else:
        eb_classifier = provclassifier.ProvisionClassifier(provision)
        model_file_name = '{}/{}'.format(model_dir, provision + '_provclassifier.v{}.pkl'.format(PROV_CLF_VERSION))

    ebtrainer.train_eval_annotator_with_trte(provision,
                                             work_dir,
                                             model_dir,
                                             model_file_name,
                                             eb_classifier,
                                             is_doc_structure=is_doc_structure)

def eval_line_annotator_with_trte(provision,
                                  work_dir,
                                  model_dir,
                                  is_doc_structure=True):
    ebtrainer.eval_line_annotator_with_trte(provision,
                                            work_dir=work_dir,
                                            model_dir=model_dir,
                                            is_doc_structure=is_doc_structure)

def eval_ml_rule_annotator_with_trte(provision,
                                     work_dir,
                                     model_dir):
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)
    eb_runner.eval_ml_rule_annotator_with_trte(provision,
                                               work_dir=work_dir,
                                               model_dir=model_dir)


def custom_train_annotator(provision, txt_fn_list_fn, work_dir, model_dir, custom_model_dir, is_doc_structure=True):
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)

    # cust_id = '12345'
    provision = 'cust_12345'

    # model_file_name = custom_model_dir + '/' +  provision + "_scutclassifier.pkl"
    eval_status = eb_runner.custom_train_provision_and_evaluate(txt_fn_list_fn,
                                                                provision,
                                                                custom_model_dir,
                                                                is_doc_structure=is_doc_structure,
                                                                is_combine_line=False)

# test multiple annotators    
def test_annotators(provisions, txt_fn_list_fn, word_dir, model_dir, custom_model_dir, threshold=None):
    provision_set = set([])
    if provisions:
        provision_set = set(provisions.split(','))

    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)
    eval_status = eb_runner.test_annotators(txt_fn_list_fn, provision_set, threshold)

    # return some json accuracy info
    pprint(eval_status)

# test only 1 annotator    
def test_one_annotator(txt_fn_list_fn, work_dir, model_file_name):
    eb_classifier = joblib.load(model_file_name)
    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fn_list_fn, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    # update the hashmap of annotators
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    ant_status = prov_annotator.test_antdoc_list(ebantdoc_list)

    ant_status['provision'] = provision
    ant_status['pred_status'] = pred_status

    pprint(ant_status)


def test_title_annotator(txt_fn_list_fn, work_dir, model_file_name):

    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fn_list_fn, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    # update the hashmap of annotators
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    ant_status = prov_annotator.test_antdoc_list(ebantdoc_list)

    ant_status['provision'] = provision
    ant_status['pred_status'] = pred_status

    pprint(ant_status)


def annotate_document(file_name, work_dir, model_dir, custom_model_dir):
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)

    pdf_offsets_filename = file_name.replace('.txt', '.offsets.json')

    #prov_labels_map, doc_text = eb_runner.annotate_document(file_name,
    #                                                        work_dir=work_dir)

    if os.path.exists(pdf_offsets_filename):
        prov_labels_map, doc_text = eb_runner.annotate_pdfboxed_document(file_name,
                                                                         pdf_offsets_filename,
                                                                         work_dir=work_dir)
    else:
        prov_labels_map, doc_text = eb_runner.annotate_htmled_document(file_name,
                                                                       work_dir=work_dir)

    # prov_labels_map, doc_text = eb_runner.annotate_document(file_name, set(['choiceoflaw','change_control', 'indemnify', 'jurisdiction', 'party', 'warranty', 'termination', 'term']))
    pprint(prov_labels_map)

def annotate_htmled_document(file_name, work_dir, model_dir, custom_model_dir):
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)

    prov_labels_map, doc_text = eb_runner.annotate_htmled_document(file_name, work_dir=work_dir)
    # prov_labels_map, doc_text = eb_runner.annotate_document(file_name, set(['choiceoflaw','change_control', 'indemnify', 'jurisdiction', 'party', 'warranty', 'termination', 'term']))
    pprint(prov_labels_map)


# TODO, this is the same as ebrunner.annotate_pdfboxed_document?
def annotate_pdfboxed_document(file_name, linfo_file_name, work_dir, model_dir, custom_model_dir):
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)

    prov_labels_map, doc_text = eb_runner.annotate_pdfboxed_document(file_name, linfo_file_name, work_dir=work_dir)

    pprint(prov_labels_map)    


def annotate_doc_party(fn_list_fn, work_dir, model_dir, custom_model_dir, threshold=None):
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)
    if threshold:
        party_annotator = eb_runner.provision_annotator_map['party']
        party_annotator.threshold = threshold
    with open(fn_list_fn, 'rt') as fin:
        for line in fin:
            txt_fn = line.strip()
            eb_runner.annotate_provision_in_document(txt_fn, provision='party')

            
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument("--cmd", required=True, help='command to run')
    parser.add_argument('--doc', help='a file to be annotated')
    parser.add_argument('--docs', help='a file containing list of .txt files')
    parser.add_argument('--provision', help='the provision to train')
    parser.add_argument('--provisions', help='the provisions to split files')
    parser.add_argument('--provfiles_dir', help='directory with all provision file info')
    parser.add_argument('--work_dir', required=True, help='output directory for .corenlp.json')
    parser.add_argument('--model_dir', help='output directory for trained models')
    parser.add_argument('--model_dirs', help='output directory for trained models')
    parser.add_argument('--custom_model_dir', required=True, help='output directory for custom trained models')
    parser.add_argument('--scut', action='store_true', help='build short-cut trained models')
    parser.add_argument('--model_file', help='model file name to test a doc')
    parser.add_argument('--threshold', type=float, help='threshold for annotator')

    args = parser.parse_args()
    cmd = args.cmd
    provision = args.provision
    txt_fn_list_fn = args.docs
    work_dir = args.work_dir
    model_dir = args.model_dir
    custom_model_dir = args.custom_model_dir

    if cmd == 'train_classifier':
        train_classifier(provision, txt_fn_list_fn, work_dir, model_dir, args.scut)
    elif cmd == 'train_annotator':
        train_annotator(provision,
                        txt_fn_list_fn,
                        work_dir,
                        model_dir,
                        args.scut,
                        is_doc_structure=True)
    elif cmd == 'custom_train_annotator':
        custom_train_annotator(provision, txt_fn_list_fn, work_dir, model_dir, custom_model_dir, is_doc_structure=True)
    elif cmd == 'test_annotators':
        # if no --provisions is specified, all annotators are tested
        test_annotators(args.provisions, txt_fn_list_fn, work_dir, model_dir, custom_model_dir,
                        threshold=args.threshold)
    elif cmd == 'test_one_annotator':
        if not args.model_file:
            print('please specify --model_file', file=sys.stderr)
            sys.exit(1)
        test_one_annotator(txt_fn_list_fn, work_dir, args.model_file)
    elif cmd == 'annotate_document':
        if not args.doc:
            print('please specify --doc', file=sys.stderr)
            sys.exit(1)
        annotate_document(args.doc, work_dir, model_dir, custom_model_dir)
    elif cmd == 'annotate_htmled_document':
        if not args.doc:
            print('please specify --doc', file=sys.stderr)
            sys.exit(1)
        annotate_htmled_document(args.doc, work_dir, model_dir, custom_model_dir)
    elif cmd == 'annotate_pdfboxed_document':
        if not args.doc:
            print('please specify --doc', file=sys.stderr)
            sys.exit(1)
        linfo_fn = args.doc.replace('.txt', '.lineinfo.json')
        annotate_pdfboxed_document(args.doc, linfo_fn, work_dir, model_dir, custom_model_dir)
    elif cmd == 'annotate_doc_party':
        if not args.docs:
            print('please specify --docs', file=sys.stderr)
            sys.exit(1)
        annotate_doc_party(args.docs, work_dir, model_dir, custom_model_dir,
                           threshold=args.threshold)
    elif cmd == 'eval_line_annotator':
        if not args.provision:
            print('please specify --provision', file=sys.stderr)
            sys.exit(1)
        eval_line_annotator_with_trte(args.provision,
                                      work_dir,
                                      model_dir,
                                      is_doc_structure=True)
    elif cmd == 'eval_ml_rule_annotator':
        if not args.provision:
            print('please specify --provision', file=sys.stderr)
            sys.exit(1)
        eval_ml_rule_annotator_with_trte(args.provision,
                                         work_dir,
                                         model_dir)
    elif cmd == 'split_provision_trte':
        if not args.provfiles_dir:
            print('please specify --provfiles_dir', file=sys.stderr)
            sys.exit(1)        
        if not args.model_dirs:
            print('please specify --model_dirs', file=sys.stderr)
            sys.exit(1)        
        model_dir_list = args.model_dirs.split(',')
        # for HTML document, without doc structure
        # is_doc_structure has to be false.
        splittrte.split_provision_trte(args.provfiles_dir,
                                       work_dir,
                                       model_dir_list,
                                       is_doc_structure=True)
    else:
        print("unknown command: '{}'".format(cmd))

    logging.info('Done.')


"""
    elif cmd == 'split_provisions_from_posdocs':
        if not args.provisions:
            print('please specify --provisions', file=sys.stderr)
            sys.exit(1)
        split_provisions_from_posdocs(args.provisions, txt_fn_list_fn, work_dir, model_dir)
    elif cmd == 'split_provision_trte':
        if not args.provisions:
            print('please specify --provisions', file=sys.stderr)
            sys.exit(1)
        if not args.model_dirs:
            print('please specify --model_dirs', file=sys.stderr)
            sys.exit(1)        
        provision_list = args.provisions.split(',')
        model_dir_list = args.model_dirs.split(',')
        split_provision_trte(provision_list, txt_fn_list_fn, work_dir, model_dir_list)
"""
    
