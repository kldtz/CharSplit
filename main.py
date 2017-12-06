#!/usr/bin/env python

import argparse
import configparser
import copy
import json
import logging
import pprint
import re
import sys
import warnings
import re
import json
import time
from collections import defaultdict
import os

from sklearn.externals import joblib

from kirke.docclassifier.unigramclassifier import UnigramDocClassifier
from kirke.eblearn import ebrunner, ebtrainer, provclassifier, scutclassifier
from kirke.eblearn import ebannotator
from kirke.ebrules import rateclassifier
from kirke.utils import osutils, splittrte, strutils
from kirke.ebrules import rateclassifier

config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']
PROV_CLF_VERSION = config['ebrevia.com']['PROV_CLF_VERSION']

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


IS_SUPPORT_DOC_CLASSIFICATION = True

# This trains on ALL data, no separate testing
def train_classifier(provision, txt_fn_list_fn, work_dir, model_dir, is_scut):
    if is_scut:
        eb_classifier = scutclassifier.ShortcutClassifier(provision)
        model_file_name = '{}/{}_scutclassifier.v{}.pkl'.format(model_dir,
                                                                provision,
                                                                SCUT_CLF_VERSION)
    else:
        eb_classifier = provclassifier.ProvisionClassifier(provision)
        model_file_name = '{}/{}_provclassifier.v{}.pkl'.format(model_dir,
                                                                provision,
                                                                PROV_CLF_VERSION)

    # pylint: disable=protected-access
    ebtrainer._train_classifier(txt_fn_list_fn,
                                work_dir,
                                model_file_name,
                                eb_classifier)


# PATH = "/home/jshaw/proj/KirkeDocCat/sample_data2"
def train_doc_classifier(txt_fn_list_fn, model_dir):
    doc_classifier = UnigramDocClassifier()
    model_file_name = model_dir + "/ebrevia_docclassifier.pkl"

    doc_classifier.train(txt_fn_list_fn, model_file_name)


def train_eval_doc_classifier(txt_fn_list_fn, is_step1=False):
    doc_classifier = UnigramDocClassifier()

    doc_classifier.train_and_evaluate(txt_fn_list_fn, is_step1)


def classify_document(file_name, model_dir):
    eb_runner = ebrunner.EbDocCatRunner(model_dir)

    preds = eb_runner.classify_document(file_name)

    pprint(preds)


# This separates out training and testing data, trains only on training data.
def train_annotator(provision, txt_fn_list_fn, work_dir, model_dir, is_scut, is_doc_structure=True):
    if is_scut:
        eb_classifier = scutclassifier.ShortcutClassifier(provision)
        model_file_name = '{}/{}_scutclassifier.v{}.pkl'.format(model_dir,
                                                                provision,
                                                                SCUT_CLF_VERSION)
    else:
        eb_classifier = provclassifier.ProvisionClassifier(provision)
        model_file_name = '{}/{}_provclassifier.v{}.pkl'.format(model_dir,
                                                                provision,
                                                                PROV_CLF_VERSION)
    eval_stats, log_json = ebtrainer.train_eval_annotator_with_trte(provision,
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


def custom_train_annotator(provision, txt_fn_list_fn, work_dir, model_dir,
                           custom_model_dir, is_doc_structure=True):
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)

    # cust_id = '12345'
    provision = 'cust_12345'

    base_model_fname = '{}_scutclassifier.v{}.pkl'.format(provision,
                                                          SCUT_CLF_VERSION)

    eval_status = eb_runner.custom_train_provision_and_evaluate(txt_fn_list_fn,
                                                                provision,
                                                                custom_model_dir,
                                                                base_model_fname,
                                                                is_doc_structure=is_doc_structure,
                                                                work_dir=work_dir)


# test multiple annotators
# pylint: disable=too-many-arguments
def test_annotators(provisions, txt_fn_list_fn, work_dir, model_dir, custom_model_dir,
                    threshold=None):
    provision_set = set([])
    if provisions:
        provision_set = set(provisions.split(','))

    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)
    eval_status, log_json = eb_runner.test_annotators(txt_fn_list_fn, provision_set, threshold)

    # return some json accuracy info
    pprint.pprint(eval_status)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_fn = '_'.join(provision_set)+'-test-' + timestr + ".log"
    strutils.dumps(json.dumps(log_json), log_fn)

# test only 1 annotator
def test_one_annotator(txt_fn_list_fn, work_dir, model_file_name):
    eb_classifier = joblib.load(model_file_name)
    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(txt_fn_list_fn, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    # update the hashmap of annotators
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    ant_status, log_json = prov_annotator.test_antdoc_list(ebantdoc_list)

    ant_status['provision'] = provision
    ant_status['pred_status'] = pred_status

    pprint(ant_status)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_fn = '_'.join(provision_set)+'-test-' + timestr + ".log"
    strutils.dumps(json.dumps(log_json), log_fn)


def test_title_annotator(txt_fn_list_fn, work_dir, model_file_name):

    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(txt_fn_list_fn, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    # update the hashmap of annotators
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    ant_status, _ = prov_annotator.test_antdoc_list(ebantdoc_list)

    ant_status['provision'] = provision
    ant_status['pred_status'] = pred_status

    pprint.pprint(ant_status)


def annotate_document(file_name,
                      work_dir,
                      model_dir,
                      custom_model_dir,
                      provision_set=None,
                      is_doc_structure=True):
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)
    eb_langdetect_runner = ebrunner.EbLangDetectRunner()
   
    atext = strutils.loads(file_name)
    doc_lang = eb_langdetect_runner.detect_lang(atext)
    logging.info("detected language '{}'".format(doc_lang))
    

    # provision_set = set(['choiceoflaw','change_control', 'indemnify', 'jurisdiction',
    #                      'party', 'warranty', 'termination', 'term']))
    prov_labels_map, _ = eb_runner.annotate_document(file_name,
                                                     provision_set=provision_set,
                                                     work_dir=work_dir,
                                                     doc_lang=doc_lang,
                                                     is_doc_structure=is_doc_structure)

    # because special case of 'effectivdate_auto'
    if prov_labels_map.get('effectivedate'):
        effectivedate_annotations = copy.deepcopy(prov_labels_map.get('effectivedate', []))
        for eff_ant in effectivedate_annotations:
            eff_ant['label'] = 'effectivedate_auto'
        prov_labels_map['effectivedate_auto'] = effectivedate_annotations
        del prov_labels_map['effectivedate']


    # prov_labels_map, doc_text = eb_runner.annotate_document(file_name, set(['choiceoflaw','change_control', 'indemnify', 'jurisdiction', 'party', 'warranty', 'termination', 'term']))
    pprint.pprint(prov_labels_map)

    eb_doccat_runner = None
    doccat_model_fn = model_dir + '/ebrevia_docclassifier.pkl'
    if IS_SUPPORT_DOC_CLASSIFICATION and os.path.exists(doccat_model_fn):
        eb_doccat_runner = ebrunner.EbDocCatRunner(model_dir)

    print("eb_doccat_runner = {}".format(eb_doccat_runner))
    if eb_doccat_runner != None:
        doc_catnames = eb_doccat_runner.classify_document(file_name)
        pprint.pprint({'tags': doc_catnames})


def annotate_htmled_document(file_name, work_dir, model_dir, custom_model_dir):
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)

    prov_labels_map, doc_text = eb_runner.annotate_htmled_document(file_name, work_dir=work_dir)
    # prov_labels_map, doc_text = eb_runner.annotate_document(file_name, set(['choiceoflaw','change_control', 'indemnify', 'jurisdiction', 'party', 'warranty', 'termination', 'term']))
    pprint(prov_labels_map)

    eb_doccat_runner = None
    doccat_model_fn = model_dir + '/ebrevia_docclassifier.pkl'
    if IS_SUPPORT_DOC_CLASSIFICATION and os.path.exists(doccat_model_fn):
        eb_doccat_runner = ebrunner.EbDocCatRunner(model_dir)

    if eb_doccat_runner != None:
        doc_catnames = eb_doccat_runner.classify_document(file_name)
        pprint({'tags': doc_catnames})


# TODO, this is the same as ebrunner.annotate_pdfboxed_document?
def annotate_pdfboxed_document(file_name, linfo_file_name, work_dir, model_dir, custom_model_dir):
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)

    prov_labels_map, doc_text = eb_runner.annotate_pdfboxed_document(file_name, linfo_file_name, work_dir=work_dir)

    pprint(prov_labels_map)

    eb_doccat_runner = None
    doccat_model_fn = model_dir + '/ebrevia_docclassifier.pkl'
    if IS_SUPPORT_DOC_CLASSIFICATION and os.path.exists(doccat_model_fn):
        eb_doccat_runner = ebrunner.EbDocCatRunner(model_dir)

    if eb_doccat_runner != None:
        doc_catnames = eb_doccat_runner.classify_document(file_name)
        pprint({'tags': doc_catnames})


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
    parser.add_argument('--custom_model_dir', required=True,
                        help='output directory for custom trained models')
    parser.add_argument('--scut', action='store_true', help='build short-cut trained models')
    parser.add_argument('--model_file', help='model file name to test a doc')
    parser.add_argument('--threshold', type=float, default=0.24, help='threshold for annotator')

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
        custom_train_annotator(provision,
                               txt_fn_list_fn,
                               work_dir,
                               model_dir,
                               custom_model_dir,
                               is_doc_structure=True)
    elif cmd == 'test_annotators':
        # if no --provisions is specified, all annotators are tested
        test_annotators(args.provisions, txt_fn_list_fn, work_dir, model_dir,
                        custom_model_dir, threshold=args.threshold)
    elif cmd == 'test_one_annotator':
        if not args.model_file:
            print('please specify --model_file', file=sys.stderr)
            sys.exit(1)
        test_one_annotator(txt_fn_list_fn, work_dir, args.model_file)
    elif cmd == 'annotate_document':
        if not args.doc:
            print('please specify --doc', file=sys.stderr)
            sys.exit(1)
        annotate_document(args.doc, work_dir, model_dir, custom_model_dir, is_doc_structure=True)
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
                                       is_doc_structure=False)
    elif cmd == 'train_doc_classifier':
        # doccatutils.train_doc_classifier(txt_fn_list_fn, model_dir)
        train_doc_classifier(txt_fn_list_fn, model_dir)
    elif cmd == 'train_eval_doc_classifier':
        # doccatutils.train_eval_doc_classifier(txt_fn_list_fn)
        train_eval_doc_classifier(txt_fn_list_fn)
    elif cmd == 'train_eval_doc_step1_classifier':
        # doccatutils.train_eval_doc_classifier(txt_fn_list_fn)
        train_eval_doc_classifier(txt_fn_list_fn, is_step1=True)
    elif cmd == 'classify_doc':
        classify_document(args.doc, model_dir)
    else:
        print("unknown command: '{}'".format(cmd))

    logging.info('Done.')
