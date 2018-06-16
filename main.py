#!/usr/bin/env python3

import argparse
import configparser
import copy
import logging
import os
import pprint
import sys
from typing import List, Optional, Set

# pylint: disable=import-error
from sklearn.externals import joblib

from kirke.docclassifier.unigramclassifier import UnigramDocClassifier
from kirke.docclassifier import doccatsplittrte
from kirke.eblearn import ebannotator, ebrunner, ebtrainer, provclassifier, scutclassifier
# from kirke.ebrules import rateclassifier
# pylint: disable=unused-import
from kirke.eblearn.ebclassifier import EbClassifier
from kirke.utils import corenlputils, ebantdoc4, osutils, splittrte, strutils

# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

# NOTE: Remove the following line to get rid of all logging messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']
PROV_CLF_VERSION = config['ebrevia.com']['PROV_CLF_VERSION']
ANNOTATOR_CLF_VERSION = config['ebrevia.com']['ANNOTATOR_CLF_VERSION']

DOCCAT_MODEL_FILE_NAME = ebrunner.DOCCAT_MODEL_FILE_NAME


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# start corenlp server
corenlputils.init_corenlp_server()

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
def train_doc_classifier(txt_fn_list_fn: str, model_dir: str) -> None:
    doc_classifier = UnigramDocClassifier()
    model_file_name = '{}/{}'.format(model_dir, DOCCAT_MODEL_FILE_NAME)
    osutils.mkpath(model_dir)
    doc_classifier.train(txt_fn_list_fn, model_file_name)


def train_eval_doc_classifier(txt_fn_list_fn: str, model_dir: str) -> None:
    doc_classifier = UnigramDocClassifier()
    # This does not save model, only train and eval.
    # We are passing in the model file name to save the status of the production models
    # for future references.
    model_file_name = '{}/{}'.format(model_dir, DOCCAT_MODEL_FILE_NAME)
    doc_classifier.train_and_evaluate(txt_fn_list_fn,
                                      prod_status_fname=model_file_name.replace('.pkl', '.status'))


def classify_document(file_name: str, model_dir: str) -> None:
    eb_runner = ebrunner.EbDocCatRunner(model_dir)

    preds = eb_runner.classify_document(file_name)

    pprint.pprint(preds)


# This separates out training and testing data, trains only on training data.
# pylint: disable=too-many-arguments
def train_annotator(provision: str,
                    work_dir: str,
                    model_dir: str,
                    is_scut: bool,
                    is_cache_enabled: bool = True,
                    is_doc_structure: bool = True) -> None:
    if is_scut:
        eb_classifier = scutclassifier.ShortcutClassifier(provision)  # type: EbClassifier
        model_file_name = '{}/{}_scutclassifier.v{}.pkl'.format(model_dir,
                                                                provision,
                                                                SCUT_CLF_VERSION)
    else:
        eb_classifier = provclassifier.ProvisionClassifier(provision)
        model_file_name = '{}/{}_provclassifier.v{}.pkl'.format(model_dir,
                                                                provision,
                                                                PROV_CLF_VERSION)

    ebtrainer.train_eval_annotator_with_trte(provision,
                                             work_dir,
                                             model_dir,
                                             model_file_name,
                                             eb_classifier,
                                             is_cache_enabled=is_cache_enabled,
                                             is_doc_structure=is_doc_structure)

def train_span_annotator(label: str,
                         nbest: int,
                         candidate_types: List[str],
                         work_dir: str,
                         model_dir: str) -> None:
    if candidate_types == ['SENTENCE']:
        train_annotator(label, work_dir, model_dir, True)
    else:
        ebtrainer.train_eval_span_annotator(label,
                                            383838,
                                            'en',
                                            nbest,
                                            candidate_types,
                                            work_dir,
                                            model_dir)


def eval_span_annotator(label: str,
                        candidate_types: List[str],
                        txt_fn_list_fn: str,
                        work_dir: str,
                        model_dir: str):

    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir=model_dir)
    # ebtrainer.eval_rule_annotator_with_trte(label, is_train_mode=True)
    eb_runner.eval_span_annotator(label,
                                  candidate_types,
                                  txt_fn_list_fn)


def eval_rule_annotator(label: str,
                        model_dir: str,
                        is_train_mode: bool = False):
    # ebtrainer.eval_rule_annotator_with_trte(label, is_train_mode=True)
    ebtrainer.eval_rule_annotator_with_trte(label,
                                            model_dir=model_dir,
                                            is_train_mode=is_train_mode)


def eval_line_annotator_with_trte(provision: str,
                                  txt_fn_list_fn: str,
                                  work_dir: str,
                                  is_doc_structure: bool = True):
    """Test line annotators based on txt_fn_list."""
    ebtrainer.eval_line_annotator_with_trte(provision,
                                            txt_fn_list_fn,
                                            work_dir=work_dir,
                                            is_doc_structure=is_doc_structure)


def eval_mlxline_annotator_with_trte(provision: str,
                                     txt_fn_list_fn: str,
                                     work_dir: str,
                                     model_dir: str):
    # custom_model_dir is not used
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir='dir-scut-model')
    eb_runner.eval_mlxline_annotator_with_trte(provision,
                                               txt_fn_list_fn,
                                               work_dir=work_dir)


# pylint: disable=too-many-arguments
def custom_train_annotator(provision: str,
                           candidate_types: List[str],
                           nbest: int,
                           txt_fn_list_fn: str,
                           work_dir: str,
                           model_dir: str,
                           custom_model_dir: str) -> None:
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)

    # cust_id = '12345'
    # provision = 'cust_12345'

    base_model_fname = '{}_scutclassifier.v{}.pkl'.format(provision,
                                                          SCUT_CLF_VERSION)

    unused_eval_status, unused_log_json = \
        eb_runner.custom_train_provision_and_evaluate(txt_fn_list_fn,
                                                      provision,
                                                      custom_model_dir,
                                                      base_model_fname,
                                                      candidate_types=candidate_types,
                                                      nbest=nbest,
                                                      model_num=383838,
                                                      work_dir=work_dir)


# test multiple annotators
# pylint: disable=too-many-arguments
def test_annotators(provisions,
                    txt_fn_list_fn: str,
                    work_dir: str,
                    model_dir: str,
                    custom_model_dir: str,
                    out_dir: str = '',
                    threshold: Optional[float] = None) -> None:
    provision_set = set([])  # type: Set[str]
    if provisions:
        provision_set = set(provisions.split(','))

    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)
    prov_antstat_logjson_map = eb_runner.test_annotators(txt_fn_list_fn,
                                                         provision_set,
                                                         threshold)

    # return some json accuracy info
    for provision in provision_set:
        ant_status, log_json = prov_antstat_logjson_map[provision]
        print("{}:".format(provision))
        pprint.pprint(ant_status)
        print("log_json:")
        pprint.pprint(log_json)

        print("ant_status:")
        print(ant_status)

        if out_dir:
            osutils.mkpath(out_dir)
            out_fname = '{}/{}.status'.format(out_dir, provision)
            with open(out_fname, 'wt') as fout:
                print(ant_status, file=fout)
                print('wrote "{}"'.format(out_fname))


# test only 1 annotator
# not sure anyone calling this
def test_one_annotator(txt_fn_list_fn: str,
                       work_dir: str,
                       model_file_name: str):
    eb_classifier = joblib.load(model_file_name)
    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebantdoc4.doclist_to_ebantdoc_list(txt_fn_list_fn, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    unused_provision_status_map = {'provision': provision,
                                   'pred_status': pred_status}

    # update the hashmap of annotators
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    ant_status, unused_log_json = prov_annotator.test_antdoc_list(ebantdoc_list)

    ant_status['provision'] = provision
    ant_status['pred_status'] = pred_status

    pprint.pprint(ant_status)


# pylint: disable=too-many-locals
def annotate_document(file_name: str,
                      work_dir: str,
                      model_dir: str,
                      custom_model_dir: str,
                      provision_set: Optional[Set[str]] = None,
                      is_doc_structure: bool = True) -> None:
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)
    eb_langdetect_runner = ebrunner.EbLangDetectRunner()

    atext = strutils.loads(file_name)
    doc_lang = eb_langdetect_runner.detect_lang(atext)
    if not doc_lang:
        doc_lang = 'en'
    logging.info("detected language '%s'", doc_lang)

    if not provision_set:
        provision_set = set([])

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

    # prov_labels_map, doc_text = eb_runner.annotate_document(file_name,
    #                                                         set(['choiceoflaw','change_control',
    #                                                              'indemnify', 'jurisdiction',
    #                                                              'party', 'warranty',
    #                                                              'termination', 'term']))
    pprint.pprint(prov_labels_map)

    eb_doccat_runner = None
    if IS_SUPPORT_DOC_CLASSIFICATION and os.path.exists('{}/{}'.format(model_dir,
                                                                       DOCCAT_MODEL_FILE_NAME)):
        eb_doccat_runner = ebrunner.EbDocCatRunner(model_dir)

    print("eb_doccat_runner = {}".format(eb_doccat_runner))
    if eb_doccat_runner:
        doc_catnames = eb_doccat_runner.classify_document(file_name)
        pprint.pprint({'tags': doc_catnames})


# pylint: disable=too-many-branches, too-many-statements
def main():
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
    parser.add_argument('--custom_model_dir', help='output directory for custom trained models')
    parser.add_argument('--out_dir', help='output directory for testing annotators')
    parser.add_argument('--scut', action='store_true', help='build short-cut trained models')
    parser.add_argument('--model_file', help='model file name to test a doc')
    parser.add_argument('--threshold', type=float, default=0.24, help='threshold for annotator')
    parser.add_argument('--candidate_types', default='SENTENCE',
                        help='types of candidate generator')
    parser.add_argument('--cache_disabled', action="store_true",
                        help='disable loading cached files')
    parser.add_argument('--nbest', default=-1, help='number of annotations per doc')
    # only for eval_rule_annotator
    parser.add_argument('--is_train_mode', action="store_true",
                        help="training mode for eval_rule_annotator")

    args = parser.parse_args()
    cmd = args.cmd
    provision = args.provision
    txt_fn_list_fn = args.docs
    work_dir = args.work_dir
    model_dir = args.model_dir
    custom_model_dir = args.custom_model_dir
    if args.cache_disabled:
        is_cache_enabled = False
    else:
        is_cache_enabled = True


    if cmd == 'train_classifier':  # jshaw, nobody should be using this?
        train_classifier(provision, txt_fn_list_fn, work_dir, model_dir, args.scut)
    elif cmd == 'train_annotator':
        train_annotator(provision,
                        work_dir,
                        model_dir,
                        args.scut,
                        is_cache_enabled=is_cache_enabled,
                        is_doc_structure=True)
    elif cmd == 'train_span_annotator':
        train_span_annotator(provision,
                             args.nbest,
                             candidate_types=args.candidate_types.split(','),
                             work_dir=work_dir,
                             model_dir=model_dir)
    elif cmd == 'eval_span_annotator':
        eval_span_annotator(provision,
                            candidate_types=args.candidate_types.split(','),
                            txt_fn_list_fn=txt_fn_list_fn,
                            work_dir=work_dir,
                            model_dir=model_dir)
    elif cmd == 'eval_rule_annotator':
        eval_rule_annotator(provision,
                            model_dir=model_dir,
                            is_train_mode=args.is_train_mode)
    elif cmd == 'custom_train_annotator':
        custom_train_annotator(provision,
                               args.candidate_types.split(','),
                               int(args.nbest),
                               txt_fn_list_fn,
                               work_dir,
                               model_dir,
                               custom_model_dir)
    elif cmd == 'test_annotators':
        # if no --provisions is specified, all annotators are tested
        out_dir = args.out_dir
        test_annotators(args.provisions,
                        txt_fn_list_fn,
                        work_dir,
                        model_dir,
                        custom_model_dir,
                        out_dir=out_dir,
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
        annotate_document(args.doc, work_dir, model_dir, custom_model_dir, is_doc_structure=True)
    elif cmd == 'eval_line_annotator':
        if not args.provision:
            print('please specify --provision', file=sys.stderr)
            sys.exit(1)
        if not args.docs:
            print('please specify --docs', file=sys.stderr)
            sys.exit(1)
        eval_line_annotator_with_trte(args.provision,
                                      txt_fn_list_fn,
                                      work_dir,
                                      is_doc_structure=True)
    elif cmd == 'eval_mlxline_annotator':
        if not args.provision:
            print('please specify --provision', file=sys.stderr)
            sys.exit(1)
        if not args.docs:
            print('please specify --docs', file=sys.stderr)
            sys.exit(1)
        eval_mlxline_annotator_with_trte(args.provision,
                                         txt_fn_list_fn,
                                         work_dir=work_dir,
                                         model_dir=model_dir)
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
    elif cmd == 'split_doccat_trte':
        doccatsplittrte.split_doccat_trte(txt_fn_list_fn)
    elif cmd == 'train_doc_classifier':
        train_doc_classifier(txt_fn_list_fn, model_dir)
    elif cmd == 'train_eval_doc_classifier':
        train_eval_doc_classifier(txt_fn_list_fn, model_dir)
    elif cmd == 'classify_doc':
        classify_document(args.doc, model_dir)
    else:
        print("unknown command: '{}'".format(cmd))

    logging.info('Done.')


if __name__ == '__main__':
    main()
