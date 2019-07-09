#!/usr/bin/env python3

import argparse
import logging
import os
from typing import Any, Dict, List, Optional, Set, Dict

import pprint

from kirke.eblearn import ebrunner
from kirke.sampleutils import regexgen
from kirke.utils import ebantdoc4, strutils

# usage: python -m cmd.run_gen_cands_cdnp data/japanese/txt/1005.txt > 1005.out

# NOTE: Remove the following line to get rid of all logging messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


IS_SUPPORT_DOC_CLASSIFICATION = True
DOCCAT_MODEL_FILE_NAME = ebrunner.DOCCAT_MODEL_FILE_NAME


# pylint: disable=too-many-locals
def annotate_document(file_name: str,
                      work_dir: str,
                      model_dir: str,
                      custom_model_dir: str,
                      provision_set: Optional[Set[str]] = None,
                      is_dev_mode: bool = False) -> Dict[str, Any]:
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)
    eb_langdetect_runner = ebrunner.EbLangDetectRunner()

    atext = strutils.loads(file_name)
    doc_lang = eb_langdetect_runner.detect_lang(atext)
    if not doc_lang:
        doc_lang = 'en'
    logging.info("detected language '%s'", doc_lang)

    print('File: {}'.format(file_name))
    print()
    print()    

    if not provision_set:
        provision_set = set([])

    # provision_set = set(['choiceoflaw','change_control', 'indemnify', 'jurisdiction',
    #                      'party', 'warranty', 'termination', 'term']))
    prov_labels_map, _ = eb_runner.annotate_document(file_name,
                                                     provision_set=provision_set,
                                                     work_dir=work_dir,
                                                     doc_lang=doc_lang,
                                                     is_dev_mode=is_dev_mode)

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
    # pprint.pprint(prov_labels_map)

    eb_doccat_runner = None
    doc_catnames = []  # type: List[str]
    if IS_SUPPORT_DOC_CLASSIFICATION and os.path.exists('{}/{}'.format(model_dir,
                                                                       DOCCAT_MODEL_FILE_NAME)):
        eb_doccat_runner = ebrunner.EbDocCatRunner(model_dir)

    logger.info("eb_doccat_runner = %r", eb_doccat_runner)
    if eb_doccat_runner:
        doc_catnames = eb_doccat_runner.classify_document(file_name)

    ebannotations = {}  # type: Dict[str, Any]
    ebannotations['lang'] = doc_lang
    ebannotations['tags'] = doc_catnames
    ebannotations['ebannotations'] = dict(prov_labels_map)

    return ebannotations


def main():
    parser = argparse.ArgumentParser(description='extract candidates.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")

    # parser.add_argument("-c", "--candidate_types", help='candidate types')
    parser.add_argument("file", help='file to extract candidates')

    args = parser.parse_args()
    afile = args.file

    # candidate_types = set(args.candidate_types.split())
    candidate_types = set(['DATE', 'CURRENCY', 'PERCENT', 'NUMBER'])
    # print('candidate_types: {}'.format(candidate_types))

    # WORK_DIR=dir-work
    # SCUT_MODEL_DIR=dir-scut-model
    # CUSTOM_MODEL_DIR=dir-custom-model

    prov_ants_map = annotate_document(afile,
                                      work_dir='dir-work',
                                      model_dir='dir-scut-model',
                                      custom_model_dir='dir-custom-model',
                                      provision_set=candidate_types)

    eb_antdoc = ebantdoc4.text_to_ebantdoc(afile, work_dir='dir-work')
    doc_text = eb_antdoc.text
    doc_len = len(doc_text)

    # print('prov_ants_map:')
    # pprint.pprint(prov_ants_map)

    ebants = prov_ants_map['ebannotations']
        
    for prov in ['CAND_DATE', 'CURRENCY', 'PERCENT', 'NUMBER']:
        print('===== {} ====='.format(prov))
        alist = ebants.get(prov, [])
        for ant_i, ant in enumerate(alist):

            start, end = ant['start'], ant['end']
            pre_st = doc_text[max(start-20, 0):start].replace('\n', ' || ')
            post_st = doc_text[end:min(end+20, doc_len)].replace('\n', ' || ')

            print('\n{} #{}'.format(prov, ant_i))
            print('found:\t{}\t{}\t({}, {})'.format(ant['text'],
                                                    ant['norm'],
                                                    ant['start'],
                                                    ant['end']))                                  
            print('context:\t{} >>>   {}   <<< {}'.format(pre_st, ant['text'], post_st))

        print()
        print()
                                                    
                                                    
                                                    
    
    # for prov, ants in prov_ants_map.items():
    #    print('prov: [{}]'.format(prov))
                                      
if __name__ == '__main__':
    # usage: python -m cmd.run_gen_cands_cdnp data/japanese/txt/1005.txt > 1005.out
    main()
