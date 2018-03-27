#!/usr/bin/env python3

import argparse
import logging
import os
import shutil

from kirke.eblearn import ebrunner
from kirke.utils import ebantdoc2, osutils, strutils
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    txt_file_name = args.file

    work_dir = '/tmp/data-from-web'
    model_dir = '/tmp/dir-scut-model'
    custom_model_dir = '"/tmp/dir-custom-model'
    is_doc_structure = True
    doc_lang = 'en'

    """
    provision_set = set(['term', 'cust_3'])

    
    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)
    eb_langdetect_runner = ebrunner.EbLangDetectRunner()

    atext = strutils.loads(txt_file_name)
    doc_lang = eb_langdetect_runner.detect_lang(atext)
    logging.info("detected language '{}'".format(doc_lang))


    # provision_set = set(['choiceoflaw','change_control', 'indemnify', 'jurisdiction',
    #                      'party', 'warranty', 'termination', 'term']))
    prov_labels_map, _ = eb_runner.annotate_document(txt_file_name,
                                                     provision_set=provision_set,
                                                     work_dir=work_dir,
                                                     doc_lang=doc_lang,
                                                     is_doc_structure=is_doc_structure)

    print(prov_labels_map)

    """

    is_cache_enabled = True
    is_bespoke_mode = True

    offsets_fname = txt_file_name.replace('.txt', '.offsets.json')
    ant_fname = txt_file_name.replace('.txt', '.ant')
    
    txt_base_fname = os.path.basename(txt_file_name)
    offsets_base_fname = os.path.basename(offsets_fname)
    ant_base_fname = os.path.basename(ant_fname)

    full_txt_fname = '{}/{}'.format(work_dir, txt_base_fname)        
    full_offsets_fname = '{}/{}'.format(work_dir, offsets_base_fname)
    full_ant_fname = '{}/{}'.format(work_dir, ant_base_fname)    

    if not os.path.exists(full_txt_fname):
        osutils.mkpath(work_dir)
        shutil.copy2(txt_file_name, full_txt_fname)
        shutil.copy2(offsets_fname, full_offsets_fname)
        shutil.copy2(ant_fname, full_ant_fname)

    osutils.mkpath(model_dir)
    osutils.mkpath(custom_model_dir)        

    target_term_model_fn = '{}/term_scutclassifier.v1.2.1.pkl'.format(model_dir)
    target_cust3_model_fn = '{}/cust_3.1_scutclassifier.v1.2.pkl'.format(custom_model_dir)
    
    if not os.path.exists(target_term_model_fn):
        shutil.copy2('resources/dir-scut-model/term_scutclassifier.v1.2.1.pkl',
                      target_term_model_fn)
    if not os.path.exists(target_cust3_model_fn):
        shutil.copy2('resources/dir-custom-model/cust_3.1_scutclassifier.v1.2.pkl',
                      target_cust3_model_fn)

    eb_runner = ebrunner.EbRunner(model_dir, work_dir, custom_model_dir)
    
    eb_antdoc = ebantdoc2.text_to_ebantdoc2(txt_file_name,
                                            work_dir,
                                            is_cache_enabled=is_cache_enabled,
                                            is_bespoke_mode=is_bespoke_mode,
                                            is_doc_structure=is_doc_structure,
                                            doc_lang=doc_lang)

    eb_runner.update_custom_models(set(['cust_3']), lang='en')
    
    for x, y in eb_runner.provision_annotator_map.items():
        print("provision: {}".format(x))
    for prov in eb_runner.custom_annotator_map.keys():
        print("cust_provision: {}".format(prov))

    prov_human_ant_list = eb_antdoc.prov_annotation_list
    for jjj in prov_human_ant_list:
        print("ant: {}".format(jjj))

    cust_3_annotator = eb_runner.custom_annotator_map.get('cust_3')
    print('threshold = ', cust_3_annotator.threshold)
    ant_list, threshold = cust_3_annotator.annotate_antdoc(eb_antdoc,
                                                           threshold=cust_3_annotator.threshold,
                                                           prov_human_ant_list=prov_human_ant_list)

    print(eb_antdoc)
    

