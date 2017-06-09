#!/usr/bin/env python

import argparse
import logging
from pprint import pprint
import sys
import warnings
import re

from pathlib import Path

from collections import defaultdict
import os

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from kirke.eblearn import ebrunner, ebtrainer, provclassifier, scutclassifier
from kirke.eblearn import ebtext2antdoc, ebannotator
from kirke.utils import osutils, splittrte, ebantdoc

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")

    args = parser.parse_args()

    provision = 'party'

    work_dir = 'dir-work'
    txt_file_set = set([])
    doclist_fn = 'dir-scut-model/party_train_doclist.txt'
    # doclist_fn = 'dir-scut-model/change_control_train_doclist.txt'
    # doclist_fn = 'dir-scut-model/' + provision + '_test_doclist.txt'
    with open(doclist_fn, 'rt') as fin:
        for line in fin:
            line = line.strip()
            txt_file_set.add(line)

    text_file_list = sorted(txt_file_set)
    fn_ebantdoc_map = ebtext2antdoc.fnlist_to_fn_ebantdoc_map(text_file_list,
                                                              work_dir=work_dir)

    

    for fn in text_file_list:
        tmp_ebantdoc = fn_ebantdoc_map[fn]
        doc_text = tmp_ebantdoc.text.replace('\n', ' ')
        # print('fn = {}, tmp_ebantdoc = {}'.format(fn, tmp_ebantdoc))
        pv_list = []
        i = 1
        for prov_ann in tmp_ebantdoc.prov_annotation_list:
            if prov_ann.label == provision:
                start = prov_ann.start
                end = prov_ann.end
                print('{}\t{}\t{}\t{}\t{}\t{}'.format(fn, i, provision, doc_text[start:end], start, end))
                i += 1
        
        
