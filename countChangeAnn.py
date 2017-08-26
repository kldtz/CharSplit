#!/usr/bin/env python

import argparse
from collections import defaultdict
import logging
from pathlib import Path

from kirke.utils import ebantdoc2


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")

    args = parser.parse_args()

    provision = 'change_control'

    work_dir = 'dir-work'
    txt_file_set = set([])
    doclist_fn = 'dir-scut-model/change_control_test_doclist.txt'
    # doclist_fn = 'dir-scut-model/change_control_train_doclist.txt'
    # doclist_fn = 'dir-scut-model/' + provision + '_test_doclist.txt'
    with open(doclist_fn, 'rt') as fin:
        for line in fin:
            line = line.strip()
            txt_file_set.add(line)

    text_file_list = sorted(txt_file_set)
    fn_ebantdoc_map = ebantdoc2.fnlist_to_fn_ebantdoc_map(text_file_list,
                                                          work_dir=work_dir)

    

    be_antdoc_list = []
    cc_list = []
    fn_prov_list_map = defaultdict(list)
    fn_prov2_list_map = defaultdict(list)
    for fn in text_file_list:
        tmp_ebantdoc = fn_ebantdoc_map[fn]
        txt = tmp_ebantdoc.text.replace('\n', ' ')
        # print('fn = {}, tmp_ebantdoc = {}'.format(fn, tmp_ebantdoc))
        pv_list = []
        for prov_ann in tmp_ebantdoc.prov_annotation_list:
            # if prov_ann.label == 'change_control':
            if prov_ann.label == provision:
                cc_list.append((prov_ann, txt[prov_ann.start:prov_ann.end]))
                pv_list.append(prov_ann)
        fn_prov_list_map[fn] = pv_list

        xprov_annotation_list, is_test = ebantdoc2.load_prov_annotation_list(fn)

        pv_list2 = []
        for prov_ann in xprov_annotation_list:
            # if prov_ann.label == 'change_control':
            if prov_ann.label == provision:
                pv_list2.append(prov_ann)
        fn_prov2_list_map[fn] = pv_list2
        

    for fn, pv_list in fn_prov_list_map.items():
        len2 = len(fn_prov2_list_map[fn])
        print("{}\t{}\t{}".format(fn, len(pv_list), len2))
    #for i, (prov, txt) in enumerate(cc_list):
    #    print("prov_ann #{}: {}".format(i, len(prov)))
    #    print("    [[{}]]".format(txt))

        
