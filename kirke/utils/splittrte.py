#!/usr/bin/env python

import logging
from collections import defaultdict
import warnings
import os

from kirke.eblearn import ebtext2antdoc
from kirke.utils import osutils


# Currently, we don't have the information on whether we annotated a document
# for a particular provision or not.  Will modify this code if the situation
# changes.
# @deprecated
def provisions_split(provision_list, txt_fn_list, work_dir=None):
    warnings.warn("Shouldn't split based on positive labeled docs only.", DeprecationWarning)

    ebantdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
    # print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    provision_posneg_doc_list_map = defaultdict(lambda: defaultdict(list))
    for ebantdoc in ebantdoc_list:
        provision_set = ebantdoc.get_provision_set()
        for arg_prov_st in provision_list:
            if arg_prov_st in provision_set:
                provision_posneg_doc_list_map[arg_prov_st]['pos'].append(ebantdoc)
            else:
                provision_posneg_doc_list_map[arg_prov_st]['neg'].append(ebantdoc)

    # for "party", we have 287 positives and 13 negatives

    provision_antdocs_map = {}
    # assume we have 5 train-test sets for each provision, by mod
    for provision, posneg_doc_list_map in provision_posneg_doc_list_map.items():
        # We only care about pos instances for now because we don't know
        # for sure if the negative docs are annotated for this particular
        # provision.
        provision_antdocs_map[provision] = posneg_doc_list_map['pos']

    return provision_antdocs_map


def save_antdoc_fn_list(eb_antdoc_list, doclist_file_name):
    logging.debug("save_antdoc_fn_list(%s)", doclist_file_name)

    with open(doclist_file_name, 'wt') as fout:
        for eb_antdoc in eb_antdoc_list:
            txt_fn = eb_antdoc.get_file_id()
            # print(txt_fn, len(instance_list), sep='\t', file=fout)
            print(txt_fn, file=fout)

def split_provision_trte_old_pre_0410(provisions, txt_fn_list_fn, work_dir, model_dir_list):
    osutils.mkpath(work_dir)
    for moddir in model_dir_list:
        osutils.mkpath(moddir)

    eb_antdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fn_list_fn, work_dir=work_dir)

    for provision in provision_list:
        X = eb_antdoc_list
        y = [provision in ebantdoc.get_provision_set()
             for ebantdoc in eb_antdoc_list]

        num_pos, num_neg = 0, 0
        for yval in y:
            if yval:
                num_pos += 1
            else:
                num_neg += 1
        print("provision: {}, pos= {}, neg= {}".format(provision, num_pos, num_neg))
        # jshaw, hack, such as for sechead
        if num_neg < 2:
            y[0] = 0
            y[1] = 0

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        for moddir in model_dir_list:
            antdoc_fn_list = "{}/{}.doclist.txt".format(moddir, provision)
            splittrte.save_antdoc_fn_list(eb_antdoc_list, antdoc_fn_list)

            train_doclist_fn = "{}/{}_train_doclist.txt".format(moddir, provision)    
            splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)

            test_doclist_fn = "{}/{}_test_doclist.txt".format(moddir, provision)
            splittrte.save_antdoc_fn_list(X_test, test_doclist_fn)

# This approach make the testing across systems less reliable.
# Using file ID looks simpler.
"""
def split_provision_trte2(provfiles_dir, work_dir, model_dir_list):
    osutils.mkpath(work_dir)
    for moddir in model_dir_list:
        osutils.mkpath(moddir)

    txt_file_set = set([])
    prov_filelist_map = defaultdict(list)
    provision_list = []

    for file_name in os.listdir(provfiles_dir):
        if file_name.endswith('.doclist.txt'):
            prefix = file_name[:-12]
            print("prov = [{}]".format(prefix))
            provision_list.append(prefix)
            with open("{}/{}".format(provfiles_dir, file_name), 'rt') as fin:
                for line in fin:
                    line = line.strip()
                    txt_file_set.add(line)
                    prov_filelist_map[prefix].append(line)
        
    # fn_ebantdoc_map = ebtext2antdoc.fnlist_to_fn_ebantdoc_map(list(txt_file_set), work_dir=work_dir)
    fn_ebantdoc_map = ebtext2antdoc.fnlist_to_fn_ebantdoc_provset_map(list(txt_file_set), work_dir=work_dir)    

    for provision in provision_list:
        eb_antdoc_list = []
        for fn in prov_filelist_map[provision]:
            eb_antdoc_list.append(fn_ebantdoc_map[fn])
            
        X = eb_antdoc_list
        y = [provision in ebantdoc.get_provision_set()
             for ebantdoc in eb_antdoc_list]

        num_pos, num_neg = 0, 0
        for yval in y:
            if yval:
                num_pos += 1
            else:
                num_neg += 1
        print("provision: {}, pos= {}, neg= {}".format(provision, num_pos, num_neg))
        # jshaw, hack, such as for sechead
        if num_neg < 2:
            y[0] = 0
            y[1] = 0

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        for moddir in model_dir_list:
            antdoc_fn_list = "{}/{}.doclist.txt".format(moddir, provision)
            splittrte.save_antdoc_fn_list(eb_antdoc_list, antdoc_fn_list)

            train_doclist_fn = "{}/{}_train_doclist.txt".format(moddir, provision)    
            splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)

            test_doclist_fn = "{}/{}_test_doclist.txt".format(moddir, provision)
            splittrte.save_antdoc_fn_list(X_test, test_doclist_fn)
"""            

def split_provision_trte(provfiles_dir, work_dir, model_dir_list):
    osutils.mkpath(work_dir)
    for moddir in model_dir_list:
        osutils.mkpath(moddir)

    txt_file_set = set([])
    prov_filelist_map = defaultdict(list)
    provision_list = []

    for file_name in os.listdir(provfiles_dir):
        if file_name.endswith('.doclist.txt'):
            prefix = file_name[:-12]
            print("prov = [{}]".format(prefix))
            provision_list.append(prefix)
            with open("{}/{}".format(provfiles_dir, file_name), 'rt') as fin:
                for line in fin:
                    line = line.strip()
                    txt_file_set.add(line)
                    prov_filelist_map[prefix].append(line)
        
    # fn_ebantdoc_map = ebtext2antdoc.fnlist_to_fn_ebantdoc_map(list(txt_file_set), work_dir=work_dir)
    fn_ebantdoc_map = ebtext2antdoc.fnlist_to_fn_ebantdoc_provset_map(list(txt_file_set), work_dir=work_dir)    

    for provision in provision_list:
        eb_antdoc_list = []
        X_train = []
        X_test = []        
        for fn in prov_filelist_map[provision]:
            tmp_ebantdoc = fn_ebantdoc_map[fn]
            eb_antdoc_list.append(tmp_ebantdoc)
            if tmp_ebantdoc.is_test_set:
                X_test.append(fn_ebantdoc_map[fn])
            else:
                X_train.append(fn_ebantdoc_map[fn])
            # print("fnxxx = [{}], id= [{}]".format(fn, mat.group(1)))
        print("provision: {}, len(train)= {}, len(test)= {}".format(provision, len(X_train), len(X_test)))

        if len(X_train) < 3:  # skip provisions with insufficient data
            logging.info("skipping provision, '{}', in split_provision_trte3() because len(X_train) = {}  is < 3".format(provision, len(X_train)))
            continue
        
        for moddir in model_dir_list:
            antdoc_fn_list = "{}/{}.doclist.txt".format(moddir, provision)
            save_antdoc_fn_list(eb_antdoc_list, antdoc_fn_list)

            train_doclist_fn = "{}/{}_train_doclist.txt".format(moddir, provision)    
            save_antdoc_fn_list(X_train, train_doclist_fn)

            test_doclist_fn = "{}/{}_test_doclist.txt".format(moddir, provision)
            save_antdoc_fn_list(X_test, test_doclist_fn)
            


# @deprecated
def split_provisions_from_posdocs(provisions, txt_fn_list_fn, work_dir, model_dir):
    warnings.warn("Shouldn't split based on positive labeled docs only.", DeprecationWarning)

    osutils.mkpath(work_dir)
    osutils.mkpath(model_dir)
    provision_list = provisions.split(',')

    provision_filelist_map = splittrte.provisions_split(provision_list, txt_fn_list_fn, work_dir=work_dir)
    for provision in provision_list:
        eb_antdoc_list = provision_filelist_map[provision]
        antdoc_fn_list = "{}/{}.doclist.txt".format(model_dir, provision)
        splittrte.save_antdoc_fn_list(eb_antdoc_list, antdoc_fn_list)


            
