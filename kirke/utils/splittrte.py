#!/usr/bin/env python3

import logging
from collections import defaultdict
import warnings
import os

from kirke.utils import osutils, ebantdoc5


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Currently, we don't have the information on whether we annotated a document
# for a particular provision or not.  Will modify this code if the situation
# changes.
# @deprecated
def provisions_split(provision_list, txt_fn_list, work_dir=None, is_doc_structure=False):
    warnings.warn("Shouldn't split based on positive labeled docs only.", DeprecationWarning)

    ebantdoc_list = ebantdoc5.doclist_to_ebantdoc_list(txt_fn_list,
                                                       work_dir=work_dir,
                                                       is_doc_structure=is_doc_structure)
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
    logger.debug("save_antdoc_fn_list(%s)", doclist_file_name)

    with open(doclist_file_name, 'wt') as fout:
        for eb_antdoc in eb_antdoc_list:
            txt_fn = eb_antdoc.get_file_id()
            # print(txt_fn, len(instance_list), sep='\t', file=fout)
            print(txt_fn, file=fout)


def has_provision_ant(ebantdoc_provset: ebantdoc5.EbAntdocProvSet,
                      provision: str) -> bool:
    for prov_annotation in ebantdoc_provset.prov_annotation_list:
        if prov_annotation.label == provision:
            return True
    return False


# pylint: disable=too-many-locals
def split_provision_trte(provfiles_dir, work_dir, model_dir_list, is_doc_structure=False):
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

    fn_ebantdoc_map = ebantdoc5.fnlist_to_fn_ebantdoc_provset_map(list(txt_file_set),
                                                                  work_dir=work_dir,
                                                                  is_doc_structure=is_doc_structure)

    for provision in provision_list:
        eb_antdoc_list = []
        # pylint: disable=invalid-name
        X_train = []
        X_test = []
        X_train_positive = []
        for fname in prov_filelist_map[provision]:
            tmp_ebantdoc_provset = fn_ebantdoc_map[fname]
            eb_antdoc_list.append(tmp_ebantdoc_provset)
            if tmp_ebantdoc_provset.is_test_set:
                X_test.append(fn_ebantdoc_map[fname])
            else:
                X_train.append(fn_ebantdoc_map[fname])

                if has_provision_ant(tmp_ebantdoc_provset, provision):
                    X_train_positive.append(fn_ebantdoc_map[fname])

            # print("fnxxx = [{}], id= [{}]".format(fname, mat.group(1)))
        print("provision: {}, len(train)= {}, len(test)= {}, len(train_pos)".format(provision,
                                                                                    len(X_train),
                                                                                    len(X_test),
                                                                                    len(X_train_positive)))

        if len(X_train) < 3:  # skip provisions with insufficient data
            logger.info("skipping '%s' in split_provision_trte(), len(X_train) = %d is < 3",
                         provision, len(X_train))
            continue

        for moddir in model_dir_list:
            antdoc_fn_list = "{}/{}.doclist.txt".format(moddir, provision)
            save_antdoc_fn_list(eb_antdoc_list, antdoc_fn_list)

            train_doclist_fn = "{}/{}_train_doclist.txt".format(moddir, provision)
            save_antdoc_fn_list(X_train, train_doclist_fn)

            train_pos_doclist_fn = "{}/{}_train_pos_doclist.txt".format(moddir, provision)
            save_antdoc_fn_list(X_train_positive, train_pos_doclist_fn)

            test_doclist_fn = "{}/{}_test_doclist.txt".format(moddir, provision)
            save_antdoc_fn_list(X_test, test_doclist_fn)



# @deprecated
def split_provisions_from_posdocs(provisions, txt_fn_list_fn, work_dir, model_dir):
    warnings.warn("Shouldn't split based on positive labeled docs only.", DeprecationWarning)

    osutils.mkpath(work_dir)
    osutils.mkpath(model_dir)
    provision_list = provisions.split(',')

    provision_filelist_map = provisions_split(provision_list,
                                              txt_fn_list_fn,
                                              work_dir=work_dir)
    for provision in provision_list:
        eb_antdoc_list = provision_filelist_map[provision]
        antdoc_fn_list = "{}/{}.doclist.txt".format(model_dir, provision)
        save_antdoc_fn_list(eb_antdoc_list, antdoc_fn_list)
