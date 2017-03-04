#!/usr/bin/env python

import logging
from collections import defaultdict
import warnings

from kirke.eblearn import ebtext2antdoc


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
    logging.debug("save_antdoc_fn_list({})".format(doclist_file_name))

    with open(doclist_file_name, 'wt') as fout:
        for eb_antdoc in eb_antdoc_list:
            txt_fn = eb_antdoc.get_file_id()
            # print(txt_fn, len(instance_list), sep='\t', file=fout)
            print(txt_fn, file=fout)
