#!/usr/bin/env python3

import argparse
import logging
import operator
from typing import List

from kirke.utils import ebantdoc4
from kirke.utils import memutils, osutils

IS_DEBUG_MODE = False

# NOTE: Remove the following line to get rid of all logging messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


def print_list_size(list_obj: List, prefix_st: str) -> None:
    # list_obj_size = memutils.get_size(list_obj)
    for i, obj in enumerate(list_obj):
        obj_size = memutils.get_size(obj)

        print(prefix_st, end='')
        print(" #{} size={}".format(i, obj_size))
        print_obj_fields_size(obj, obj_size, prefix_st=' ' * (len(prefix_st) + 4))

def print_obj_fields_size(obj, obj_size: int, prefix_st: str) -> None:
    val_sz_acc = 0
    attr_size_list = []
    if isinstance(obj, tuple):
        for i, value in enumerate(obj):
            attr = 'tuple field #{}'.format(i)
            val_sz = memutils.get_size(value)
            attr_size_list.append((attr, val_sz))
    elif hasattr(obj, '__slots__'):
        for attr in obj.__slots__:
            value = getattr(obj, attr)
            val_sz = memutils.get_size(value)
            attr_size_list.append((attr, val_sz))
    else:
        for attr, value in obj.__dict__.items():
            val_sz = memutils.get_size(value)
            attr_size_list.append((attr, val_sz))

    for attr, val_sz in sorted(attr_size_list, key=operator.itemgetter(1), reverse=True):
        val_sz_acc += val_sz
        print(prefix_st, end='')
        print("{}\tsize\t{}\t{:.5f}%\tacc\t{}\t{:.3f}%".format(attr,
                                                               val_sz,
                                                               val_sz * 100.0 / obj_size,
                                                               val_sz_acc,
                                                               val_sz_acc * 100.0 / obj_size))


# pylint: disable=too-many-locals
def main():
    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file')

    args = parser.parse_args()

    txt_file_name = args.file
    work_dir = 'dir-work'
    osutils.mkpath(work_dir)
    eb_antdoc = ebantdoc4.text_to_ebantdoc4(txt_file_name, work_dir)

    # ebantdoc4.print_attrvec_list(eb_antdoc)
    # ebantdoc4.print_para_list(eb_antdoc)

    # print(vars(eb_antdoc))

    if IS_DEBUG_MODE:
        paras_with_attrs = eb_antdoc.paras_with_attrs
        for i, para_with_attrs in enumerate(paras_with_attrs):
            # print("  entities: {}".format(para_with_attrs.entities))
            # print("  labels: {}".format(para_with_attrs.labels))
            # print_list_size(para_with_attrs.entities, prefix_st='      entity')

            para_with_attrs_size = memutils.get_size(para_with_attrs)
            print("para_with_attrs #{}, size= {}".format(i, para_with_attrs_size))
            obj_size = para_with_attrs_size

            val_sz_acc = 0
            attr_size_list = []
            if len(para_with_attrs) != 2:
                print("len(para_with_attrs) = {}".format(len(para_with_attrs)))
            for j, value in enumerate(para_with_attrs):
                if j == 0:
                    attr = 'lnpos_pair_list'
                elif j == 1:
                    attr = 'attrs'
                val_sz = memutils.get_size(value)
                attr_size_list.append((attr, val_sz))

            prefix_st = '    '
            for attr, val_sz in sorted(attr_size_list, key=operator.itemgetter(1), reverse=True):
                val_sz_acc += val_sz
                print(prefix_st, end='')
                print("{}\tsize\t{}\t{:.5f}%\tacc\t{}\t{:.3f}%" \
                      .format(attr,
                              val_sz,
                              val_sz * 100.0 / obj_size,
                              val_sz_acc,
                              val_sz_acc * 100.0 / obj_size))

    print("\n\n")
    print("memory_size(eb_antdoc) = %d bytes" % (memutils.get_size(eb_antdoc),))
    print("memory_size(eb_antdoc) = %.2f kilobytes" % (memutils.get_size_kbytes(eb_antdoc),))
    print("memory_size(eb_antdoc) = %.2f megabytes" % (memutils.get_size_mbytes(eb_antdoc),))

    print("len(eb_antdoc.attrvec) = {}".format(len(eb_antdoc.attrvec_list)))
    print("len(eb_antdoc.origin_sx_lnpos_list) = {}" \
          .format(len(eb_antdoc.get_origin_sx_lnpos_list())))
    print("len(eb_antdoc.nlp_sx_lnpos_list) = {}".format(len(eb_antdoc.get_nlp_sx_lnpos_list())))

    eb_antdoc_size = memutils.get_size(eb_antdoc)
    print_obj_fields_size(eb_antdoc, eb_antdoc_size, prefix_st='')

    doc_text = eb_antdoc.get_text()
    nl_text = eb_antdoc.get_nl_text()
    paraline_text = eb_antdoc.get_paraline_text()

    if len(nl_text) != len(doc_text):
        print("len(nl_text) {} != len(doc_text) {}".format(len(nl_text),
                                                           len(doc_text)))
    if len(paraline_text) != len(doc_text):
        print("len(paraline_text) {} != len(doc_text) {}".format(len(nl_text),
                                                                 len(doc_text)))

    # this is just to make sure nlp_text is available
    nlp_text = eb_antdoc.get_nlp_text()
    if len(nlp_text) <= 0:
        print("len(nlp_text) = {}".format(len(nlp_text)))


if __name__ == '__main__':
    main()
