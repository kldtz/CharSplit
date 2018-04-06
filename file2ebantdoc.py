#!/usr/bin/env python3

import argparse
import json
import logging
from pprint import pprint
import operator
from pprint import pprint
import sys


from kirke.utils.corenlputils import annotate

from kirke.utils import ebantdoc2
from kirke.eblearn import ebattrvec
from kirke.utils import memutils, osutils

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file')

    args = parser.parse_args()

    txt_file_name = args.file
    work_dir = 'dir-work'
    osutils.mkpath(work_dir)
    eb_antdoc = ebantdoc2.text_to_ebantdoc2(txt_file_name, work_dir)

    # ebantdoc2.print_attrvec_list(eb_antdoc)
    ebantdoc2.print_para_list(eb_antdoc)

    print(vars(eb_antdoc))


    print("memory_size(eb_antdoc) = %d bytes" % (memutils.get_size(eb_antdoc),))
    print("memory_size(eb_antdoc) = %.2f kilobytes" % (memutils.get_size_kbytes(eb_antdoc),))
    print("memory_size(eb_antdoc) = %.2f megabytes" % (memutils.get_size_mbytes(eb_antdoc),))

    ebantdoc_size = memutils.get_size(eb_antdoc)

    val_sz_acc = 0
    attr_size_list = []
    for attr, value in eb_antdoc.__dict__.items():
        val_sz = memutils.get_size(value)
        attr_size_list.append((attr, val_sz))

    for attr, val_sz in sorted(attr_size_list, key=operator.itemgetter(1), reverse=True):
        val_sz_acc += val_sz
        print("{}\tsize\t{}\t{:.5f}%\tacc\t{}\t{:.3f}%".format(attr,
                                                               val_sz,
                                                               val_sz * 100.0 / ebantdoc_size,
                                                               val_sz_acc,
                                                               val_sz_acc * 100.0 / ebantdoc_size))


    """
    print("checking change_control annotation:")
    for prov_ann in eb_antdoc.prov_annotation_list:
        # if prov_ann.label == 'change_control':
        #if prov_ann.label == 'change_control':
        print("\n\n{}\t{}".format(prov_ann, eb_antdoc.text[prov_ann.start:prov_ann.end]))
"""


    """
    print("get_size(eb_antdoc) = {} bytes".format(osutils.get_size(eb_antdoc)))

    #print("ebsents = %d bytes" %
    #      (osutils.get_size(eb_antdoc.ebsents), ))
    print("prov_annotations_list = %d bytes" %
          (osutils.get_size(eb_antdoc.prov_annotation_list), ))
    print("attrvec_list = %d bytes" %
          (osutils.get_size(eb_antdoc.attrvec_list), ))
    print("text = %d bytes" %
          (osutils.get_size(eb_antdoc.text), ))
    """

    """
    print("len(ebsents) = %d" % (len(eb_antdoc.ebsents),))
    total_byte = 0
    for i, ebsent in enumerate(eb_antdoc.ebsents):
        sz = osutils.get_size(ebsent)
        #print("ebsent #%d = %d bytes" % (i, sz))
        #print("    tokens = %d bytes" % (osutils.get_size(ebsent.tokens), ))
        #print("    text = %d bytes" % (osutils.get_size(ebsent.text), ))
        #print("    tokens_text = %d bytes" % (osutils.get_size(ebsent.tokens_text), ))
        # print("    entities = %d bytes" % (osutils.get_size(ebsent.entities), ))
        total_byte += sz
    print("avg %.2f bytes per ebsent" %
          (total_byte / len(eb_antdoc.ebsents)))
    """

if __name__ == '__main__':
    main()
