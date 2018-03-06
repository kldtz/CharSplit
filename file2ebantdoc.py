#!/usr/bin/env python3

import argparse
import logging
from pprint import pprint
import sys
from pprint import pprint
import json

from kirke.utils.corenlputils import annotate

from kirke.utils import ebantdoc2
from kirke.eblearn import ebattrvec
from kirke.utils import osutils

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file')

    args = parser.parse_args()

    txt_file_name = args.file
    work_dir = 'dir-work'
    eb_antdoc = ebantdoc2.text_to_ebantdoc2(txt_file_name, work_dir)

    # ebantdoc2.print_attrvec_list(eb_antdoc)
    ebantdoc2.print_para_list(eb_antdoc)

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
