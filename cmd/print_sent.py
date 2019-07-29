#!/usr/bin/env python3

import argparse
import os

from kirke.utils import ebantdoc4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    fname = args.file

    # ebantdoc4.clear_cache(fname, work_dir='dir-work')
    eb_antdoc = ebantdoc4.text_to_ebantdoc(fname, work_dir='dir-work')

    nlp_text = eb_antdoc.get_nlp_text()
    # print('nlp_text:')
    # print(nlp_text)

    for sent_i, attrvec in enumerate(eb_antdoc.attrvec_list):
        print('sent #{}\t[{}]'.format(sent_i, nlp_text[attrvec.start:attrvec.end].replace('\n', ' || ')))
