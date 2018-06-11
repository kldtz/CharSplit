#!/usr/bin/env python3

import argparse
from kirke.utils import ebantdoc2


def print_ebantdoc(ebantdoc):
    doc_text = eb_antdoc.text
    ts_col = 'TRAIN'
    if eb_antdoc.is_test_set:
        ts_col = 'TEST'
    # print("doc_sents_fn = {}".format(doc_sents_fn))
    for i, attrvec in enumerate(eb_antdoc.attrvec_list, 1):
        # print("attrvec = {}".format(attrvec))
        tmp_start = attrvec.start
        tmp_end = attrvec.end
        sent_text = doc_text[tmp_start:tmp_end].replace(r'[\n\t]', ' ')
        # sent_text = attrvec.bag_of_words
        labels_st = ""
        if attrvec.labels:
            labels_st = ','.join(sorted(attrvec.labels))
        cols = [str(i), '({}, {})'.format(tmp_start, tmp_end),
                ts_col, labels_st, sent_text, str(attrvec)]        
        print('\t'.join(cols))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file')

    args = parser.parse_args()

    txt_file_name = args.file
    eb_antdoc, eb_antdoc_fn = ebantdoc2.load_cached_ebantdoc2(txt_file_name)

    ebantdoc2.print_ebantdoc(eb_antdoc)

