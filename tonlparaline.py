#!/usr/bin/env python

import argparse
import logging
from pprint import pprint
import sys
import warnings

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from kirke.docstruct import doc_pdf_reader
from kirke.utils import osutils, splittrte

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse a document into a document structure.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file', help='a file to be annotated')

    args = parser.parse_args()
    doc_fn = args.file

    offsets_fname = doc_fn.replace(".txt", ".offsets.json")

    # work_dir = 'repo-work'
    work_dir = 'mytest'
    # doc_pdf_reader.parse_document(doc_fn, offsets_fname, work_dir=work_dir)

    orig_doc_text, nl_text, paraline_text, nl_fn, paraline_fn = doc_pdf_reader.to_nl_paraline_texts(doc_fn, offsets_fname, work_dir)

    logging.info('Done.')
