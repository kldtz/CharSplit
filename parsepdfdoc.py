#!/usr/bin/env python

import argparse
import logging
from pprint import pprint
import warnings

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from kirke.docstruct import pdftxtparser
from kirke.utils import osutils, splittrte

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse a document into a document structure.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file', help='a file to be annotated')

    args = parser.parse_args()
    txt_fname = args.file

    # offsets_fname = doc_fn.replace(".txt", ".lineinfo.json")    
    # doc_pdf_reader.parse_document(doc_fn, offsets_fname, work_dir="/tmp")

    work_dir = 'dir-work'
    pdf_txt_doc = pdftxtparser.parse_document(txt_fname, work_dir=work_dir)
    pdftxtparser.to_paras_with_attrs(pdf_txt_doc, txt_fname, work_dir=work_dir)

    # pdf_txt_doc.print_debug_lines()
    pdf_txt_doc.print_debug_blocks()

    pdf_txt_doc.save_debug_pages(txt_fname, work_dir=work_dir)

    logging.info('Done.')
