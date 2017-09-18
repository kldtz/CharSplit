#!/usr/bin/env python

import argparse
import logging
from pprint import pprint
import sys
from pprint import pprint
import json
import os

from kirke.docstruct import pdftxtparser

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file')

    args = parser.parse_args()

    txt_fname = args.file
    work_dir = 'dir-work'

    base_fname = os.path.basename(txt_fname)

    pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=work_dir)

    pdftxtparser.save_debug_files(pdf_text_doc, base_fname, work_dir=work_dir)
