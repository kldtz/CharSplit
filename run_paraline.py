#!/usr/bin/env python3

import argparse
from kirke.docstruct import doc_pdf_reader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    fname = args.file

    offsets_fname = fname.replace('.txt', '.offsets.json')
    
    doc_pdf_reader.to_nl_paraline_texts(fname, offsets_fname,
                                        'dir-work')
    print("done.")
    
