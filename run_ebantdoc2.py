#!/usr/bin/env python3

import argparse
import os

from kirke.utils import ebantdoc2
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    fname = args.file

    offset_fname = fname.replace('.txt', '.offsets.json')
    if os.path.exists(offset_fname):
        eb_antdoc = ebantdoc2.pdf_to_ebantdoc2(fname, offset_fname, work_dir='dir-work')
    else:
        eb_antdoc = ebantdoc2.html_to_ebantdoc2(fname, work_dir='dir-work')
    ebantdoc2.dump_ebantdoc_attrvec_with_secheads(eb_antdoc)
    
    """
    with open(fname, 'rt') as fin:
        for txt_fname in fin:
            txt_fname = txt_fname.strip()
            # eb_antdoc = ebantdoc2.html_no_docstruct_to_ebantdoc2(txt_fname, work_dir='dir-work')
            eb_antdoc = ebantdoc2.html_to_ebantdoc2(txt_fname, work_dir='dir-work')
            ebantdoc2.dump_ebantdoc_attrvec(eb_antdoc)
    """
