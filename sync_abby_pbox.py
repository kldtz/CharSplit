#!/usr/bin/env python3

import argparse
import logging
import sys
from typing import TextIO

from kirke.docstruct import pdftxtparser

from kirke.abbyxml import abbyxmlparser
from kirke.abbyxml.pdfoffsets import AbbyTableBlock, AbbyTextBlock, AbbyXmlDoc
from kirke.abbyxml.abbypbox_syncher import sync_doc_offsets, verify_abby_xml_doc_by_offsets


logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    fname = args.file

    xml_fname = fname.replace('.txt', '.pdf.xml')

    work_dir = 'dir-work'
    abby_xml_doc = abbyxmlparser.parse_document(xml_fname, work_dir=work_dir)
    # abby_xml_doc.print_text()

    txt_fname = fname
    pdf_txt_doc = pdftxtparser.parse_document(txt_fname, work_dir=work_dir)

    # pdf_txt_doc.print_debug_blocks()
    # pdf_txt_doc.save_debug_pages(work_dir=work_dir, extension='.sync.debug.tsv')
    txt_str_fname = fname.replace('.txt', '.txt.str')
    with open(txt_str_fname, 'wt') as fout:
        pdf_txt_doc.save_str_text(file=fout)
        print('wrote {}'.format(txt_str_fname))

    sync_doc_offsets(abby_xml_doc, pdf_txt_doc)

    txt_infer_fname = fname.replace('.txt', '.txt.infer')
    with open(txt_infer_fname, 'wt') as fout:
        abby_xml_doc.print_infer_text(file=fout)
        print('wrote {}'.format(txt_infer_fname))

    # has both infer_attr_dict and attr_dict
    txt_debug_fname = fname.replace('.txt', '.txt.debug')
    with open(txt_debug_fname, 'wt') as fout:
        abby_xml_doc.print_debug_text(file=fout)
        print('wrote {}'.format(txt_debug_fname))

    # abby_xml_doc.print_text()
    txt_meta_fname = fname.replace('.txt', '.txt.meta')
    with open(txt_meta_fname, 'wt') as fout:
        abby_xml_doc.print_text_with_meta(file=fout)
        print('wrote {}'.format(txt_meta_fname))

    txt_sync_fname = fname.replace('.txt', '.txt.sync')
    txt_unsync_fname = fname.replace('.txt', '.txt.unsync')
    with open(txt_sync_fname, 'wt') as sync_fout:
        with open(txt_unsync_fname, 'wt') as unsync_fout:
            verify_abby_xml_doc_by_offsets(abby_xml_doc,
                                           pdf_txt_doc.doc_text,
                                           sync_file=sync_fout,
                                           unsync_file=unsync_fout)
    print('wrote {}'.format(txt_sync_fname))
    print('wrote {}'.format(txt_unsync_fname))
