#!/usr/bin/env python3

import argparse
import logging
import os

from kirke.docstruct import pdftxtparser

from kirke.abbyyxml import abbyyxmlparser
from kirke.abbyyxml.abbyypbox_syncher import sync_doc_offsets, print_abbyy_pbox_sync
from kirke.abbyyxml.abbyypbox_syncher import print_abbyy_pbox_unsync


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
    abbyy_xml_doc = abbyyxmlparser.parse_document(xml_fname, work_dir=work_dir)
    # abbyy_xml_doc.print_text()

    txt_fname = fname
    pdf_txt_doc = pdftxtparser.parse_document(txt_fname, work_dir=work_dir)

    base_fname = os.path.basename(txt_fname)

    work_fname = '{}/{}'.format(work_dir, base_fname)

    # pdf_txt_doc.print_debug_blocks()
    # pdf_txt_doc.save_debug_pages(work_dir=work_dir, extension='.sync.debug.tsv')
    txt_str_fname = work_fname.replace('.txt', '.txt.str')
    with open(txt_str_fname, 'wt') as fout:
        pdf_txt_doc.save_str_text(file=fout)
        print('wrote {}'.format(txt_str_fname))

    sync_doc_offsets(abbyy_xml_doc, pdf_txt_doc)

    txt_infer_fname = work_fname.replace('.txt', '.txt.infer')
    with open(txt_infer_fname, 'wt') as fout:
        abbyy_xml_doc.print_infer_text(file=fout)
        print('wrote {}'.format(txt_infer_fname))

    # has both infer_attr_dict and attr_dict
    txt_debug_fname = work_fname.replace('.txt', '.txt.debug')
    with open(txt_debug_fname, 'wt') as fout:
        abbyy_xml_doc.print_debug_text(file=fout)
        print('wrote {}'.format(txt_debug_fname))

    # abbyy_xml_doc.print_text()
    txt_meta_fname = work_fname.replace('.txt', '.txt.meta')
    with open(txt_meta_fname, 'wt') as fout:
        abbyy_xml_doc.print_text_with_meta(file=fout)
        print('wrote {}'.format(txt_meta_fname))

    txt_sync_fname = work_fname.replace('.txt', '.txt.sync')
    with open(txt_sync_fname, 'wt') as sync_fout:
        print_abbyy_pbox_sync(abbyy_xml_doc,
                              pdf_txt_doc.doc_text,
                              file=sync_fout)
        print('wrote {}'.format(txt_sync_fname))


    txt_unsync_fname = work_fname.replace('.txt', '.txt.unsync')
    with open(txt_unsync_fname, 'wt') as unsync_fout:
        print_abbyy_pbox_unsync(abbyy_xml_doc,
                                file=unsync_fout)
        print('wrote {}'.format(txt_unsync_fname))
