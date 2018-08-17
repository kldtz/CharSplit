#!/usr/bin/env python3

import argparse
import os

from kirke.docstruct import pdftxtparser
from kirke.abbyyxml import abbyyxmlparser, abbyypbox_syncher, tableutils
from kirke.abbyyxml.abbyypbox_syncher import (sync_doc_offsets,
                                            print_abbyy_pbox_sync,
                                            print_abbyy_pbox_unsync)


def main():
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    work_dir = 'dir-work'

    fname = args.file
    txt_fname = fname
    xml_fname = fname.replace(".txt", ".pdf.xml")

    base_fname = os.path.basename(txt_fname)

    abbyy_xml_doc = abbyyxmlparser.parse_document(xml_fname, work_dir)

    work_fname = '{}/{}'.format(work_dir, base_fname)

    # abbyydoc.print_raw_lines()

    # abbyydoc.print_text_with_meta()

    # abbyydoc.print_text()
    pdf_txt_doc = pdftxtparser.parse_document(txt_fname, work_dir=work_dir)
    doc_text = pdf_txt_doc.doc_text

    abbyypbox_syncher.sync_doc_offsets(abbyy_xml_doc, pdf_txt_doc)

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


    # abbyy_xml_doc.print_text()
    txt_meta_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.txt.meta2'))
    with open(txt_meta_fname, 'wt') as fout:
        abbyy_xml_doc.print_text_with_meta_with_sync(file=fout)
        print('wrote {}'.format(txt_meta_fname))


    table_html_out_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.abbyy.html'))
    with open(table_html_out_fn, 'wt') as fout:
        html_st = tableutils.to_html_tables(abbyy_xml_doc)
        print(html_st, file=fout)
    print('wrote {}'.format(table_html_out_fn))


    table_list = tableutils.get_abbyy_table_list(abbyy_xml_doc)
    for table_seq, table_block in enumerate(table_list):
        print("\ntable #{}".format(table_seq))
        start, end = tableutils.get_pbox_text_offset(table_block)
        print("table se = ({}, {})\n".format(start, end))
        # # print("  text: [{}]".format(doc_text[start:end]))

        se_list = tableutils.get_pbox_text_span_list(table_block, doc_text)

        for str_i, (start, end) in enumerate(se_list):
            print("  span #{}: ({}, {}) [{}]".format(str_i,
                                                     start, end,
                                                     doc_text[start:end]))


if __name__ == '__main__':
    main()
