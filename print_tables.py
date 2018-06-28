#!/usr/bin/env python3

import argparse
import os


from kirke.abbyxml import abbyxmlparser, abbypbox_syncher
from kirke.docstruct import pdftxtparser

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

    abby_xml_doc = abbyxmlparser.parse_document(xml_fname, work_dir)

    # abbydoc.print_raw_lines()

    # abbydoc.print_text_with_meta()

    # abbydoc.print_text()
    pdf_txt_doc = pdftxtparser.parse_document(txt_fname, work_dir=work_dir)

    abbypbox_syncher.sync_doc_offsets(abby_xml_doc, pdf_txt_doc)


    txt_sync_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.txt.sync'))
    txt_unsync_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.txt.unsync'))
    with open(txt_sync_fname, 'wt') as sync_fout:
        with open(txt_unsync_fname, 'wt') as unsync_fout:
            abbypbox_syncher.verify_abby_xml_doc_by_offsets(abby_xml_doc,
                                                            pdf_txt_doc.doc_text,
                                                            sync_file=sync_fout,
                                                            unsync_file=unsync_fout)
    print('wrote {}'.format(txt_sync_fname))
    print('wrote {}'.format(txt_unsync_fname))
    

    # abby_xml_doc.print_text()
    txt_meta_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.txt.meta2'))
    with open(txt_meta_fname, 'wt') as fout:
        abby_xml_doc.print_text_with_meta_with_sync(file=fout)
        print('wrote {}'.format(txt_meta_fname))


    table_html_out_fn = '{}/{}'.format(work_dir, base_fname.replace('.pdf.xml', '.html'))
    with open(table_html_out_fn, 'wt') as fout:
        html_st = abbyxmlparser.to_html_tables(abby_xml_doc)
        print(html_st, file=fout)
    print('wrote "{}"'.format(table_html_out_fn))


if __name__ == '__main__':
    main()
