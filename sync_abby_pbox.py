#!/usr/bin/env python3

import argparse
import logging
from collections import defaultdict, OrderedDict
import pprint
import sys
import warnings
import re
import xmltodict
import operator

import json

from typing import Any, DefaultDict, Dict, List

from kirke.abbyxml import abbyxmlparser
from kirke.docstruct import pdftxtparser
from kirke.docstruct.pdfoffsets import PDFTextDoc

from kirke.utils.alignedstr import AlignedStrMapper


def print_pbox_doc(pbox_doc: PDFTextDoc) -> None:
    lineinfo_list = []
    doc_text = pbox_doc.doc_text
    for page_num, page in enumerate(pbox_doc.page_list):
        print('pbox page #{} ========================='.format(page_num))
        for pblockinfo in page.pblockinfo_list:
            print('\n    pbox block ---------------------------')
            lineinfo_list.extend(pblockinfo.lineinfo_list)
            for lineinfo in pblockinfo.lineinfo_list:
                for strinfo in lineinfo.strinfo_list:
                    start = strinfo.start
                    end = strinfo.end
                    multiplier = 300.0 / 72
                    x = int(strinfo.xStart * multiplier)
                    y = int(strinfo.yStart * multiplier)
                    str_text = doc_text[start:end]
                    print("        strinfo se={}, x,y={}    [{}]".format((start, end), (x, y), doc_text[start:end]))

def find_strinfo_by_xy(x: int,
                       y: int,
                       pbox_xy_map):
    # print("x, y = {}, {}".format(x, y))
    # for attr, val in pbox_xy_map.items():
    #     print("pbox_xy_map[{}] = {}".format(attr, val))

    # in experiment, x - tmp_x ranges from 0 to 1, inclusive
    for tmp_x in range(x-2, x+3):
        # in experiment, y - tmp_y ranges from 0 to 12, inclusive,
        # heavy toward 12.
        # the y diff between lines is in the 60 range, so ok.
        for tmp_y in range(y-15, y+3):
            # print("try tmp_x, tmp_y = {}, {}".format(tmp_x, tmp_y))
            strinfo = pbox_xy_map.get((tmp_x, tmp_y))
            if strinfo:
                # print("jjjdiff abby-pbox:\t{}\t{}".format(x - tmp_x, y - tmp_y))
                return strinfo
    return None

def sync_page_offsets(abby_page, pbox_page, doc_text: str) -> None:

    strinfo_list = []
    pbox_xy_map = OrderedDict()
    for pblockinfo in pbox_page.pblockinfo_list:
        # print('\n    pbox block ---------------------------')
        for lineinfo in pblockinfo.lineinfo_list:
            for strinfo in lineinfo.strinfo_list:
                start = strinfo.start
                end = strinfo.end
                multiplier = 300.0 / 72
                x = int(strinfo.xStart * multiplier)
                y = int(strinfo.yStart * multiplier)
                str_text = doc_text[start:end]
                # print("        strinfo se={}, x,y={}    [{}]".format((start, end), (x, y), doc_text[start:end]))

                strinfo_list.append(((start, end), (x, y), doc_text[start:end]))
                pbox_xy_map[(x, y)] = ((start, end), doc_text[start:end])

    ab_line_list = []
    for ab_block in abby_page.ab_text_blocks:
        for ab_par in ab_block.ab_pars:
            ab_line_list.extend(ab_par.ab_lines)

    for ab_line in ab_line_list:
        pbox_strinfo = find_strinfo_by_xy(ab_line.infer_attr_dict['x'],
                                          ab_line.infer_attr_dict['y'],
                                          pbox_xy_map)
        if not pbox_strinfo:
            print("\n NOT FOUND ab_line: {}".format(ab_line))
            print()
            raise Exception("cannot find ab_line '%r' in pbox" % (ab_line, ))

        (start, end), pbox_text = pbox_strinfo
        try:
            ab_line.abby_pbox_offset_mapper = AlignedStrMapper(ab_line.text,
                                                               pbox_text,
                                                               start)
        except Exception as exc:
            print('caught [{}]'.format(str(exc)))
            print("BBBBBBBBBBBBBBBBBBBBBBBbbb")
            print("  abline [{}]".format(ab_line.text))
            print("    pbox [{}]".format(pbox_text))
            raise


def sync_doc_offsets(abby_doc, pbox_doc) -> None:
    """Update lines in abb_doc withe offsets in pbox_doc lines.
    """

    doc_text = pbox_doc.doc_text
    for page_num, ab_page in enumerate(abby_doc.ab_pages):
        pbox_page = pbox_doc.page_list[page_num]
        sync_page_offsets(ab_page, pbox_page, doc_text)


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
    abby_xml_doc = abbyxmlparser.parse_document(xml_fname, work_dir)
    # abby_xml_doc.print_text()

    txt_fname = fname
    pdf_txt_doc = pdftxtparser.parse_document(txt_fname, work_dir=work_dir)

    # pdf_txt_doc.print_debug_blocks()
    # pdf_txt_doc.save_debug_pages(work_dir=work_dir, extension='.sync.debug.tsv')
    # print_pbox_doc(pdf_txt_doc)

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



