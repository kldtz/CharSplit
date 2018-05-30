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

from typing import Any, DefaultDict, Dict, List, Match, Optional, TextIO, Tuple

from kirke.abbyxml import abbyxmlparser
from kirke.abbyxml.pdfoffsets import AbbyLine, AbbyPar, AbbyPage
from kirke.abbyxml.pdfoffsets import AbbyTableBlock, AbbyTextBlock, AbbyXmlDoc, UnmatchedAbbyLine
from kirke.docstruct import pdftxtparser
from kirke.docstruct.pdfoffsets import PDFTextDoc

from kirke.utils import alignedstr
from kirke.utils.alignedstr import AlignedStrMapper



logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')

logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

IS_DEBUG_XY_DIFF = False


def find_strinfo_by_xy(x: int,
                       y: int,
                       pbox_xy_map: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]]) \
                       -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
    """
    Returns
        The first tuple, is original X, Y
        The 2nd tuple, the the start, end
        The 3rd string is the text
    """
    # print("x, y = {}, {}".format(x, y))
    # for attr, val in pbox_xy_map.items():
    #     print("pbox_xy_map[{}] = {}".format(attr, val))

    # in experiment, x - tmp_x ranges from 0 to 1, inclusive
    for tmp_x in range(x-2, x+3):
        # in experiment, y - tmp_y ranges from 0 to 12, inclusive,
        # heavy toward 12.
        # the y diff between lines is in the 60 range, so ok.
        for tmp_y in range(y-12, y+3):
            # print("try tmp_x, tmp_y = {}, {}".format(tmp_x, tmp_y))
            strinfo = pbox_xy_map.get((tmp_x, tmp_y))
            if strinfo:
                if IS_DEBUG_XY_DIFF:
                    print("jjjdiff1 abby-pbox:\t{}\t{}".format(x - tmp_x, y - tmp_y))
                    # to see the typical diffs between x and y
                    # grep jjjdiff kkk | cut -f 2 | sort | uniq -c
                    # grep jjjdiff kkk | cut -f 3 | sort | uniq -c
                return strinfo
        # in another doc, I have seen 17, so do y-13 to (y+3] first
        # then do y-14 to (y-21]
        for tmp_y in range(y-13, y-21, -1):
            # print("try tmp_x, tmp_y = {}, {}".format(tmp_x, tmp_y))
            strinfo = pbox_xy_map.get((tmp_x, tmp_y))
            if strinfo:
                if IS_DEBUG_XY_DIFF:
                    print("jjjdiff2 abby-pbox:\t{}\t{}".format(x - tmp_x, y - tmp_y))
                return strinfo
    return None

class StrMappedTracker:

    def __init__(self) -> None:
        self.strxy_used_map = {}  # Dict[Tuple[int, int], bool]

    def add(self, xy_pair: Tuple[int, int]) -> None:
        self.strxy_used_map[xy_pair] = False

    def get_unused_xy_list(self) -> List[Tuple[int, int]]:
        return [xy_pair for xy_pair, is_used
                in self.strxy_used_map.items() if not is_used]

    def set_used(self, xy_pair: Tuple[int, int]) -> None:
        self.strxy_used_map[xy_pair] = True


def find_unique_str_in_unmatched_ablines(stext: str,
                                         unmatched_ablines: List[UnmatchedAbbyLine]) \
                                         -> Optional[Tuple[Match[str],
                                                           UnmatchedAbbyLine]]:
    mats_um_abline_list = []  # type: List[Tuple[List[Match[str]], UnmatchedAbbyLine]]
    for um_abline in unmatched_ablines:
        mat_st = re.escape(stext)
        tmp_mat_list = list(re.finditer(mat_st, um_abline.ab_line.text))
        if tmp_mat_list:
            mats_um_abline_list.append((tmp_mat_list, um_abline))
    # Must only have only 1 ab_line has the str,
    # otherwise, too ambiguous and return None
    if len(mats_um_abline_list) == 1:
        # if ab_line.text matches, it can match once
        mats, um_abline = mats_um_abline_list[0]
        if len(mats) != 1:
            return None
        return mats[0], um_abline
    return None

def find_unique_abline_in_pbox_strs(um_abline: UnmatchedAbbyLine,
                                    xy_se_str_list: List[Tuple[Tuple[int, int], Tuple[int, int], str]]) \
                                    -> Optional[Tuple[Tuple[int, int],
                                                      Tuple[Tuple[int, int], Tuple[int, int]],
                                                      UnmatchedAbbyLine]]:
    xy_matse_tose_list = []  # type: List[Tuple[List[Match[str]], Tuple[int, int]]]
    mat_st = re.escape(um_abline.ab_line.text)
    for xypair, to_se, text in xy_se_str_list:
        tmp_mat_list = list(re.finditer(mat_st, text))
        if len(tmp_mat_list) == 1:
            mat = tmp_mat_list[0]
            xy_matse_tose_list.append((xypair, (mat.start(), mat.end()), to_se))

    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(xy_matse_tose_list) == 1:
        xypair, (mstart, mend), (to_start, to_end) = xy_matse_tose_list[0]
        adj_se = (to_start + mstart, to_start + mend)
        return xypair, (adj_se, adj_se), um_abline
    return None

def find_aligned_abline_in_pbox_strs(um_abline: UnmatchedAbbyLine,
                                     xy_se_str_list: List[Tuple[Tuple[int, int], Tuple[int, int], str]]) \
                                     -> Optional[Tuple[Tuple[int, int],
                                                       List[Tuple[int, int]],
                                                       List[Tuple[int, int]],
                                                       UnmatchedAbbyLine]]:
    xy_fromto_list = []  # type: List[Tuple[xypair, List[Tuple[int, int]], List[Tuple[int, int]]]]  
    abline_st = um_abline.ab_line.text
    for xypair, to_se, text in xy_se_str_list:
        try:
            abby_pbox_offset_mapper = AlignedStrMapper(abline_st,
                                                       text)
            xy_fromto_list.append((xypair,
                                   abby_pbox_offset_mapper.from_se_list,
                                   abby_pbox_offset_mapper.to_se_list,
                                   um_abline))
            print("aligned matched !! abline_st [{}]".format(abline_st))
            print("                   pdfbox_st [{}]".format(text))
            
        except Exception as exc:
            print("exc: {}".format(exc))
            # pass
            print("skipping mapping abline_st [{}]".format(abline_st))
            print("                   pbox_st [{}]".format(text))
            

    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(xy_fromto_list) == 1:
        return xy_fromto_list[0]

    return None


def sync_page_offsets(abby_page: AbbyPage,
                      pbox_page,
                      doc_text: str,
                      abbydoc_unmatched_ablines: List[UnmatchedAbbyLine]) -> None:
    # Note, strinfo_list maybe not used
    # strinfo_list = []  # type: List[Tuple[Tuple[int, int], Tuple[int, int], str]]
    pbox_xy_map = OrderedDict()  # type: Dict[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int], str]]
    str_mapped_tracker = StrMappedTracker()
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

                # strinfo_list.append(((start, end), (x, y), doc_text[start:end]))
                xy_pair = (x, y)
                pbox_xy_map[xy_pair] = (xy_pair, (start, end), doc_text[start:end])
                str_mapped_tracker.add(xy_pair)

    page_unmatched_ablines = []  # type: List[UnmatchedAbbyLine]
    ab_line_list = []  # type: List[AbbyLine]
    for ab_block in abby_page.ab_text_blocks:
        for ab_par in ab_block.ab_pars:
            ab_line_list.extend(ab_par.ab_lines)

    for ab_block in abby_page.ab_table_blocks:
        for ab_row in ab_block.ab_rows:
            for ab_cell in ab_row.ab_cells:
                for ab_par in ab_cell.ab_pars:
                    ab_line_list.extend(ab_par.ab_lines)            

    for ab_line in ab_line_list:
        pbox_strinfo = find_strinfo_by_xy(ab_line.infer_attr_dict['x'],
                                          ab_line.infer_attr_dict['y'],
                                          pbox_xy_map)
        if pbox_strinfo:
            # found X, Y that mached in pdfbox
            xypair, (start, end), pbox_text = pbox_strinfo
            try:
                ab_line.abby_pbox_offset_mapper = AlignedStrMapper(ab_line.text,
                                                                   pbox_text,
                                                                   start)
                str_mapped_tracker.set_used(xypair)
            except Exception as exc:
                logger.debug('sync_page_offsets warning: %s', str(exc))
                logger.debug("  abline [%s]", ab_line.text)
                logger.debug("    pbox [%s]", pbox_text)
                # raise
                page_unmatched_ablines.append(UnmatchedAbbyLine(ab_line, abby_page))
        else:
            logger.debug('sync_page_offsets() warning: NOT FOUND ab_line by x, y')
            logger.debug(ab_line)
            # raise Exception("cannot find ab_line '%r' in pbox" % (ab_line, ))
            page_unmatched_ablines.append(UnmatchedAbbyLine(ab_line, abby_page))

    unused_pbox_xy_list = str_mapped_tracker.get_unused_xy_list()
    check_unused_pbox_str_list = []  # type: List[Tuple[Tuple[int, int], Tuple[int, int], str]]
    if unused_pbox_xy_list:
        logger.debug("--prev unused pdfbox strs in page #%d", pbox_page.page_num)
        for xypair in unused_pbox_xy_list:
            tmp_t3 = pbox_xy_map[xypair]
            xypair2, se_pair, stext = tmp_t3
            logger.debug("--prev unused pdfbox str: xy=%r, %r [%s]", xypair, se_pair, stext)
            check_unused_pbox_str_list.append(tmp_t3)

    if page_unmatched_ablines:
        # to remove pbox's xy from pbox_xy_map
        to_remove_unused_pbox_xy_set = set([])  # type: Set[Tuple[int, int]]
        # pylint: disable=line-too-long
        um_abline_fromto_selist_map = defaultdict(list)  # type: Dict[UmatchedAbbyLine, List[Tuple[Tuple[int, int], Tuple[int,int]]]]
        # find abline in a part of pdfbox's str, such as
        # ab_line '16606208-9' in str 'I16606208-9', with prefix 'I'
        for um_abline in page_unmatched_ablines:
            atext = um_abline.ab_line.text

            xy_fromto_se_lists_um_abline = find_aligned_abline_in_pbox_strs(um_abline,
                                                                            check_unused_pbox_str_list)
            if xy_fromto_se_lists_um_abline:
                xypair, from_se_list, to_se_list, um_abline = xy_fromto_se_lists_um_abline
                to_remove_unused_pbox_xy_set.add(xypair)
                for fromto_se_pair in zip(from_se_list, to_se_list):
                    um_abline_fromto_selist_map[um_abline].append(fromto_se_pair)
            else:
                # now try to see if it is a sub-part of pdfbox's strs
                xy_fromto_se_um_abline = find_unique_abline_in_pbox_strs(um_abline,
                                                                         check_unused_pbox_str_list)
                
                if xy_fromto_se_um_abline:
                    xypair, fromto_se_pair, um_abline = xy_fromto_se_um_abline
                    to_remove_unused_pbox_xy_set.add(xypair)
                    um_abline_fromto_selist_map[um_abline].append(fromto_se_pair)

        # add fromto_selist to abline, so it is set up correctly
        # remove unmatched_abline
        for um_abline, fromto_selist in um_abline_fromto_selist_map.items():
            # now remove it
            page_unmatched_ablines.remove(um_abline)
            um_abline.ab_line.abby_pbox_offset_mapper = \
                alignedstr.make_aligned_str_mapper(sorted(fromto_selist))
        for xypair in to_remove_unused_pbox_xy_set:
            unused_pbox_xy_list.remove(xypair)


    # abbydoc.unmatched_ablines are not found in pbox_doc
    # Now try to use more expensive string matching to find
    # those str's in pdfbox in unmatched_ablines
    # try to find missing
    if unused_pbox_xy_list:
        um_abline_fromto_selist_map = defaultdict(list)
        to_remove_unused_pbox_xy_list = []  # type: List[Tuple[int, int]]
        for xypair in unused_pbox_xy_list:
            unused_xypair2, se_pair, stext = pbox_xy_map[xypair]

            mat_unmatched_abline = find_unique_str_in_unmatched_ablines(stext,
                                                                        page_unmatched_ablines)
            if mat_unmatched_abline:
                mat, unmatched_abline = mat_unmatched_abline
                um_abline_fromto_selist_map[unmatched_abline].append(((mat.start(), mat.end()),
                                                                      se_pair))
                to_remove_unused_pbox_xy_list.append(xypair)

        # add fromto_selist to abline, so it is set up correctly
        # remove unmatched_abline
        for um_abline, fromto_selist in um_abline_fromto_selist_map.items():
            # now remove it
            page_unmatched_ablines.remove(um_abline)
            um_abline.ab_line.abby_pbox_offset_mapper = \
                alignedstr.make_aligned_str_mapper(sorted(fromto_selist))
        for xypair in to_remove_unused_pbox_xy_list:
            unused_pbox_xy_list.remove(xypair)

    if unused_pbox_xy_list:
        logger.debug("unused pbox strs in page #%d", pbox_page.page_num)
        for xypair in unused_pbox_xy_list:
            unused_xypair2, se_pair, stext = pbox_xy_map[xypair]
            logger.debug("    unused str: xy=%r, %r [%s]", xypair, se_pair, stext)

    abbydoc_unmatched_ablines.extend(page_unmatched_ablines)

def sync_doc_offsets(abby_doc, pbox_doc) -> None:
    """Update lines in abb_doc withe offsets in pbox_doc lines.
    """

    doc_text = pbox_doc.doc_text
    for page_num, ab_page in enumerate(abby_doc.ab_pages):
        pbox_page = pbox_doc.page_list[page_num]
        sync_page_offsets(ab_page,
                          pbox_page,
                          doc_text,
                          abby_doc.unmatched_ab_lines)

def verify_abby_xml_doc_by_offsets(abby_doc: AbbyXmlDoc,
                                   doc_text: str,
                                   file: TextIO = sys.stdout) -> None:
    count_diff = 0
    for abby_page in abby_doc.ab_pages:
        if abby_page.num != 0:
            print('\n', file=file)
        print("========= page  #{:3d} ========".format(abby_page.num), file=file)

        for ab_block in abby_page.ab_blocks:

            if isinstance(ab_block, AbbyTextBlock):
                ab_text_block = ab_block
                for ab_par in ab_text_block.ab_pars:

                    for ab_line in ab_par.ab_lines:
                        st_list = []
                        amapper = ab_line.abby_pbox_offset_mapper
                        # print("\nab_line: [{}]".format(ab_line.text), file=file)

                        if amapper:
                            # print("from_se_list: {}".format(amapper.from_se_list))
                            # print("  to_se_list: {}".format(amapper.to_se_list))

                            for start, end in amapper.from_se_list:
                                to_start = amapper.get_to_offset(start)
                                to_end = amapper.get_to_offset(end)
                                st_list.append(doc_text[to_start:to_end])
                            to_st = ''.join(st_list)

                            if ab_line.text == to_st:
                                # print(  "   line: [{}]".format(to_st), file=file)
                                pass
                            else:
                                # slight differ due to space or '_', not important enough
                                # print("\nab_line: [{}]".format(ab_line.text), file=file)
                                # print(  "   line: [{}]".format(to_st), file=file)
                                # count_diff += 1
                                pass
                        else:
                            print("\nab_line: [{}]".format(ab_line.text), file=file)
                            print("---Not found in PDFBox---", file=file)
                            count_diff += 1
            elif isinstance(ab_block, AbbyTableBlock):
                ab_table_block = ab_block

                for ab_row in ab_table_block.ab_rows:
                    for ab_cell in ab_row.ab_cells:
                        for ab_par in ab_cell.ab_pars:

                            for ab_line in ab_par.ab_lines:
                                st_list = []
                                amapper = ab_line.abby_pbox_offset_mapper
                                # print("\nab_line: [{}]".format(ab_line.text), file=file)

                                if amapper:
                                    # print("from_se_list: {}".format(amapper.from_se_list), file=file)
                                    # print("  to_se_list: {}".format(amapper.to_se_list), file=file)

                                    for start, end in amapper.from_se_list:
                                        to_start = amapper.get_to_offset(start)
                                        to_end = amapper.get_to_offset(end)
                                        st_list.append(doc_text[to_start:to_end])
                                    to_st = ''.join(st_list)

                                    if ab_line.text == to_st:
                                        # print(  "   line: [{}]".format(to_st), file=file)
                                        pass
                                    else:
                                        # slight differ due to space or '_', not important enough                                        
                                        # print("\nab_line: [{}]".format(ab_line.text), file=file)
                                        # print(  "   line: [{}]".format(to_st), file=file)
                                        # count_diff += 1
                                        pass
                                else:
                                    print("\nab_line: [{}]".format(ab_line.text), file=file)
                                    print("---Not found in PDFBox---", file=file)
                                    count_diff += 1                

    print("\ncount_diff = {}".format(count_diff), file=file)




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

    txt_unsync_fname = fname.replace('.txt', '.txt.unsync')
    with open(txt_unsync_fname, 'wt') as fout:
        verify_abby_xml_doc_by_offsets(abby_xml_doc, pdf_txt_doc.doc_text, file=fout)
        print('wrote {}'.format(txt_unsync_fname))
