#!/usr/bin/env python3
# pylint: disable=too-many-lines

from collections import defaultdict
import logging
import re
import sys
# pylint: disable=unused-import
from typing import Dict, List, Match, Optional, TextIO, Tuple

from kirke.abbyyxml import abbyyutils, abbyyxmlparser
from kirke.abbyyxml.pdfoffsets import AbbyyLine, AbbyyPage, UnsyncedPBoxLine, UnsyncedStrWithY
from kirke.abbyyxml.pdfoffsets import AbbyyTableBlock, AbbyyTextBlock, AbbyyXmlDoc
from kirke.abbyyxml.pdfoffsets import print_abbyy_page_unsynced, print_abbyy_page_unsynced_aux
from kirke.docstruct.pdfoffsets import PDFTextDoc, PageInfo3
from kirke.utils import mathutils
from kirke.utils.alignedstr import AlignedStrMapper, MatchedStrMapper

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

IS_DEBUG_SYNC = False

IS_DEBUG_MODE = False
# IS_DEBUG_SYNC = True
IS_DEBUG_X2 = False

class AbbyyPBoxTable:

    def __init__(self,
                 page_num: int,
                 bot_left: Tuple[int, int],
                 top_right: Tuple[int, int],
                 span_list: List[Tuple[int, int]],
                 abbyy_text: str,
                 abbyy_table: AbbyyTableBlock):
        # a table can be made of mutliple tables from
        # adjacent pages
        self.bltr_list = [(page_num, bot_left, top_right)]
        self.span_list = list(span_list)
        self.abbyy_text = abbyy_text
        self.abbyy_table_list = [abbyy_table]

    def add_partition(self,
                      page_num: int,
                      bot_left: Tuple[int, int],
                      top_right: Tuple[int, int],
                      span_list: List[Tuple[int, int]],
                      abbyy_text: str,
                 abbyy_table: AbbyyTableBlock):                      
        self.bltr_list.append((page_num, bot_left, top_right))
        self.span_list.extend(span_list)
        self.abbyy_text = self.abbyy_text + '\n' + abbyy_text
        self.appyy_table_list.append(abbyy_table)


def extract_tables(abbyy_doc: AbbyyXmlDoc,
                   pbox_doc: PDFTextDoc) \
                   -> List[AbbyyPBoxTable]:
    """Extract tables from abbyy_doc and translate them into
    pbox's representation.

    The synching process does ONE page at a time.  Because of this 1
    page limit, there are opportunities to perform stray matching
    based on the fact that there are only limited unmatched candidates
    in a page.
    """

    result = []  # type: List[AbbyyPBoxTable]
    doc_text = pbox_doc.doc_text
    for page_num, ab_page in enumerate(abbyy_doc.ab_pages):
        pbox_page = pbox_doc.page_list[page_num]
        table_list = extract_page_tables(ab_page,
                                         pbox_page,
                                         doc_text)
        """
        if ab_page.num == 3:
            print('extrace_table, page 3:')
            for tablx in table_list:
                print("\n== table 235: {}".format(tablx.abbyy_text))
        """
            
        result.extend(table_list)

    return result


def extract_page_tables(abbyy_page: AbbyyPage,
                        pbox_page: PageInfo3,
                        doc_text: str) \
                        -> List[AbbyyPBoxTable]:

    # ab_line_list = abbyyxmlparser.get_page_abbyy_lines(abbyy_page)
    if IS_DEBUG_MODE:
        print("========= page  #{:3d} ========".format(abbyy_page.num))

    table_count = 0
    # table_seq, top, bottom, left, right
    table_rect_list = []  # type: List[Tuple[int, Tuple[int, int], Tuple[int, int]]]
    table_text_list = []  # type: List[str]
    table_abbyy_table_list = []  # type: List[AbbyTableBlock]
    for ab_block in abbyy_page.ab_blocks:

        if isinstance(ab_block, AbbyyTableBlock):
            ab_table_block = ab_block

            attr_dict = ab_block.attr_dict
            if IS_DEBUG_MODE:
                print("\nfound table ----------------------")
                print('attrs: {}'.format(ab_block.attr_dict))
                print(abbyyutils.block_to_text(ab_block))

            table_rect_list.append((table_count,
                                    (attr_dict['@l'], attr_dict['@b']), 
                                    (attr_dict['@r'], attr_dict['@t'])))

            table_text_list.append(abbyyutils.block_to_text(ab_block))
            table_abbyy_table_list.append(ab_block)

            if IS_DEBUG_MODE:
                print("table [{}] bot_left = {}, top_right = {}".format(table_count,
                                                                        (attr_dict['@l'], attr_dict['@b']),
                                                                        (attr_dict['@r'], attr_dict['@t'])))
            table_count += 1 

    table_strlist_map = defaultdict(list)
    multiplier = 300.0 / 72
    num_toc_line = 0
    for linex in pbox_page.line_list:

        is_in_table = False

        xStart = int(linex.lineinfo.xStart * multiplier)
        xEnd = int(linex.lineinfo.xEnd * multiplier)
        yStart = int(linex.lineinfo.yStart * multiplier)

        if linex.attrs.toc:
            num_toc_line += 1
        
        for table_count, table_bot_left, table_top_right in table_rect_list:
            if mathutils.is_rect_overlap(table_bot_left,
                                         table_top_right,
                                         (xStart, yStart+1),
                                         (xEnd, yStart)):
                table_strlist_map[table_count].append(linex)

                break
        """
        if is_in_table:
            print("in table: ", end='')
            print('  bot_left = {}, top_right= {}'.format((xStart, yStart+1),
                                                          (xEnd, yStart)))
            print('linex2: {}'.format(linex.tostr2()), end='')
            print(doc_text[linex.lineinfo.start:
                           linex.lineinfo.end])
        else:
            print("NOT in table: ", end='')
            print('  bot_left = {}, top_right= {}'.format((xStart, yStart+1),
                                                          (xEnd, yStart)))
            print(doc_text[linex.lineinfo.start:
                           linex.lineinfo.end])
        """

    if num_toc_line > 5:
        # this is a part of the is_invalid_table()
        # doing it here because this is where the pbox page information is
        # available
        if IS_DEBUG_X2 and table_strlist_map:
            print("\n^^^^x2 table is rejected in page {}, too many toc in page".format(abbyy_page.num))
            for table_seq, table_linex_list in table_strlist_map.items():

                # table_text = table_text_list[table_seq]
                # table_abbyy_table = table_abbyy_table_list[table_seq]
                print("\n^^^^ table {} in page {}".format(table_seq, abbyy_page.num))
                for linex in table_linex_list:                
                    print("  linex: [{}]".format(doc_text[linex.lineinfo.start:
                                                          linex.lineinfo.end]))


        abbyy_page.invalid_tables.extend(table_abbyy_table_list)
        # there are TOC lines, no table in such page
        return []

    out_table_list = []  # type: List[AbbyyPBoxTable]
    for table_seq, table_linex_list in table_strlist_map.items():

        table_rec = table_rect_list[table_seq]
        table_text = table_text_list[table_seq]
        table_abbyy_table = table_abbyy_table_list[table_seq]

        if IS_DEBUG_MODE:
            print("\n^^^^ table {} in page {}".format(table_seq, abbyy_page.num))
        se_list = []  # type: List[Tuple[int, int]]
        min_x_start, max_x_end = 10000, 0
        min_y_start, max_y_start = 10000, 0
        for linex in table_linex_list:
            if IS_DEBUG_MODE:            
                print("  linex: [{}]".format(doc_text[linex.lineinfo.start:
                                                      linex.lineinfo.end]))
            se_list.append((linex.lineinfo.start, linex.lineinfo.end))

            xStart = int(linex.lineinfo.xStart * multiplier)
            xEnd = int(linex.lineinfo.xEnd * multiplier)
            yStart = int(linex.lineinfo.yStart * multiplier)

            if xStart < min_x_start:
                min_x_start = xStart
            if xEnd > max_x_end:
                max_x_end = xEnd
            if yStart < min_y_start:
                min_y_start = yStart
            elif yStart > max_y_start:
                max_y_start = yStart
            
        span_list = se_list_to_span_list(se_list, doc_text)
        table_seq1, abbyy_bot_left, abbyy_top_right = table_rec
        if IS_DEBUG_MODE:
            print("span_list: {}".format(span_list))        
            print("orig bot_left={}, top_right={}".format(abbyy_bot_left,
                                                          abbyy_top_right))
            print("pbox bot_left={}, top_right={}".format((min_x_start, max_y_start),
                                                          (max_x_end, min_y_start)))

            for start, end in span_list:
                print("-- span -- {} {}".format(start, end))
                print(doc_text[start:end])

        if table_abbyy_table.is_invalid_kirke_table:
            if IS_DEBUG_X2:
                print("This is a is_invalid_kirke_table, skipped.")
            continue

        out_table_list.append(AbbyyPBoxTable(abbyy_page.num,
                                             abbyy_bot_left,
                                             abbyy_top_right,
                                             span_list,
                                             table_text,
                                             table_abbyy_table))
                                             
    
    return out_table_list


def se_list_to_span_list(se_list: List[Tuple[int, int]],
                                       doc_text: str) \
                         -> List[Tuple[int, int]]:
    if not se_list:
        return se_list
    se_list.sort()
    prev_start, prev_end = se_list[0]
    span_list = []  # type: List[Tuple[int, int]]
    for start, end in se_list[1:]:
        between_text = doc_text[prev_end:start]
        if between_text.isspace():
            prev_end = end
        else:
            span_list.append((prev_start, prev_end))
            prev_start, prev_end = start, end
    span_list.append((prev_start, prev_end))

    return span_list
