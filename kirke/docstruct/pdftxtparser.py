# pylint: disable=too-many-lines

import argparse
import array
# pylint: disable=unused-import
from array import ArrayType
from collections import defaultdict
import copy
import logging
import os
import re
import sys
# pylint: disable=unused-import
from typing import Any, Dict, DefaultDict, List, Optional, Set, Tuple

from kirke.docstruct import docstructutils, linepos
from kirke.docstruct import pageformat, pdfdocutils, pdfoffsets, pdfutils, secheadutils
from kirke.docstruct.pdfoffsets import LineInfo3, LineWithAttrs
from kirke.docstruct.pdfoffsets import PLineAttrs
from kirke.docstruct.pdfoffsets import PageInfo3, PBlockInfo, PDFTextDoc, StrInfo
from kirke.docstruct.pdfoffsets import print_page_blockinfos_map
from kirke.docstruct.secheadutils import SecHeadTuple
from kirke.utils import mathutils, strutils, txtreader
from kirke.utils.textoffset import TextCpointCunitMapper

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# for setting footer attribute when reading pdf.offsets.json files from PDFBox
MAX_FOOTER_YSTART = 10000

IS_DEBUG = False
IS_DEBUG_TOC = False
IS_DEBUG_OUTPUT_NLP_TEXT = False
IS_DEBUG_CONTINUE_PAGE = False

# TODO, 445
IS_DEBUG_MODE = False
IS_DEBUG_YDIFF = False

# this is to see what are any header or footer
IS_DEBUG_DETAIL_MODE = False

IS_PARA_SEG_DEBUG_MODE = False
IS_DEBUG_PAGE_CLASSIFIER = False


EMPTY_PLINE_ATTRS = PLineAttrs()


def linex_list_to_block_map(linex_list: List[LineWithAttrs]) \
    -> Tuple[List[int], DefaultDict[int, List[LineWithAttrs]]]:
    block_linex_list_map = defaultdict(list)  # type: DefaultDict[int, List[LineWithAttrs]]
    block_id_list = []  # type: List[int]
    block_ids = set([])  # type: Set[int]
    for linex in linex_list:
        block_num = linex.block_num
        block_linex_list_map[block_num].append(linex)
        if block_num not in block_ids:
            block_ids.add(block_num)
            block_id_list.append(block_num)

    return block_id_list, block_linex_list_map


# get better newline for some documents that has 3 /n between lines
# in a paragraph.
# Original PDF, "2.3.1.18.9 180718-AXA-TA-3.pdf"
def update_nl_using_pblock(nl_text: str, pageinfo_list) -> str:
    nl2_chars = list(nl_text)
    for page_info in pageinfo_list:
        # print("\n== pageinfo {}".format(page_info.page_num))
        for pblockinfo in page_info.pblockinfo_list:
            pb_text = pblockinfo.text
            pb_chars = list(pb_text)
            nl2_chars[pblockinfo.start:pblockinfo.end] = pb_chars
            # print("\n  block {}".format(pblockinfo.bid))
            # print("  [{}]".format(pb_text))
    nl2_text = ''.join(nl2_chars)
    return nl2_text


def mark_if_continued_from_prev_page(pdf_text_doc: PDFTextDoc) -> None:
    prev_page = pdf_text_doc.page_list[0]
    prev_block_id_list, prev_block_linex_list_map = \
        linex_list_to_block_map(prev_page.content_linex_list)
    if not prev_block_id_list:
        return
    prev_last_block_id = prev_block_id_list[-1]
    prev_last_para = prev_block_linex_list_map[prev_last_block_id]
    for apage in pdf_text_doc.page_list[1:]:
        apage_block_id_list, apage_block_linex_list_map = \
                linex_list_to_block_map(apage.content_linex_list)

        # pylint: disable=unused-variable
        prev_page_has_footer = False
        if prev_page.footer_linex_list:
            prev_page_has_footer = True

        if not apage_block_id_list:
            logger.info("%s, page #%d has no content block.",
                        pdf_text_doc.file_name, apage.page_num)
            continue

        apage_first_block_id = apage_block_id_list[0]
        apage_first_para = apage_block_linex_list_map[apage_first_block_id]
        # apage_last_block_id = apage_block_id_list[-1]
        # apage_last_para = apage_block_linex_list_map[apage_last_block_id]

        last_linex = prev_last_para[-1]
        cur_first_linex = apage_first_para[0]

        if IS_DEBUG_CONTINUE_PAGE:
            print("\npage {}, checking is_continue:".format(apage.page_num))
            print('last_line: {}'.format(last_linex.line_text))
            print('cur_first_line: {}'.format(cur_first_linex.line_text))

        if not last_linex.attrs.not_en and \
           (last_linex.line_text[-1].islower() or
            last_linex.line_text[-1] != '.'):

            if cur_first_linex.line_text[0].islower() and \
               not cur_first_linex.attrs.center and \
               not secheadutils.is_line_sechead_prefix(cur_first_linex.line_text):
                apage.is_continued_para_from_prev_page = True
                prev_page.is_continued_para_to_next_page = True
            elif re.search(r'\b(a|the|these|those|that)$', last_linex.line_text, re.I) and \
                 not secheadutils.is_line_sechead_prefix(cur_first_linex.line_text):
                apage.is_continued_para_from_prev_page = True
                prev_page.is_continued_para_to_next_page = True

        # it's possible that a paragraph is all caps and was split between pages
        # current not handle.  page 6 in 8290.txt
        # if prev_page_has_footer and \
        #    docutils.is_all_cap_words(last_linex.line_text) and
        #    docutils.is_all_cap_words(first_line)
        #    both not centered,  Watch out for title
        #    TOC, etc
        # Not implemented yet.

        # print("checking page %d [%s] with page %d [%s]" %
        #       (prev_page.page_num, last_linex.line_text[-20:],
        #        apage.page_num, cur_first_linex.line_text[:20]))
        # if apage.is_continued_para_from_prev_page:
        #     print("+++they are continued")
        # else:
        #     print("---they are NOT continued")

        prev_page = apage
        prev_block_id_list, prev_block_linex_list_map = \
            apage_block_id_list, apage_block_linex_list_map
        prev_last_block_id = prev_block_id_list[-1]
        prev_last_para = prev_block_linex_list_map[prev_last_block_id]

    if IS_DEBUG_CONTINUE_PAGE:
        for apage in pdf_text_doc.page_list:
            if apage.is_continued_para_from_prev_page:
                print("page {}, +++ they are continued".format(apage.page_num))
            else:
                print("page {}, --- NOT continued".format(apage.page_num))


def init_page_content_other_lines(apage: PageInfo3) -> None:
    header_linex_list, content_linex_list, footer_linex_list = [], [], []
    # toc_linex_list = []

    for linex in apage.line_list:
        if linex.attrs.header:
            header_linex_list.append(linex)
        elif linex.attrs.footer:
            footer_linex_list.append(linex)
        # elif linex.attrs.toc:
        #     toc_linex_list.append(linex)
        else:
            content_linex_list.append(linex)

    apage.header_linex_list = header_linex_list
    apage.footer_linex_list = footer_linex_list
    apage.content_linex_list = content_linex_list
    # apage.toc_linex_list = toc_linex_list

    if IS_DEBUG_DETAIL_MODE:
        print('=== page #{} header_len = {}, footer_len = {}, content_len = {}'.format(
            apage.page_num,
            len(apage.header_linex_list),
            len(apage.footer_linex_list),
            len(apage.content_linex_list)))


# pylint: disable=too-many-arguments
def output_linex_list_with_offset(header_linex_list: List[LineWithAttrs],
                                  offsets_line_list: List[Tuple[List[Tuple[linepos.LnPos,
                                                                           linepos.LnPos]],
                                                                PLineAttrs]],
                                  offset: int,
                                  sechead_context: Optional[Tuple[str, str, str, int]],
                                  pdf_text_doc: PDFTextDoc,
                                  # only for footer and header, we must separate each line
                                  is_output_line_break_per_line=False) \
                                  -> Tuple[int,
                                           Optional[Tuple[str, str, str, int]]]:
    """This updates offsets_line_list in-place with
       lines in header_linex_list.
    """

    # output this pages header
    if header_linex_list:
        for lxidx, linex in enumerate(header_linex_list):
            out_line = pdf_text_doc.doc_text[linex.lineinfo.start:linex.lineinfo.end]
            pline_attrs = linex.to_attrvals()  # type: PLineAttrs

            if linex.line_text and linex.attrs.sechead:
                sechead_context = linex.attrs.sechead
            elif sechead_context:
                pline_attrs.sechead = sechead_context

            span_se_list = [(linepos.LnPos(linex.lineinfo.start,
                                           linex.lineinfo.end),
                             linepos.LnPos(offset, offset + len(out_line)))]
            offsets_line_list.append((span_se_list, pline_attrs))
            offset += len(out_line) + 1  # to add eoln

            # if is_output_line_break_per_line:
            #     print('--------------------------------- {} {} [{}]'.format(linex.lineinfo.start,
            #                                                                 linex.lineinfo.end,
            #                                                                 linex.line_text))

            # this is to prevent a multi-line header that was split in the .txt.
            # In this case, put them together in the same paragraph will
            # include all the text in that page into this header
            if is_output_line_break_per_line and lxidx != len(header_linex_list) -1:
                span_se_list = [(linepos.LnPos(linex.lineinfo.end+2, linex.lineinfo.end+2),
                                 linepos.LnPos(offset, offset))]
                offsets_line_list.append((span_se_list, EMPTY_PLINE_ATTRS))
                offset += 1

        # because we already performed "if_header_linex_list:" check,
        # linex is guaranteed to be initialized with some value
        # pylint: disable=undefined-loop-variable
        span_se_list = [(linepos.LnPos(linex.lineinfo.end+2, linex.lineinfo.end+2),
                         linepos.LnPos(offset, offset))]
        offsets_line_list.append((span_se_list, EMPTY_PLINE_ATTRS))
        offset += 1

    return offset, sechead_context


def output_linebreak(from_offset: int,
                     offset: int,
                     offsets_line_list: List[Tuple[List[Tuple[linepos.LnPos,
                                                              linepos.LnPos]],
                                                   PLineAttrs]]) \
                                                   -> int:
    span_se_list = [(linepos.LnPos(from_offset, from_offset),
                     linepos.LnPos(offset, offset))]
    offsets_line_list.append((span_se_list, EMPTY_PLINE_ATTRS))
    offset += 1
    return offset


# Note: The code below should handle if a paragraph that is broken across more than 2 pages.
# to_use_page_footer_linx_list_queue stores anything that hasn't been outputed yet

# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def to_nlp_paras_with_attrs(pdf_text_doc: PDFTextDoc) \
                            -> List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]],
                                          PLineAttrs]]:
    """Convert a pdfbox's text into NLP text, with no line removal.

    Warning: The offsets after this transformation differs from original text.
    The outfile might NOT be the same length as the original file.

    Continuous paragraphs broken across pages in the original text will be joined here.

    The key variable here is apage.is_continue_para_from_prev_page.

    Returns: nlp_paras_with_attrs
    """

    offset = 0
    # pylint: disable=line-too-long
    offsets_line_list = []  # type: List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]], PLineAttrs]]

    # get better newline for some documents that has 3 /n between lines
    # in a paragraph.
    # Original PDF, "2.3.1.18.9 180718-AXA-TA-3.pdf"
    # better_text = update_nl_using_pblock(pdf_text_doc.doc_text, pdf_text_doc.page_list)

    # figure out if any pages has first paragraph continue from the previous page
    mark_if_continued_from_prev_page(pdf_text_doc)

    # now output each page to NLP text
    sechead_context = None  # type: Optional[Tuple[str, str, str, int]]

    # cannot mix footer and header together, cause whole page to included
    # if they are treated as a block.  So use a list of 2 lists
    to_use_page_footer_linex_list_queue = []  # type: List[List[LineWithAttrs]]
    prev_linex = None  # type: Optional[LineWithAttrs]
    linex = None  # type: Optional[LineWithAttrs]
    # pylint: disable=too-many-nested-blocks
    for apage in pdf_text_doc.page_list:
        # TODO, 09/20, xxx yyy
        # move this to         add_doc_structure_to_page(page, pdftxt_doc)
        # instead of here.  make this a part of the 'page'
        # header_linex_list, content_linex_list, footer_linex_list = \
        #     to_header_content_footer_linex_list(apage.line_list)
        block_id_list, block_linex_list_map = \
            linex_list_to_block_map(apage.content_linex_list)

        if IS_DEBUG_OUTPUT_NLP_TEXT:
            print('\n-----432 page {}'.format(apage.page_num))
            for hh_linex in apage.header_linex_list:
                print('page {}, header  linex = [{}]'.format(apage.page_num,
                                                             hh_linex.line_text))
                print('   attrs={}'.format(str(hh_linex.attrs)))
            for ff_linex in apage.footer_linex_list:
                print('page {}, footer  linex = [{}]'.format(apage.page_num,
                                                             ff_linex.line_text))
                print('   attrs={}'.format(str(ff_linex.attrs)))

            for block_id_x in block_id_list:
                bb_linex_list = block_linex_list_map[block_id_x]
                for pblock_count, bb_linex in enumerate(bb_linex_list):
                    print('page {}, block {}, linex = [{}]'.format(apage.page_num,
                                                                   pblock_count,
                                                                   bb_linex.line_text))
                    print('   attrs={}'.format(str(bb_linex.attrs)))

        if not block_id_list:
            # cannot be continued from previous page since there is NO text in this page.

            # output this pages header
            offset, sechead_context = output_linex_list_with_offset(apage.header_linex_list,
                                                                    offsets_line_list=offsets_line_list,
                                                                    offset=offset,
                                                                    sechead_context=sechead_context,
                                                                    pdf_text_doc=pdf_text_doc,
                                                                    is_output_line_break_per_line=True)

            offset, sechead_context = output_linex_list_with_offset(apage.footer_linex_list,
                                                                    offsets_line_list=offsets_line_list,
                                                                    offset=offset,
                                                                    sechead_context=sechead_context,
                                                                    pdf_text_doc=pdf_text_doc,
                                                                    is_output_line_break_per_line=True)
            continue

        # ok, so there is some text on this page
        last_block_id = block_id_list[-1]

        if apage.is_continued_para_from_prev_page:
            to_use_page_footer_linex_list_queue.append(apage.header_linex_list)
        else:
            # add a line break
            if prev_linex:
                offset = output_linebreak(from_offset=prev_linex.lineinfo.end+2,
                                          offset=offset,
                                          offsets_line_list=offsets_line_list)

            # output this pages header
            offset, sechead_context = output_linex_list_with_offset(apage.header_linex_list,
                                                                    offsets_line_list=offsets_line_list,
                                                                    offset=offset,
                                                                    sechead_context=sechead_context,
                                                                    pdf_text_doc=pdf_text_doc,
                                                                    is_output_line_break_per_line=True)


        # output the content blocks
        for seq, block_num in enumerate(block_id_list):
            block_linex_list = block_linex_list_map[block_num]

            if to_use_page_footer_linex_list_queue and seq == 1:
                # output previous page's footer and header, there can be multiple pages
                for footer_header_linex_list in to_use_page_footer_linex_list_queue:
                    offset, sechead_context = output_linex_list_with_offset(footer_header_linex_list,
                                                                            offsets_line_list=offsets_line_list,
                                                                            offset=offset,
                                                                            sechead_context=sechead_context,
                                                                            pdf_text_doc=pdf_text_doc,
                                                                            is_output_line_break_per_line=True)
                to_use_page_footer_linex_list_queue = []

            first_linex = block_linex_list[0]
            pline_attrs = first_linex.to_attrvals()

            # don't check for block.line_list length here
            # There are lines with sechead followed by sentences
            if first_linex.line_text and first_linex.attrs.sechead:
                sechead_context = first_linex.attrs.sechead
            elif sechead_context:
                pline_attrs.sechead = sechead_context

            span_se_list = []
            for linex in block_linex_list:
                out_line = pdf_text_doc.doc_text[linex.lineinfo.start:linex.lineinfo.end]
                span_se_list.append((linepos.LnPos(linex.lineinfo.start, linex.lineinfo.end),
                                     linepos.LnPos(offset, offset + len(out_line))))
                offset += len(out_line) + 1  # to add eoln
                prev_linex = linex

            offsets_line_list.append((span_se_list, pline_attrs))

            # merge the two broken paragraphs
            # we only do this if this is the first paragraph in the page
            # and continued from previous page
            # and this paragraph is NOT is_multi_line.
            # In the case of is_mutli_line, there are already too many linex added already
            # from this page.  Not sure if merging will benefit if this is a mutlie-line
            # paragraph.
            if seq == 0 and \
               apage.is_continued_para_from_prev_page:
                continued_span_se_list, unused_continued_para_attrs = offsets_line_list.pop()
                prev_span_se_list, prev_para_attrs = offsets_line_list.pop()
                combined_span_se_list = list(prev_span_se_list)
                combined_span_se_list.extend(continued_span_se_list)
                # the prev_para_attrs has precedence over continued_para_attrs
                offsets_line_list.append((combined_span_se_list, prev_para_attrs))

            if block_num != last_block_id:
                # add a line break
                offset = output_linebreak(from_offset=linex.lineinfo.end+2,
                                          offset=offset,
                                          offsets_line_list=offsets_line_list)

        if apage.is_continued_para_to_next_page:
            to_use_page_footer_linex_list_queue.append(apage.footer_linex_list)
        else:
            # linex cannot be null because there must be some text in this doc?
            # add a line break
            offset = output_linebreak(from_offset=linex.lineinfo.end+2,  # type: ignore
                                      offset=offset,
                                      offsets_line_list=offsets_line_list)

            # If there is only 1 block in the page, the header and footer queue is not yet outputed.
            # This will results in footer of this page appear before the previous header and footer.
            # Output it now.
            if to_use_page_footer_linex_list_queue and len(block_id_list) == 1:
                # output previous page's footer and header, there can be multiple pages
                for footer_header_linex_list in to_use_page_footer_linex_list_queue:
                    # each line in footer or header must be output separately, otherwise
                    # the whole text between header and footer will be output as a line (start from header,
                    # end from footer).  This will output a whole page, which is wrong.
                    offset, sechead_context = output_linex_list_with_offset(footer_header_linex_list,
                                                                            offsets_line_list=offsets_line_list,
                                                                            offset=offset,
                                                                            sechead_context=sechead_context,
                                                                            pdf_text_doc=pdf_text_doc,
                                                                            is_output_line_break_per_line=True)
                to_use_page_footer_linex_list_queue = []

            # output this pages header
            offset, sechead_context = output_linex_list_with_offset(apage.footer_linex_list,
                                                                    offsets_line_list=offsets_line_list,
                                                                    offset=offset,
                                                                    sechead_context=sechead_context,
                                                                    pdf_text_doc=pdf_text_doc,
                                                                    is_output_line_break_per_line=True)

    """
    # for BHI's doc with just one URL.  129073.txt
    if linex:
        # for the last block in the last page
        span_se_list = [(linepos.LnPos(linex.lineinfo.end+2, linex.lineinfo.end+2),
                         linepos.LnPos(offset, offset))]
        offsets_line_list.append((span_se_list, EMPTY_PLINE_ATTRS))
        offset += 1
    """

    # compute the not_empty_line_num for original text and nlp text
    start_from_lnpos_list = []  # type: List[Tuple[int, int, linepos.LnPos]]
    start_to_lnpos_list = []  # type: List[Tuple[int, int, linepos.LnPos]]
    for offsets_line in offsets_line_list:
        span_se_list, unused_attrs = offsets_line
        for from_lnpos, to_lnpos in span_se_list:
            # because of "gap lnpos", start can be the same
            start_from_lnpos_list.append((from_lnpos.start, from_lnpos.end, from_lnpos))
            start_to_lnpos_list.append((to_lnpos.start, to_lnpos.end, to_lnpos))

    not_empty_line_num = 0
    for unused_startx, _, from_lnpos in sorted(start_from_lnpos_list):
        # the 3rd branch is not used
        if from_lnpos.start == from_lnpos.end:
            from_lnpos.line_num = not_empty_line_num
            not_empty_line_num += 1
        elif from_lnpos.start != from_lnpos.end:
            from_lnpos.line_num = not_empty_line_num
            not_empty_line_num += 1
        else:
            from_lnpos.line_num = not_empty_line_num

    # do the same as above for start_to_lnpos_list
    not_empty_line_num = 0
    for unused_startx, _, to_lnpos in sorted(start_to_lnpos_list):
        # the 3rd branch is not used
        if to_lnpos.start == to_lnpos.end:
            to_lnpos.line_num = not_empty_line_num
            not_empty_line_num += 1
        elif to_lnpos.start != to_lnpos.end:
            to_lnpos.line_num = not_empty_line_num
            not_empty_line_num += 1
        else:
            to_lnpos.line_num = not_empty_line_num

    return offsets_line_list



# pylint: disable=too-many-arguments
def pbox_page_blocks_init(page_num: int,
                          block_num: int,
                          nl_text: str,
                          lxid_strinfos_map: Dict[int, List[StrInfo]],
                          page_lineoffset_list: List[Dict],
                          page_blockoffset_list: List[Dict],
                          # the variables below are
                          # modified inline
                          pgid_pblockinfos_map: Dict[int, List[PBlockInfo]],
                          para_not_linebreak_offsets: List[int]) \
                          -> int:
    blockid_linenums_map = defaultdict(list)  # type: Dict[int, List[int]]
    for lineoffset in page_lineoffset_list:
        line_num = lineoffset['lineNum']
        blockid_linenums_map[lineoffset['blockNum']].append(line_num)

    for blockoffset in page_blockoffset_list:
        pbox_block_num = blockoffset['id']
        linenum_list = blockid_linenums_map[pbox_block_num]
        if not linenum_list:
            # print('skipping empty paragraph, with just spaces, block_num {}'.
            #       format(pbox_block_num))
            continue

        block_num += 1
        tmp_strinfos = []  # type: List[StrInfo]
        for linenum in linenum_list:
            tmp_strinfos.extend(lxid_strinfos_map[linenum])

        # Do not get 'start' and 'end' from tmp_strinfos.
        # The strinfos in a block might NOT be continuous
        # and include too much text (whole page?).  Need to get
        # 'start' and 'end' from blockoffset instead.

        start = blockoffset['start']
        end = blockoffset['end']

        bxid_lineinfos = []  # type: List[LineInfo3]
        paraline_text = nl_text[start:end]
        # is_multi_lines = False
        # compute the para_not_linebreak_offsets
        for char_i, chx in enumerate(paraline_text, start):
            if chx == '\n':
                para_not_linebreak_offsets.append(char_i)
        paraline_text = paraline_text.replace('\n', ' ')

        # print('paraline_text: [{}]'.format(paraline_text))
        bxid_lineinfos.append(LineInfo3(start,
                                        end,
                                        linenum_list[0],  # take first one
                                        block_num,
                                        -1,  # prev_linebreak_ratio,
                                        tmp_strinfos))

        block_info = PBlockInfo(start,
                                end,
                                block_num,
                                page_num,
                                paraline_text,
                                bxid_lineinfos,
                                is_multi_lines=False)
        pgid_pblockinfos_map[page_num].append(block_info)

    return block_num


# pylint: disable=too-many-arguments
def page_paras_ydiff_init(page_linenum_list_map: Dict[int, List[int]],
                          lxid_strinfos_map: Dict[int, List[StrInfo]],
                          nl_text: str,
                          all_ydiffs: List[float],
                          lxid_pnum_map: Dict[int, int],
                          page_lineoffsets_map: Dict[int, List[Dict]],
                          page_blockoffsets_map: Dict[int, List[Dict]],
                          pgid_pblockinfos_map: Dict[int, List[PBlockInfo]],
                          para_not_linebreak_offsets: List[int],
                          page_numcol_map: Dict[int, int]) \
                          -> None:

    page_num_list = sorted(page_linenum_list_map.keys())

    # compute the page_ydiff_mode for all pages
    # Please note that if a page is a form-page, page_ydiff_mode is -1.
    page_ydiff_mode_map, pf_page_numcol_map = \
        pageformat.calc_page_ydiff_modes_num_cols(page_linenum_list_map,
                                                  lxid_strinfos_map,
                                                  nl_text,
                                                  all_ydiffs)
    # pass number of column in a page to outside
    page_numcol_map.update(pf_page_numcol_map)

    block_num = 0
    for page_num in page_num_list:
        page_linenum_list = page_linenum_list_map[page_num]

        unused_line_start, unused_line_end = -1, -1  # type Tuple[int, int]
        unused_lxline_strinfos = []  # type: List[StrInfo]

        page_ydiff_mode = page_ydiff_mode_map[page_num]
        # print('923 page_ydiff_mode: {}'.format(page_ydiff_mode))

        if page_ydiff_mode < 0:  # a form-page
            # use pdfbox's block info to create the paragraphs in
            # pgid_pblockinfos_map
            block_num = pbox_page_blocks_init(page_num,
                                              block_num,
                                              nl_text,
                                              lxid_strinfos_map=lxid_strinfos_map,
                                              page_lineoffset_list=page_lineoffsets_map[page_num],
                                              page_blockoffset_list=page_blockoffsets_map[page_num],
                                              pgid_pblockinfos_map=pgid_pblockinfos_map,
                                              para_not_linebreak_offsets=para_not_linebreak_offsets)
            # Eone setting up pgid_blockinfos_map, go to next page
            continue

        # not a form-page, use page_ydiff_mode to create paragraphs
        start = -1  # type: int
        # tmp_strinfos = []
        bxid_lineinfos = []  # type: List[LineInfo3]
        page_linenum_set = set(page_linenum_list)

        # print('page_lineum_list: {}'.format(len(page_linenum_list)))
        prev_y = 0
        for line_num in page_linenum_list:

            lx_strinfos = lxid_strinfos_map[line_num]
            tmp_start = lx_strinfos[0].start
            tmp_end = lx_strinfos[-1].end
            line_text = nl_text[tmp_start:tmp_end]
            line_len = len(line_text.strip())

            # checks the difference in y val between this line and the next,
            # if below the mode, join into a block, otherwise add block to block_info
            if line_num + 1 in page_linenum_set and \
               line_len > 0:
                y_diff_next_line = round(lxid_strinfos_map[line_num + 1][0].yStart - \
                                         lx_strinfos[0].yStart,
                                         2)
            else:
                y_diff_next_line = -1

            y_diff_prev_line = round(lx_strinfos[0].yStart - prev_y, 2)

            if IS_DEBUG_YDIFF:
                print('  page_ydiff_mode = {}, ydiff = {}'.format(page_ydiff_mode,
                                                                  y_diff_next_line))

            if tmp_start != tmp_end and \
               (y_diff_next_line < 0 or \
                y_diff_next_line > page_ydiff_mode + 1):
                if IS_DEBUG_YDIFF:
                    print("  kk88, branch 1")
                block_num += 1

                if start == -1:
                    start = tmp_start
                end = tmp_end
                page_num = lxid_pnum_map[line_num]

                # print('lineinfo3.init_2(({}, {}), line_num={}, block_num={}'.format(tmp_start,
                #                                                                    tmp_end,
                #                                                                    line_num,
                #                                                                    block_num))
                bxid_lineinfos.append(LineInfo3(tmp_start,
                                                tmp_end,
                                                line_num,
                                                block_num,
                                                y_diff_prev_line / page_ydiff_mode,
                                                lxid_strinfos_map[line_num]))

                if IS_DEBUG_YDIFF:
                    print("  para_line: [{}]".format(nl_text[start:end]))
                    print("  para_line2: [{}]".format(strutils.sub_newlines(nl_text[start:end])))
                # since all the lines in a paragraph should have maximum 1 newline between
                # the lines, we are replace multple \n's with just 1.  All others become spaces.
                para_line, is_multi_lines, not_linebreaks = \
                    pdfutils.para_to_para_list(strutils.sub_newlines(nl_text[start:end]))
                if IS_DEBUG_YDIFF:
                    print("  paraline3: [{}]".format(para_line))

                if not is_multi_lines:
                    for i in not_linebreaks:
                        para_not_linebreak_offsets.append(start + i)

                # print('block_num: {}'.format(block_num))
                for bxid_lineinfo in bxid_lineinfos:
                    # print('bxid_lineinfo.obid = {}'.format(bxid_lineinfo.obid))
                    bxid_lineinfo.obid = block_num
                    bxid_lineinfo.ybid = block_num

                block_info = PBlockInfo(start,
                                        end,
                                        block_num,
                                        page_num,
                                        para_line,
                                        bxid_lineinfos,
                                        is_multi_lines)
                pgid_pblockinfos_map[page_num].append(block_info)
                # block_info_list.append(block_info)
                # tmp_strinfos = []
                start = -1
                bxid_lineinfos = []
            else:
                if IS_DEBUG_YDIFF:
                    print("  kk88, branch 2")
                if start == -1:
                    start = tmp_start

                # print('lineinfo3.init_1(({}, {}), line_num={}, block_num={}'.format(tmp_start,
                #                                                                     tmp_end,
                #                                                                     line_num,
                #                                                                     block_num))
                bxid_lineinfos.append(LineInfo3(tmp_start,
                                                tmp_end,
                                                line_num,
                                                block_num,
                                                y_diff_prev_line / page_ydiff_mode,
                                                lxid_strinfos_map[line_num]))

            prev_y = lx_strinfos[0].yStart


def init_lxid_strinfos_map(str_offsets: Dict,
                           nl_text: str) \
    -> Tuple[Dict[int, List[StrInfo]],
             Dict[int, int],
             Dict[int, List[int]],
             List[float],
             Dict[int, List[float]]]:
    lxid_strinfos_map = defaultdict(list)  # type: DefaultDict[int, List[StrInfo]]
    min_ydiff, unused_max_ydiff = 5, 30
    prev_y = 0
    all_diffs = []  # type: List[float]
    lxid_pnum_map = {}  # type: Dict[int, int]
    page_linenum_list_map = defaultdict(list)  # type: Dict[int, List[int]]
    page_ydiff_list_map = defaultdict(list)  # type: Dict[int, List[float]]
    for str_offset in str_offsets:
        # 'start', 'end', 'lineNum', 'pageNum', 'xStart', 'yStart', 'xEnd'
        start = str_offset['start']
        end = str_offset['end']
        page_num = str_offset['pageNum']
        line_num = str_offset['lineNum']
        # pylint: disable=invalid-name
        xStart = str_offset['xStart']
        # pylint: disable=invalid-name
        xEnd = str_offset['xEnd']
        # pylint: disable=invalid-name
        yStart = str_offset['yStart']
        if str_offset.get('yEnd') is not None:
            yEnd = str_offset['yEnd']
            height = str_offset['height']
            font_size_in_pt = str_offset.get('fontSizeInPt', 6)
        else:
            # the smaller than the smallest we have found so far
            yEnd = yStart + 4.0
            height = 4.0
            font_size_in_pt = 6

        y_diff = yStart - prev_y

        # if y_diff > min_diff and y_diff < max_ydiff:
        if y_diff > min_ydiff:
            rounded_ydiff = mathutils.half_round(y_diff)
            all_diffs.append(rounded_ydiff)
            page_ydiff_list_map[page_num].append(rounded_ydiff)
        prev_y = yStart
        str_text = nl_text[start:end]
        if yStart < 100 and not str_text.strip():
            pass
        else:
            lxid_pnum_map[line_num] = page_num
            lxid_strinfos_map[line_num].append(StrInfo(start, end,
                                                       xStart, xEnd,
                                                       yStart, yEnd,
                                                       height, font_size_in_pt))
            page_linenum_list_map[page_num].append(line_num)

    return lxid_strinfos_map, lxid_pnum_map, page_linenum_list_map, all_diffs, page_ydiff_list_map


def parse_document(file_name: str,
                   work_dir: str) \
                   -> PDFTextDoc:
    base_fname = os.path.basename(file_name)
    doc_text = strutils.loads(file_name)

    cpoint_cunit_mapper = TextCpointCunitMapper(doc_text)
    unused_doc_len, str_offsets, line_breaks, pblock_offsets, page_offsets = \
        pdfutils.load_pdf_offsets(pdfutils.get_offsets_file_name(file_name), cpoint_cunit_mapper)

    nl_text, linebreak_offset_list = \
        pdfdocutils.text_offsets_to_nl(base_fname,
                                       doc_text,
                                       line_breaks,
                                       work_dir=work_dir)

    linebreak_arr = array.array('i', linebreak_offset_list)  # type: ArrayType

    lxid_strinfos_map, lxid_pnum_map, page_linenum_list_map, all_diffs, page_ydiff_list_map = \
        init_lxid_strinfos_map(str_offsets, nl_text)

    # keep tracke of pblock_offsets so that if form-page, we will use
    # those paragraphs instead of based on page-level y-diff
    page_blockoffsets_map = defaultdict(list)  # type: Dict[int, List[Dict]]
    for pblock_offset in pblock_offsets:
        page_num = pblock_offset['pageNum']
        page_blockoffsets_map[page_num].append(pblock_offset)
    page_lineoffsets_map = defaultdict(list)  # type: Dict[int, List[Dict]]
    for line_offset in line_breaks:
        page_num = line_offset['pageNum']
        page_lineoffsets_map[page_num].append(line_offset)

    if IS_PARA_SEG_DEBUG_MODE:
        doc_ydiff_count_map = defaultdict(int)  # type: Dict[float, int]
        for yyy in all_diffs:
            doc_ydiff_count_map[yyy] += 1
        print()
        print('At document level:')
        for val, key in sorted(((val, key) for key, val in doc_ydiff_count_map.items()),
                               reverse=True):
            print('   doc_ydiff_count_map[{}] = {}'.format(key, val))

        for tmp_pnum in sorted(page_ydiff_list_map.keys()):
            page_ydiff_list = page_ydiff_list_map[tmp_pnum]
            page_mode_ydiff = max(set(page_ydiff_list), key=page_ydiff_list.count)
            print('\npage[{}] ydiff_mode = {}'.format(tmp_pnum, page_mode_ydiff))

            page_y_diff_count = defaultdict(int)  # type: Dict[float, int]
            for yyy in page_ydiff_list:
                page_y_diff_count[yyy] += 1
            for val, key in sorted(((val, key) for key, val in page_y_diff_count.items()),
                                   reverse=True):
                print('   page y_diff_count[{}] = {}'.format(key, val))

    pgid_pblockinfos_map = defaultdict(list)  # type: DefaultDict[int, List[PBlockInfo]]
    # for nl_text, those that are not really line breaks, due to normal text
    para_not_linebreak_offsets = []  # type: List[int]
    # updating above two data structure in this call
    #    - pgid_pblockinfos_map
    #    - para_not_linebreak_offsets
    page_numcol_map = {}  # type: Dict[int, int]
    page_paras_ydiff_init(page_linenum_list_map=page_linenum_list_map,
                          lxid_strinfos_map=lxid_strinfos_map,
                          nl_text=nl_text,
                          all_ydiffs=all_diffs,
                          lxid_pnum_map=lxid_pnum_map,
                          page_lineoffsets_map=page_lineoffsets_map,
                          page_blockoffsets_map=page_blockoffsets_map,
                          pgid_pblockinfos_map=pgid_pblockinfos_map,
                          para_not_linebreak_offsets=para_not_linebreak_offsets,
                          page_numcol_map=page_numcol_map)

    # TODO, set it to empty for now
    para_not_linebreak_arr = array.array('i', para_not_linebreak_offsets)  # type: ArrayType

    # prepare paraline.txt
    paraline_fname = pdfdocutils.get_paraline_fname(base_fname, work_dir)
    ch_list = list(nl_text)
    for offset in para_not_linebreak_offsets:
        ch_list[offset] = ' '
    paraline_text = ''.join(ch_list)
    txtreader.dumps(paraline_text, paraline_fname)

    pageinfo_list = []  # type: List[PageInfo3]
    for page_offset in page_offsets:
        #id, start, end
        start = page_offset['start']
        end = page_offset['end']
        page_num = page_offset['id']
        pblockinfo_list = pgid_pblockinfos_map[page_num]
        pinfo = PageInfo3(doc_text, start, end, page_num, pblockinfo_list)
        pinfo.num_column = page_numcol_map.get(page_num, 1)  # the default is 1 column page
        pageinfo_list.append(pinfo)

    if IS_DEBUG_MODE:
        page_blockinfos_fname = os.path.join(work_dir,
                                             base_fname.replace('.txt',
                                                                '.pblockinfos.txt'))
        print_page_blockinfos_map(pgid_pblockinfos_map,
                                  nl_text,
                                  page_blockinfos_fname)

        # TODO, 445
        for x3_pageinfo in pageinfo_list:
            for linex in x3_pageinfo.line_list:
                print('442 linex.obid = {}'.format(linex.lineinfo.obid))

    pdf_text_doc = PDFTextDoc(file_name,
                              doc_text,
                              cpoint_cunit_mapper=cpoint_cunit_mapper,
                              pageinfo_list=pageinfo_list,
                              linebreak_arr=linebreak_arr,
                              para_not_linebreak_arr=para_not_linebreak_arr)

    if IS_DEBUG_MODE:
        pdf_text_doc.save_raw_pages(extension='.raw.pages.tsv')

    add_doc_structure_to_doc(pdf_text_doc)

    if IS_DEBUG_MODE:
        pdf_text_doc.save_raw_pages(extension='.raw.pages.docstruct.tsv')

    # nlp_paras_with_attrs is based on information from pdfbox.
    # Current pdfbox outputs lines with only spaces, so it sometime put the text
    # of a whole page as one block, with lines with only spaces as textual lines.
    # To preserve the original annotation performance, we still use this not-so-great
    # txt file as input to corenlp.
    # A better input file could be *.paraline.txt, which is used for lineannotator.
    # In *.paraline.txt, each line is a paragraph, based on some semi-English heuristics.
    # Section header for *.praline.txt is much better than trying to identify section for
    # pages with only 1 block.  Cannot really switch to *.paraline.txt now because double-lined text
    # might cause more trouble.

    nlp_paras_with_attrs = to_nlp_paras_with_attrs(pdf_text_doc)

    # for i, (gap_start, gap_end) in enumerate(gap2_span_list):
    #     print("gap {}: [{}]".format(i, doc_text[gap_start:gap_end]))
    if not nlp_paras_with_attrs:
        logger.info("Empty nlp_paras_with_attrs.  Not urgent.  File: %s", file_name)
        logger.info("  Likely cause: either no text or looked too much like table-of-content.")

    pdf_text_doc.nlp_paras_with_attrs = nlp_paras_with_attrs

    if IS_DEBUG_MODE:
        pdfdocutils.save_nlp_paras_with_attrs(pdf_text_doc,
                                              extension='.pdf.paras_with_attrs',
                                              work_dir=work_dir)

    # return pdf_text_doc, linebreak_arr, para_not_linebreak_arr, cpoint_cunit_mapper
    return pdf_text_doc



def merge_if_continue_to_next_page(prev_page, cur_page):
    if not prev_page.content_linex_list or not cur_page.content_linex_list:
        return
    last_line = prev_page.content_linex_list[-1]
    words = last_line.line_text.split()
    last_line_block_num = last_line.block_num
    last_line_align = last_line.align

    first_line = cur_page.content_linex_list[0]
    first_line_align = first_line.align

    # if the last line is not even toward the lower portion of the page, don't
    # bother merging
    if last_line.lineinfo.yStart < 600:
        return

    # if last line is not english sentence, no need to merge
    # Mainly for first page
    if not last_line.is_english:
        return

    # LF2 and LF3
    # or any type of sechead prefix
    # TODO, jshaw, implement a better prefix detection in secheadutil
    if last_line_align != first_line_align and last_line_align[:2] == first_line_align[:2] and \
       first_line.line_text[0] == '(':
        return

    # dont' join sechead or anything that's centered
    if first_line.is_centered or first_line.attrs.get('sechead'):
        return

    # 8 because average word per sentence is known to be around 7
    if len(words) >= 8 and (words[-1][-1].islower() or strutils.is_not_sent_punct(words[-1][-1])):
        if not first_line.attrs.get('sechead'):
            first_line_block_num = first_line.block_num
            for linex in cur_page.content_linex_list:
                if linex.block_num == first_line_block_num:
                    linex.block_num = last_line_block_num
                else:
                    break


def reset_all_is_english(pdftxt_doc):
    block_list_map = defaultdict(list)
    page_special_attrs = ['signature', 'address']
    for apage in pdftxt_doc.page_list:
        for linex in apage.content_linex_list:
            block_num = linex.block_num
            block_list_map[block_num].append(linex)
            # set up a page's special attrs for optimization later
            for special_attr in page_special_attrs:
                if linex.attrs.get(special_attr):
                    apage.attrs['has_{}'.format(special_attr)] = True

    # special_attrs = ['signature', 'address', 'table', 'chart']
    special_attrs = ['signature', 'address']
    for _, linex_list in block_list_map.items():
        if len(linex_list) > 1:
            linex_0 = linex_list[0]
            last_linex = linex_list[-1]
            if not last_linex.is_english and linex_0.is_english:
                last_linex.is_english = True

            # if signature or address block, distribute the tag
            special_attr_map = {}
            for special_attr in special_attrs:
                for linex in linex_list:
                    if linex.attrs.get(special_attr):
                        special_attr_map[special_attr] = True

            # distribute special attribute to all linex
            for special_attr in special_attrs:
                if special_attr_map.get(special_attr):
                    for linex in linex_list:
                        linex.attrs[special_attr] = True


# pylint: disable=invalid-name
def merge_adjacent_line_with_special_attr(apage):
    """Mark adjacent lines that 'signature' or 'address' attributes with same block_num
       in their linex attributes.
    """
    special_attrs = ['signature', 'address']
    for special_attr in special_attrs:
        if getattr(apage.attrs, 'has_{}'.format(special_attr)):
            prev_line = apage.content_linex_list[0]
            prev_block_num = prev_line.block_num
            prev_has_special_attr = getattr(prev_line.attrs, special_attr)
            for linex in apage.content_linex_list[1:]:
                has_special_attr = getattr(linex.attrs, special_attr)
                if has_special_attr and prev_has_special_attr:
                    linex.block_num = prev_block_num

                prev_block_num = linex.block_num
                prev_has_special_attr = has_special_attr


def update_page_removed_lines(pdftxt_doc: PDFTextDoc) -> None:
    """This function update pdftxt_doc.exclude_offsets with
       a list of (start, end).

    If consecutive lines are removed, their offsets are merged as
    one (start, end).
    """
    rm_list = []  # type: List[LineWithAttrs]
    for apage in pdftxt_doc.page_list:
        for linex in apage.line_list:
            if linex.attrs.header or linex.attrs.footer:
                pdftxt_doc.removed_lines.append(linex)
                rm_list.append(linex)
    # nothing to remove, nothing to update
    if not rm_list:
        return
    # jshaw, 2018-08-25
    # in the future, merge adjacent lines, based on
    # the text between the lines are all spaces or nl
    # Then, simply have an set of offsets that should
    # be removed in the final document.
    doc_text = pdftxt_doc.doc_text
    exclude_offsets = []  # type: List[Tuple[int, int]]
    prev_start, prev_end = rm_list[0].lineinfo.start, rm_list[0].lineinfo.end
    for linex in rm_list[1:]:
        start, end = linex.lineinfo.start, linex.lineinfo.end
        diff = start - prev_end
        if diff < 10:
            diff_text = doc_text[prev_end:start].strip()
            if diff_text:
                exclude_offsets.append((prev_start, prev_end))
                prev_start, prev_end = start, end
            else:
                prev_end = end
        else:
            exclude_offsets.append((prev_start, prev_end))
            prev_start, prev_end = start, end
    # the last one
    exclude_offsets.append((prev_start, prev_end))
    pdftxt_doc.exclude_offsets = exclude_offsets


def add_doc_structure_to_doc(pdftxt_doc: PDFTextDoc) -> None:
    # first remove obvious non-content lines, such
    # toc, page-num, header, footer
    # Also add section heads
    # page_attrs_list is to store table information?

    if IS_DEBUG_MODE:
        pdftxt_doc.save_debug_lines('.paged.before.dstruct.tsv')

    for page in pdftxt_doc.page_list:
        add_doc_structure_to_page(page, pdftxt_doc)

        # break blocks if they are in the middle of header, english sents
        # adjust_blocks_in_page(page, pdftxt_doc)
        init_page_content_other_lines(page)

        if page.page_num <= 3:
            if pdfdocutils.is_title_page(page):
                page.is_title_page = True
                pdfdocutils.adjust_title_page_blocks(page)
                # print("++ page #{} is a title page".format(page.page_num))
            # else:
            #     print("-- page #{} is NOT a title".format(page.page_num))

    # this only add footer and header's start, end to
    # apage.exclude_offsets
    update_page_removed_lines(pdftxt_doc)

    if IS_DEBUG_MODE:
        pdftxt_doc.save_debug_lines('.paged.dstruct.tsv')
        pdfdocutils.save_removed_lines(pdftxt_doc, '.rmlines.tsv')
        pdfdocutils.save_exclude_lines(pdftxt_doc, '.exclude.lines.tsv')

    # now we have basic block_group with correct
    # is_english set.  Useful for merging
    # blocks with only 1 lines as table, or signature section

    for apage in pdftxt_doc.page_list:
        # this only change lines inside a page and
        # set the same block_id if the previous line
        # is also 'address' or 'signature'
        merge_adjacent_line_with_special_attr(apage)

    if IS_DEBUG_MODE:
        pdftxt_doc.save_debug_lines('.paged.dstruct2.tsv')

    # for debugging page_format,
    # set pageformat.IS_TOP_LEVEL_DEBUG to True
    """
    if IS_DEBUG_PAGE_CLASSIFIER:
        # print("=== Page Format ===")
        # for page in pdftxt_doc.page_list:
        #     print('  page #{} page_format: {}'.format(page.page_num, page.page_format))
        pformat_fname = os.path.join(work_dir,
                                     os.path.basename(pdftxt_doc.file_name).replace('.txt',
                                                                                    '.pformat'))
        with open(pformat_fname, 'wt') as pf_out:
            # pnum_pformat_list = []  # type: List[Tuple[int, str]]
            for page in pdftxt_doc.page_list:
                print('{}\t{}'.format(page.page_num, page.page_format), file=pf_out)
        print('wrote {}'.format(pformat_fname), file=sys.stderr)
    """

    # now we have basic block_group with correct
    # is_english set.  Useful for merging
    # blocks with only 1 lines as table, or signature section

    # for apage in pdftxt_doc.page_list:
    #     merge_adjacent_line_with_special_attr(apage)

    """
    # Redo block info because they might be in different
    # pages.
    block_list_map = defaultdict(list)  # type: Dict[int, List[LineWithAttrs]]
    for apage in pdftxt_doc.page_list:
        for linex in apage.content_line_list:
            block_num = linex.block_num
            block_list_map[block_num].append(linex)

    # the block list is for the document, not a page
    paged_grouped_block_list = defaultdict(list)  # type: Dict[int, List[GroupedBlockInfo]]
    for block_num, line_list in sorted(block_list_map.items()):
        # take the page of the first line in a block as the page_num
        page_num = line_list[0].page_num
        paged_grouped_block_list[page_num].append(GroupedBlockInfo(page_num,
                                                                   block_num,
                                                                   line_list))
        " ""
        print('paged_grouped, block_num = {}'.format(block_num))
        for linexy in line_list:
            # print("  linexy: {}, {}".format(type(linexy), linexy))
            print("  linexy: [{}]".format(linexy.line_text))
        " ""

    # each page is a list of grouped_block
    pdftxt_doc.paged_grouped_block_list = []
    for page_num in range(1, pdftxt_doc.num_pages + 1):
        grouped_block_list = paged_grouped_block_list[page_num]
        pdftxt_doc.paged_grouped_block_list.append(grouped_block_list)
    """

# we do not use apage.content_line_list anymore
"""
def add_sections_to_page(apage, pdf_txt_doc):
    page_num = apage.page_num
    grouped_block_list = pdfoffsets.line_list_to_grouped_block_list(apage.content_line_list,
                                                                    page_num)

    # we don't collapse title pages and toc's
    if page_num > 9:
        grouped_block_list = collapse_similar_aligned_block_lines(grouped_block_list, page_num)

    apage.grouped_block_list = grouped_block_list

    # pylint: disable=invalid-name
    is_skip_table_and_chart_detection = False
    special_attrs = ['signature', 'address']
    for special_attr in special_attrs:
        if apage.attrs.get('has_{}'.format(special_attr)):
            # print("skip table_and_char_detection because of has_{}".format(special_attr))
            is_skip_table_and_chart_detection = True

    if is_skip_table_and_chart_detection:
        return

    # handle table and chart identification
    grouped_block_list = apage.grouped_block_list
    # if successful, markup_table_block_by_non_english will set 'has_table', a page attribute
    markup_table_block_by_non_english(grouped_block_list, apage)
    if not apage.attrs.get('has_table'):
        markup_table_block_by_columns(grouped_block_list, page_num)

    # create the annotation for table and chart from apage.content_line_list
    extract_tables_from_markups(apage, pdf_txt_doc)
"""


# pylint: disable=too-many-branches, too-many-statements, too-many-locals
def add_doc_structure_to_page(apage: PageInfo3,
                              pdf_txt_doc: PDFTextDoc):
    """Add the following information into the lines in a page.
         - toc
         - header
         - footer
         - line number, doesn't this overlap with footer?
         - signature
         - address
         - sechead

       It also create "content_lines", the lines that are NOT header or footer.

       The code here is overly complex.  Should simplify in the future.  The key
       part now is that the code add those doc-structure infos to lines.

       There shouldn't be modification to block structure here.
    """

    num_line_in_page = len(apage.line_list)
    page_num = apage.page_num
    # prev_line_text = ''
    # take out lines that are clearly not useful for annotation extractions:
    #   - toc
    #   - header
    #   - footer, include page number
    content_line_list = []
    toc_block_list = []
    # footer_index = -1  # because lines can be out of order, use yStart instead
    footer_yStart = MAX_FOOTER_YSTART
    # has_found_footer = False   # once found a footer, rest line in page are footer
    num_toc_line = 0
    has_toc_heading = False

    apage.is_multi_column = is_page_multi_column(apage)

    for line_num, line in enumerate(apage.line_list, 1):
        is_skip = False
        # a line might have the word 'contents' at top, but
        # not a toc line
        if not apage.is_multi_column and \
             docstructutils.is_line_header(line.line_text,
                                           line.lineinfo.yStart,
                                           line_num,
                                           line.is_english,
                                           line.is_centered,
                                           line.align,
                                           num_line_in_page,
                                           header_set=docstructutils.GLOBAL_PAGE_HEADER_SET):
            line.attrs.header = True
            pdf_txt_doc.special_blocks_map['header'].append(pdfoffsets \
                                                            .line_to_block_offsets(line,
                                                                                   'header',
                                                                                   page_num))
            is_skip = True
        elif docstructutils.is_line_toc_heading(line.line_text):
            if IS_DEBUG_TOC:
                print("==== tocheading== line is toc, %d [%s]" %
                      (line.page_num, line.line_text))
            line.attrs.toc = True
            has_toc_heading = True
            num_toc_line += 10  # I know this is not true yet
            is_skip = True
            toc_block_list.append(line)
        elif docstructutils.is_line_toc(line.line_text):
            if IS_DEBUG_TOC:
                print("==== linetoc== line is toc, %d [%s]" %
                      (line.page_num, line.line_text))
            line.attrs.toc = True
            num_toc_line += 1
            if num_toc_line >= 5:
                is_skip = True
            toc_block_list.append(line)
        elif docstructutils.is_line_page_num(line.line_text,
                                             line_num,
                                             num_line_in_page,
                                             line.lineinfo.prev_linebreak_ratio,
                                             line.lineinfo.yStart,
                                             line.is_centered):
            line.attrs.has_page_num = True
            # so we can detect footers after page_num, 1-based
            apage.attrs.page_num_index = line_num
            pdf_txt_doc.special_blocks_map['pagenum'].append(pdfoffsets \
                                                             .line_to_block_offsets(line,
                                                                                    'pagenum',
                                                                                    page_num))
            is_skip = True
        elif docstructutils.is_line_signature_prefix(line.line_text):
            line.attrs.signature = True
            # not skipped
        elif (docstructutils.is_line_address_prefix(line.line_text) or
              docstructutils.is_line_address(line.line_text,
                                             is_english=line.is_english,
                                             # pylint: disable=line-too-long
                                             is_sechead=secheadutils.is_line_sechead_prefix(line.line_text))):
                                             # is_sechead=line.attrs.sechead)):
            line.attrs.address = True
            # not skipped
        else:  # none-of-above
            # check if sechead
            # if secheadutils.is_line_sechead_prefix(line.line_text):
            # print("pdftxtparser 1, extract_sechead")
            sechead_tuple = secheadutils.extract_sechead(line.line_text,
                                                         is_centered=line.is_centered)
            if sechead_tuple:
                # print("  ggg check_sechead: [{}]".format(line.line_text))
                # print("      sechead_tuple: [{}]".format(sechead_tuple))

                line.attrs.sechead = sechead_tuple
                unused_sec_type, sechead_prefix, sechead_st, split_idx = sechead_tuple
                if split_idx != -1:
                    shead_end = line.lineinfo.start + split_idx
                else:
                    shead_end = line.lineinfo.end
                if not sechead_st or 'continue' in sechead_st.lower():
                    # 'exhibit c - continue'
                    sechead_st = sechead_prefix
                out_sechead = SecHeadTuple(line.lineinfo.start,
                                           shead_end,
                                           sechead_prefix,
                                           sechead_st,
                                           page_num)
                # print("sechead_tuple: {}".format(sechead_tuple))
                # print("             : {}".format(out_sechead))
                pdf_txt_doc.sechead_list.append(out_sechead)

        # 2nd stage of rules
        is_footer, unused_score = docstructutils.is_line_footer(line.line_text,
                                                                line_num,
                                                                num_line_in_page,
                                                                line.lineinfo.prev_linebreak_ratio,
                                                                # 1-based
                                                                apage.attrs.page_num_index,
                                                                line.is_english,
                                                                line.is_centered,
                                                                line.align,
                                                                line.lineinfo.yStart)
        if is_footer:
            line.attrs.footer = True
            is_skip = True
            # print("found footer in page {}: [{}]".format(page_num, line))
            pdf_txt_doc.special_blocks_map['footer'].append(pdfoffsets \
                                                            .line_to_block_offsets(line,
                                                                                   'footer',
                                                                                   page_num))
            # there can be multiple footer, keep the smallest one
            if line.lineinfo.yStart <= footer_yStart:
                footer_yStart = line.lineinfo.yStart

            # footer_index = line_num - 1
            continue  # found a footer, skip the rest in page
            # don't 'break', because the rest of the file might still be OK.
            # footer might appear first in the page instead of end, though
            # in PDF view, it is at the end.

        # prev_line_text = line.line_text
        if not is_skip and \
           (line.lineinfo.yStart < footer_yStart or
            len(line.line_text) >= 80):  # not delete normal para
            content_line_list.append(line)

    # if footer is found, set everything afterward as footer
    # if footer_index != -1:
    if footer_yStart != MAX_FOOTER_YSTART:
        for linex in apage.line_list:
            if linex.lineinfo.yStart >= footer_yStart and \
               len(linex.line_text) < 80:  # not delete normal para
                linex.attrs.footer = True

    # jshaw, NOTE
    # apage.content_linex_list = content_line_list

    # now decide if this is a toc page, based on
    # there are more than 4 toc lines
    if num_toc_line >= 5 or has_toc_heading:
        apage.attrs.has_toc = True
    else:
        # this is probably not a toc page
        # remove all toc from the lines
        for linex in apage.line_list:
            if linex.attrs.toc:
                linex.attrs.toc = False
        toc_block_list = []
        if apage.attrs.has_toc:
            apage.attrs.has_toc = False

    # if a whole page is all sechead, a toc
    num_sechead = 0
    first_sechead = None
    last_sechead = None
    for line_seq, linex in enumerate(content_line_list):
        if linex.attrs.sechead:
            last_sechead = line_seq
            if first_sechead is None:
                first_sechead = line_seq
            num_sechead += 1
    if len(content_line_list) > 5 and num_sechead / len(content_line_list) >= 0.8:
        for line_seq, linex in enumerate(content_line_list):
            if first_sechead and line_seq >= first_sechead and \
               last_sechead and line_seq <= last_sechead:
                if IS_DEBUG_TOC:
                    print("===323=7 sechead33== line is toc, %d [%s]" %
                          (linex.page_num, linex.line_text))
                linex.attrs.toc = True
        apage.attrs.has_toc = True

    # if there is no toc line, we are done here
    if not apage.attrs.has_toc:
        return

    # remove secheads after toc section
    # sechead info is not available until now.  Cannot remove
    # those earlier.
    tmp_toc_lines = []
    table_of_content_line_idx = -1  # if not found, start from beginning
    for line_seq, linex in enumerate(apage.line_list):  # this is line_list, not content_line_list
        if docstructutils.is_line_toc_heading(linex.line_text):
            table_of_content_line_idx = line_seq
        if linex.attrs.toc:
            tmp_toc_lines.append((line_seq, linex))

    if len(tmp_toc_lines) <= 3 and apage.page_num > 10 and table_of_content_line_idx == -1:
        # there is no toc, just ocr error
        for linex in apage.line_list:
            if linex.attrs.toc:
                linex.attrs.toc = False
        apage.attrs.has_toc = False
        return

    if not tmp_toc_lines:
        #print("52341235123423 apage.page_num = {}, table_of_content_line_idx = {}".format(
        #      apage.page_num, table_of_content_line_idx))
        return

    # we are here, so there must be toc lines
    first_toc_line, _ = tmp_toc_lines[0]
    last_toc_line, _ = tmp_toc_lines[-1]
    if len(tmp_toc_lines) >= 10:
        if table_of_content_line_idx != -1:
            for linex in apage.line_list[table_of_content_line_idx:last_toc_line+1]:
                if IS_DEBUG_TOC:
                    print("===323=3 beforelasttoc1== line is toc, %d [%s]" %
                          (linex.page_num, linex.line_text))
                linex.attrs.toc = True
        else:
            for linex in apage.line_list[first_toc_line:last_toc_line+1]:
                if IS_DEBUG_TOC:
                    print("===323=4 beforelasttoc1== line is toc, %d [%s]" %
                          (linex.page_num, linex.line_text))
                linex.attrs.toc = True

    deactivate_toc_detection = False
    non_sechead_count = 0
    for line in apage.line_list[last_toc_line + 1:]:
        # sechead detection is applied later
        # sechead, prefix, head, split_idx
        # print("pdftxtparser 2, extract_sechead")
        sechead_tuple = secheadutils.extract_sechead(line.line_text, is_centered=line.is_centered)
        is_sechead_prefix = secheadutils.is_line_sechead_prefix(line.line_text)
        if sechead_tuple or is_sechead_prefix:
            # print("  hhh check_sechead: [{}]".format(line.line_text))
            # print("      sechead_tuple: [{}]".format(sechead_tuple))
            # print("     is_sechead_pre: [{}]".format(is_sechead_prefix))
            if apage.attrs.has_toc and not deactivate_toc_detection:
                line.attrs.toc = True
            line.attrs.sechead = sechead_tuple
            if sechead_tuple:
                unused_sec_type, sechead_prefix, sechead_st, split_idx = sechead_tuple
                if split_idx != -1:
                    shead_end = line.lineinfo.start + split_idx
                else:
                    shead_end = line.lineinfo.end
                if not sechead_st or 'continue' in sechead_st.lower():
                    # 'exhibit c - continue'
                    sechead_st = sechead_prefix
            else:
                shead_end = line.lineinfo.end
                sechead_st = line.line_text
                sechead_prefix = ''
            out_sechead = SecHeadTuple(line.lineinfo.start,
                                       shead_end,
                                       sechead_prefix,
                                       sechead_st,
                                       # ' '.join([sechead_prefix, sechead_st]).strip(),
                                       page_num)
            # print("sechead_tuple2: {}".format(sechead_tuple))
            # print("              : {}".format(out_sechead))
            pdf_txt_doc.sechead_list.append(out_sechead)
        else:
            non_sechead_count += 1

        if non_sechead_count >= 3:
            deactivate_toc_detection = True

    # there can be toc lines that are not marked correct because they are more
    # english
    tmp_toc_lines = []
    # this is line_list, not content_line_list
    for line_seq, linex in enumerate(apage.line_list):
        if linex.attrs.toc:
            tmp_toc_lines.append((line_seq, linex))

    first_toc_line, _ = tmp_toc_lines[0]
    last_toc_line, _ = tmp_toc_lines[-1]
    # now mark all those in the middle as toc lines
    not_toc_lines_between = []
    outside_lines = []
    for line_seq, linex in enumerate(apage.line_list):
        if line_seq >= first_toc_line and line_seq <= last_toc_line:
            # collect all missed lines
            if not linex.attrs.toc:
                not_toc_lines_between.append(linex)
        else:
            outside_lines.append(linex)
    # if missed toc lines between tocs is too small, mark them as toc lines
    if len(not_toc_lines_between) / len(tmp_toc_lines) <= 0.4:
        for linex in apage.line_list:
            if linex.attrs.toc:
                pass
            # don't expect to see address or signature with toc
            elif linex.attrs.footer or linex.attrs.header or \
                 linex.attrs.has_page_num:
                pass
            else:
                if IS_DEBUG_TOC:
                    print("===323=6 too-small== line is toc, %d [%s]" %
                          (line.page_num, line.line_text))
                linex.attrs.toc = True


def is_page_multi_column(apage: PageInfo3) -> bool:
    linex_list = apage.line_list
    num_lines = len(linex_list)
    x_width_sum = 0

    num_split_col_line, num_one_col_line = 0, 0
    num_other_col_line = 0
    num_english_line = 0
    for linex in linex_list:
        x_width = linex.lineinfo.xEnd - linex.lineinfo.xStart
        x_width_sum += x_width
        # words = linex.line_text.split()
        # num_word = len(words)

        # TODO, WARNing
        # for lines from pbox_page_blocks_init(), the x_width is wrong.
        # it is only for the first line in that paragraph.
        # print('x_width: {}'.format(x_width))
        # print('line: [{}]'.format(linex.line_text))

        if linex.is_english:
            num_english_line += 1

        if x_width > 200 and x_width <= 300:
            num_split_col_line += 1
        elif x_width > 300:
            num_one_col_line += 1
        else:
            num_other_col_line += 1

        # print('line: [{}]'.format(linex.line_text))
        # print('  x_width = {}, num_word = {}, is_eng = {}'.format(x_width,
        #                                                           num_word,
        #                                                           linex.is_english))

    # print('33 pagenum= {}'.format(apage.page_num))
    # print('num_split_col_line = {}, num_one_col_line = {}, '
    #       'num_other_col_line = {}, num_english_line = {}'.format(num_split_col_line,
    #                                                               num_one_col_line,
    #                                                               num_other_col_line,
    #                                                               num_english_line))

    if num_lines == 0:
        return False
    if num_split_col_line == 0:
        return False

    if num_split_col_line > 50 and \
       num_one_col_line <= 10:
        return True

    if num_one_col_line / num_split_col_line < 0.05 and \
       num_english_line / num_lines > 0.6:
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description='Parse a document into a document structure.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file', help='a file to be annotated')

    args = parser.parse_args()
    txt_fname = args.file

    work_dir = 'dir-work'
    unused_pdf_txt_doc = parse_document(txt_fname,
                                        work_dir=work_dir)
    logger.info('Done.')


if __name__ == '__main__':
    main()
