# pylint: disable=too-many-lines

import argparse
import array
# pylint: disable=unused-import
from array import ArrayType
from collections import defaultdict
import logging
import os
import sys
# pylint: disable=unused-import
from typing import Any, Dict, DefaultDict, List, Optional, Set, Tuple

from kirke.docstruct import docstructutils, linepos
from kirke.docstruct import pdfdocutils, pdfoffsets, pdfutils, secheadutils
from kirke.docstruct.pdfoffsets import LineInfo3, LineWithAttrs
from kirke.docstruct.pdfoffsets import PLineAttrs
from kirke.docstruct.pdfoffsets import PageInfo3, PBlockInfo, PDFTextDoc, StrInfo
from kirke.utils import strutils, txtreader, mathutils
from kirke.utils.textoffset import TextCpointCunitMapper

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# for setting footer attribute when reading pdf.offsets.json files from PDFBox
MAX_FOOTER_YSTART = 10000

IS_DEBUG_MODE = False
IS_DEBUG_TOC = False

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


def mark_if_continued_from_prev_page(pdf_text_doc: PDFTextDoc) -> None:
    prev_page = pdf_text_doc.page_list[0]
    prev_block_id_list, prev_block_linex_list_map = \
        linex_list_to_block_map(prev_page.content_line_list)
    if not prev_block_id_list:
        return
    prev_last_block_id = prev_block_id_list[-1]
    prev_last_para = prev_block_linex_list_map[prev_last_block_id]
    for apage in pdf_text_doc.page_list[1:]:
        apage_block_id_list, apage_block_linex_list_map = \
                linex_list_to_block_map(apage.content_line_list)

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
        if not last_linex.attrs.not_en and \
           (last_linex.line_text[-1].islower() or
            last_linex.line_text[-1] != '.'):


            if cur_first_linex.line_text[0].islower() and \
               not secheadutils.is_line_sechead_prefix(cur_first_linex.line_text):
                apage.is_continued_para_from_prev_page = True
                prev_page.is_continued_para_to_next_page = True

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


def to_header_content_footer_linex_list(linex_list: List[LineWithAttrs]) \
    -> Tuple[List[LineWithAttrs],
             List[LineWithAttrs],
             List[LineWithAttrs]]:
    header_linex_list, content_linex_list, footer_linex_list = [], [], []
    for linex in linex_list:
        if linex.attrs.header:
            header_linex_list.append(linex)
        elif linex.attrs.footer:
            footer_linex_list.append(linex)
        else:
            content_linex_list.append(linex)

    return header_linex_list, content_linex_list, footer_linex_list


# pylint: disable=too-many-arguments, too-many-locals, too-many-statements
def init_pageinfo_list(doc_text: str,
                       nl_text: str,
                       line_breaks: List[Dict],
                       pblock_offsets: List[Dict],
                       paraline_fname: str,
                       page_offsets: List[Dict],
                       str_offsets: List[Dict]) \
                       -> List[PageInfo3]:
    """Returns the list of page."""
    # linebreak_arr = array.array('i', linebreak_offset_list)  # type: ArrayType

    lxid_strinfos_map = defaultdict(list)  # type: DefaultDict[int, List[StrInfo]]
    ## WARNING, some strx have no word/char in them, just spaces.
    ## It seems that some str with empty spaces might be intermixed with
    ## other strx, such as top of a page, blank_str, mixed with page_num
    ## toward the end of a page.  They are treated as the SAME line because
    ## no linebreak is issues in PDFBox.  As a result, removing all blank strx.
    ## Hopefully, the mix of different strx at vastly different yStart will disappear.
    ## Not sure how to fix it at the PDFBox side, in NoIndentPDFTextStripper.java, so
    ## fix it here.
    all_diffs = []  # type: List[int]
    prev_y = 0
    for str_offset in str_offsets:
        start = str_offset['start']
        end = str_offset['end']
        # page_num = str_offset['pageNum']
        line_num = str_offset['lineNum']
        # pylint: disable=invalid-name
        xStart = str_offset['xStart']
        # pylint: disable=invalid-name
        xEnd = str_offset['xEnd']
        # pylint: disable=invalid-name
        yStart = str_offset['yStart']
        y_diff = int(round(yStart - prev_y))

        if y_diff > 0:
            all_diffs.append(y_diff)
        prev_y = yStart

        # some times, empty strx might mix with page_num
        # don't add them
        str_text = nl_text[start:end]
        if yStart < 100 and not str_text.strip():
            pass
        else:
            lxid_strinfos_map[line_num].append(StrInfo(start, end,
                                                       xStart, xEnd, yStart))

    # for y_diff_count, yy in enumerate(all_diffs):
    #     print('y_diff_count= {}, ydiff = {}'.format(y_diff_count, yy))
    mode_diff = int(max(set(all_diffs), key=all_diffs.count))
    # print('mode_diff = {}'.format(mode_diff))
    # found_linenum_set = set(lxid_strinfos_map.keys())

    bxid_lineinfos_map = defaultdict(list)  # type: DefaultDict[int, List[LineInfo3]]
    tmp_prev_end = 0
    for break_offset in line_breaks:
        start = tmp_prev_end
        end = break_offset['offset']
        line_num = break_offset['lineNum']
        block_num = break_offset['blockNum']

        # adjust the start to exclude nl or space
        # print("start = {}, end= {}".format(start, end))
        while start < end and strutils.is_space_or_nl(nl_text[start]):
            # print("start2 = {}".format(start))
            start += 1
        while start <= end - 1 and strutils.is_nl(nl_text[end -1]):
            end -= 1

        if start != end:
            bxid_lineinfos_map[block_num].append(LineInfo3(start, end, line_num, block_num,
                                                           lxid_strinfos_map[line_num]))
        tmp_prev_end = end + 1

    pgid_pblockinfos_map = defaultdict(list)  # type: DefaultDict[int, List[PBlockInfo]]
    # for nl_text, those that are not really line breaks
    # para_not_linebreak_offsets = []  # type: List[int]
    blockinfo_list = []  # type: List[PBlockInfo]
    doc_block_id = 0
    for pblock_offset in pblock_offsets:
        pblock_id = pblock_offset['id']
        start = pblock_offset['start']
        end = pblock_offset['end']
        page_num = pblock_offset['pageNum']

        while start <= end - 1 and strutils.is_nl(nl_text[end -1]):
            end -= 1

        if start != end:
            lineinfo_list = bxid_lineinfos_map[pblock_id]
            # print('len(lineinfo_list): {}'.format(len(lineinfo_list)))
            if not lineinfo_list:
                # print('skipping this block since no lineinfo_list?')
                continue

            # block_linenum_set = set([lineinfo.line_num for lineinfo in lineinfo_list])
            cur_linechunk = [lineinfo_list[0]]  # type: List[LineInfo3]
            block_linechunk_list = [cur_linechunk]  # type: List[List[LineInfo3]]

            # prev_line = lineinfo_list[0]
            prev_linenum = lineinfo_list[0].line_num
            prev_ystart = lxid_strinfos_map[prev_linenum][0].yStart
            for lineinfo in lineinfo_list[1:]:
                linenum = lineinfo.line_num
                ystart = lxid_strinfos_map[linenum][0].yStart

                # checks the difference in y val between this line and the next,
                # if below the mode, join into a block, otherwise add block to block_info
                y_diff = int(ystart - prev_ystart)
                # print('block {}, y_diff = {}, mode_diff + 1 = {}'.format(pblock_id,
                #                                                          y_diff, mode_diff + 1))
                # print('prev_line: [{}]'.format(doc_text[prev_line.start:prev_line.end][:40]))
                # print('line_info: [{}]'.format(doc_text[lineinfo.start:lineinfo.end][:40]))

                if y_diff < 0 or y_diff > mode_diff + 1:
                    cur_linechunk = [lineinfo]
                    block_linechunk_list.append(cur_linechunk)
                else:
                    cur_linechunk.append(lineinfo)
                prev_linenum = line_num
                prev_ystart = ystart
                # prev_line = lineinfo

            for linechunk in block_linechunk_list:
                block_start = linechunk[0].start
                block_end = linechunk[-1].end
                paraline_chunk_text = nl_text[block_start:block_end]

                unused_para_line, xxis_multi_lines, unused_not_linebreaks = \
                        pdfutils.para_to_para_list(paraline_chunk_text)
                # print('xxis_multi_lines = {}'.format(xxis_multi_lines))

                # print("is_multi_lines = {}, paraline: [{}]\n".format(is_multi_lines,
                #                                                      para_line))
                # print("\nblock_chunk_text: [{}] is_multi={}".format(paraline_chunk_text,
                #                                                     xxis_multi_lines))
                if not xxis_multi_lines:
                    paraline_chunk_text = paraline_chunk_text.replace('\n', ' ')

                # print('page: {}, block {}'.format(page_num, doc_block_id))
                # print(paraline_chunk_text)
                # print()

                for linex in linechunk:
                    linex.bid = doc_block_id

                # print("is_multi_lines = {}, paraline: [{}]\n".format(is_multi_lines, para_line))
                block_info = PBlockInfo(block_start,
                                        block_end,
                                        doc_block_id,
                                        page_num,
                                        paraline_chunk_text,
                                        # bxid_lineinfos_map[pblock_id],
                                        linechunk,
                                        xxis_multi_lines)
                pgid_pblockinfos_map[page_num].append(block_info)
                blockinfo_list.append(block_info)
                doc_block_id += 1

    pdfdocutils.save_nltext_as_paraline_file(nl_text,
                                             blockinfo_list,
                                             paraline_fname)

    pageinfo_list = []  # type: List[PageInfo3]
    for page_offset in page_offsets:
        #id, start, end
        start = page_offset['start']
        end = page_offset['end']
        page_num = page_offset['id']
        pblockinfo_list = pgid_pblockinfos_map[page_num]
        # print('page xx: {}, len(pblockinfo_list) = {}'.format(page_num, len(pblockinfo_list)))
        pinfo = PageInfo3(doc_text, start, end, page_num, pblockinfo_list)
        pageinfo_list.append(pinfo)

    return pageinfo_list


# pylint: disable=too-many-arguments
def output_linex_list_with_offset(header_linex_list: List[LineWithAttrs],
                                  offsets_line_list: List[Tuple[List[Tuple[linepos.LnPos,
                                                                           linepos.LnPos]],
                                                                PLineAttrs]],
                                  offset: int,
                                  sechead_context: Optional[Tuple[str, str, str, int]],
                                  pdf_text_doc: PDFTextDoc) \
                                  -> Tuple[int,
                                           Optional[Tuple[str, str, str, int]]]:
    """This updates offsets_line_list in-place with
       lines in header_linex_list.
    """

    # output this pages header
    if header_linex_list:
        for linex in header_linex_list:
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

# pylint: disable=too-many-locals, too-many-statements
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

    # figure out if any pages has first paragraph continue from the previous page
    mark_if_continued_from_prev_page(pdf_text_doc)

    # now output each page to NLP text
    sechead_context = None  # type: Optional[Tuple[str, str, str, int]]

    to_use_page_footer_linex_list_queue = []  # type: List[List[LineWithAttrs]]
    prev_linex = None  # type: Optional[LineWithAttrs]
    linex = None  # type: Optional[LineWithAttrs]
    # pylint: disable=too-many-nested-blocks
    for apage in pdf_text_doc.page_list:

        header_linex_list, content_linex_list, footer_linex_list = \
            to_header_content_footer_linex_list(apage.line_list)

        block_id_list, block_linex_list_map = \
            linex_list_to_block_map(content_linex_list)

        if not block_id_list:
            # cannot be continued from previous page since there is NO text in this page.

            # output this pages header
            offset, sechead_context = output_linex_list_with_offset(header_linex_list,
                                                                    offsets_line_list=offsets_line_list,
                                                                    offset=offset,
                                                                    sechead_context=sechead_context,
                                                                    pdf_text_doc=pdf_text_doc)

            offset, sechead_context = output_linex_list_with_offset(footer_linex_list,
                                                                    offsets_line_list=offsets_line_list,
                                                                    offset=offset,
                                                                    sechead_context=sechead_context,
                                                                    pdf_text_doc=pdf_text_doc)
            continue

        # ok, so there is some text on this page
        last_block_id = block_id_list[-1]

        if apage.is_continued_para_from_prev_page:
            to_use_page_footer_linex_list_queue.append(header_linex_list)
        else:
            # add a line break
            if prev_linex:
                offset = output_linebreak(from_offset=prev_linex.lineinfo.end+2,
                                          offset=offset,
                                          offsets_line_list=offsets_line_list)

            # output this pages header
            offset, sechead_context = output_linex_list_with_offset(header_linex_list,
                                                                    offsets_line_list=offsets_line_list,
                                                                    offset=offset,
                                                                    sechead_context=sechead_context,
                                                                    pdf_text_doc=pdf_text_doc)

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
                                                                            pdf_text_doc=pdf_text_doc)
                to_use_page_footer_linex_list_queue = []

            is_multi_line = pdfdocutils.is_block_multi_line(block_linex_list)

            if is_multi_line:
                # TODO, jshaw, this doesn't handle the page_num gap line correct yet.
                # It should similar to the code for not is_multi-line
                for linex in block_linex_list:
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
                    prev_linex = linex
            else:
                block_line_st_list = []  # type: List[str]
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
                    block_line_st_list.append(out_line)
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
               not is_multi_line and \
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
            to_use_page_footer_linex_list_queue.append(footer_linex_list)
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
                    offset, sechead_context = output_linex_list_with_offset(footer_header_linex_list,
                                                                            offsets_line_list=offsets_line_list,
                                                                            offset=offset,
                                                                            sechead_context=sechead_context,
                                                                            pdf_text_doc=pdf_text_doc)
                to_use_page_footer_linex_list_queue = []

            # output this pages header
            offset, sechead_context = output_linex_list_with_offset(footer_linex_list,
                                                                    offsets_line_list=offsets_line_list,
                                                                    offset=offset,
                                                                    sechead_context=sechead_context,
                                                                    pdf_text_doc=pdf_text_doc)


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


def parse_document(file_name: str,
                   work_dir: str) \
                   -> PDFTextDoc:
    base_fname = os.path.basename(file_name)

    doc_text = strutils.loads(file_name)
    # len_doc_text = len(doc_text)

    cpoint_cunit_mapper = TextCpointCunitMapper(doc_text)
    unused_doc_len, str_offsets, line_breaks, pblock_offsets, page_offsets = \
        pdfutils.load_pdf_offsets(pdfutils.get_offsets_file_name(file_name),
                                  cpoint_cunit_mapper)
    # print('doc_len = {}, another {}'.format(doc_len, len(doc_text)))

    nl_text, linebreak_offset_list = \
        pdfdocutils.text_offsets_to_nl(base_fname,
                                       doc_text,
                                       line_breaks,
                                       work_dir=work_dir)
    if IS_DEBUG_MODE:
        newline_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.newline.txt'))
        strutils.dumps(nl_text, newline_fname)

    # txtreader.dumps(nl_text, '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nl.txt')))

    linebreak_arr = array.array('i', linebreak_offset_list)  # type: ArrayType

    paraline_fn = pdfdocutils.get_paraline_fname(base_fname, work_dir)
    pageinfo_list = init_pageinfo_list(doc_text=doc_text,
                                       nl_text=nl_text,
                                       line_breaks=line_breaks,
                                       pblock_offsets=pblock_offsets,
                                       paraline_fname=paraline_fn,
                                       page_offsets=page_offsets,
                                       str_offsets=str_offsets)

    if IS_DEBUG_MODE:
        pdfdocutils.save_page_list_by_lines(pageinfo_list,
                                            doc_text=doc_text,
                                            file_name=file_name,
                                            extension='.raw.byline.tsv',
                                            work_dir=work_dir)

    pdf_text_doc = PDFTextDoc(file_name,
                              doc_text,
                              cpoint_cunit_mapper=cpoint_cunit_mapper,
                              pageinfo_list=pageinfo_list,
                              linebreak_arr=linebreak_arr)

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
    return pdf_text_doc


# pylint: disable=invalid-name
def merge_adjacent_line_with_special_attr(apage):
    """Mark adjacent lines that 'signature' or 'address' attributes with same block_num
       in their linex attributes.
    """
    special_attrs = ['signature', 'address']
    for special_attr in special_attrs:
        if getattr(apage.attrs, 'has_{}'.format(special_attr)):
            prev_line = apage.content_line_list[0]
            prev_block_num = prev_line.block_num
            prev_has_special_attr = getattr(prev_line.attrs, special_attr)
            for linex in apage.content_line_list[1:]:
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
        pdftxt_doc.save_debug_lines('.paged.bef.merge.tsv')

    for page in pdftxt_doc.page_list:
        add_doc_structure_to_page(page, pdftxt_doc)
        # break blocks if they are in the middle of header, english sents
        # adjust_blocks_in_page(page, pdftxt_doc)

    # this only add footer and header's start, end to
    # apage.exclude_offsets
    update_page_removed_lines(pdftxt_doc)

    if IS_DEBUG_MODE:
        pdftxt_doc.save_debug_lines('.paged.after.merge.tsv')
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

    # block_list_map is not reference by anything afterward
    # Not impact anything, so removed.
    """
    # Redo block info because they might be in different
    # pages.  This block_list_map is at document level instead
    # of at page level.  Maybe remove in the future.
    # we don't want to deal with multi-page paragraphs
    block_list_map = defaultdict(list)  # type: DefaultDict[int, List[LineWithAttrs]]
    for apage in pdftxt_doc.page_list:
        for linex in apage.content_line_list:
            block_num = linex.block_num
            block_list_map[block_num].append(linex)
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
    prev_line_text = ''
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

    for line_num, line in enumerate(apage.line_list, 1):
        is_skip = False
        if docstructutils.is_line_toc_heading(line.line_text):
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
                                             line.linebreak,
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
        elif docstructutils.is_line_header(line.line_text,
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
            if secheadutils.is_line_sechead_prefix(line.line_text):
                sechead_tuple = docstructutils.extract_line_sechead(line.line_text)
                if sechead_tuple:
                    line.attrs.sechead = sechead_tuple

        # 2nd stage of rules
        is_footer, unused_score = docstructutils.is_line_footer(line.line_text,
                                                                line_num,
                                                                num_line_in_page,
                                                                line.linebreak,
                                                                # 1-based
                                                                apage.attrs.page_num_index,
                                                                line.is_english,
                                                                line.is_centered,
                                                                line.align,
                                                                line.lineinfo.yStart)
        if is_footer:
            line.attrs.footer = True
            is_skip = True
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

        prev_line_text = line.line_text
        if not is_skip and line.lineinfo.yStart < footer_yStart:
            content_line_list.append(line)

    # if footer is found, set everything afterward as footer
    # if footer_index != -1:
    if footer_yStart != MAX_FOOTER_YSTART:
        for linex in apage.line_list:
            if linex.lineinfo.yStart >= footer_yStart:
                linex.attrs.footer = True

    apage.content_line_list = content_line_list
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
        print("52341235123423 apage.page_num = {}, table_of_content_line_idx = {}".format(
            apage.page_num, table_of_content_line_idx))
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
        sechead_tuple = docstructutils.extract_line_sechead(line.line_text, prev_line_text)
        is_sechead_prefix = secheadutils.is_line_sechead_prefix(line.line_text)
        if sechead_tuple or is_sechead_prefix:
            if apage.attrs.has_toc and not deactivate_toc_detection:
                if IS_DEBUG_TOC:
                    print("===323=5 sechead== line is toc, %d [%s]" %
                          (line.page_num, line.line_text))
                line.attrs.toc = True
            line.attrs.sechead = sechead_tuple
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
