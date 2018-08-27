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
from kirke.docstruct.pdfoffsets import GroupedBlockInfo, LineInfo3, LineWithAttrs
from kirke.docstruct.pdfoffsets import PLineAttrs
from kirke.docstruct.pdfoffsets import PageInfo3, PBlockInfo, PDFTextDoc, StrInfo
from kirke.utils import strutils, txtreader, mathutils
from kirke.utils.textoffset import TextCpointCunitMapper

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# for setting footer attribute when reading pdf.offsets.json files from PDFBox
MAX_FOOTER_YSTART = 10000

IS_DEBUG_MODE = True
IS_DEBUG_TOC = False

# linepos.LnPos = start, end, line_num, is_gap
# attr_list = List[Any]    # this is very unsatisfying
# span_se_list = List[Tuple[linepos.LnPos, linepos.LnPos]]
# offsets_line_list = List[Tuple[span_se_list, str, attr_list]]
# gap_span_list = List[Tuple[int, int]]

# pylint: disable=too-many-locals, too-many-statements
def to_paras_with_attrs(pdf_text_doc: PDFTextDoc,
                        file_name: str,
                        work_dir: str) \
                        -> Tuple[List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]],
                                            PLineAttrs]],
                                 str,
                                 List[Tuple[int, int]]]:
    """Convert a pdfbox's text into NLP text, with page number gaps.

    Warning: The offsets after this transformation differs from original text.

    Returns: paras2_with_attrs
             para2_doc_text (the nlp text)
             gap_span_list (probably should be removed in the future)
    """
    base_fname = os.path.basename(file_name)

    if IS_DEBUG_MODE:
        pdf_text_doc.save_debug_blocks(work_dir=work_dir, extension='.paged.blocks.tsv')

    offset = 0
    out_line_list = []  # type: List[str]
    # pylint: disable=line-too-long
    offsets_line_list = []  # type: List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]], PLineAttrs]]

    # para_with_attrs, from_z, to_z, line_text, attrs (toc, header, footer, sechead)
    # sechead_context is now either
    #    sechead_context = None or ('sechead', 'Section 9.02.', "Vendors' Warranties. ", 35)
    sechead_context = None  # type: Optional[Tuple[str, str, str, int]]
    not_gapped_line_nums = set([])  # type: Set[int]

    # not_empty_line_num = 0
    # pylint: disable=too-many-nested-blocks
    for page_num, grouped_block_list in enumerate(pdf_text_doc.paged_grouped_block_list, 1):
        apage = pdf_text_doc.page_list[page_num - 1]
        # because we merge lines across pages, we should do this gap span identification at
        # global level
        for grouped_block in grouped_block_list:
            # is_multi_line = pdfdocutils.is_block_multi_line(grouped_block.line_list)
            is_multi_line = False
            if is_multi_line:
                # TODO, jshaw, this doesn't handle the page_num gap line correct yet.
                # It should similar to the code for not is_multi-line
                for linex in grouped_block.line_list:
                    out_line = pdf_text_doc.doc_text[linex.lineinfo.start:linex.lineinfo.end]
                    # sorted(linex.attrs.items())
                    pline_attrs = linex.to_attrvals()  # type: PLineAttrs

                    if linex.line_text and linex.attrs.sechead:
                        sechead_context = linex.attrs.sechead
                    elif sechead_context:
                        pline_attrs.sechead = sechead_context

                    out_line_list.append(out_line)
                    span_se_list = [(linepos.LnPos(linex.lineinfo.start,
                                                   linex.lineinfo.end),
                                     linepos.LnPos(offset, offset + len(out_line)))]
                    offsets_line_list.append((span_se_list, pline_attrs))
                    offset += len(out_line) + 1  # to add eoln

                    not_gapped_line_nums.add(linex.lineinfo.line_num)
            else:
                block_lines = []  # type: List[str]
                first_linex = grouped_block.line_list[0]
                pline_attrs = first_linex.to_attrvals()

                # don't check for block.line_list length here
                # There are lines with sechead followed by sentences
                if first_linex.line_text and first_linex.attrs.sechead:
                    sechead_context = first_linex.attrs.sechead
                elif sechead_context:
                    pline_attrs.sechead = sechead_context

                prev_page_num = -1
                prev_linex = None
                span_se_list = []
                for linex in grouped_block.line_list:
                    if not prev_linex:
                        # prev_linex is None is never used due to prev_page_num == 1
                        prev_linex = linex

                    if prev_page_num != -1 and linex.page_num != prev_page_num:
                        gap_frto_list = pdfdocutils.get_gap_frto_list(prev_linex,
                                                                      linex,
                                                                      apage,
                                                                      # page_num is the next page
                                                                      pdf_text_doc.page_list[page_num])
                        # gap_frto_list = False
                        if gap_frto_list:
                            print("gap_frto_list: {}".format(gap_frto_list))
                            # span_se_list.extend(gap_frto_list)
                            # simply add a break line, the gap will be done correctly elsewhere
                            gap_line_x_attrs = gap_frto_list[0]
                            # use -2, just in case -1 + 1 == 0, and line_num is 0
                            # jshaw, 2018-08-25, the two lines below are the only time
                            # is_gap=True was set for LnPos before.  Now, removed.
                            span_se_list.append((linepos.LnPos(gap_line_x_attrs.lineinfo.start,
                                                               gap_line_x_attrs.lineinfo.start),
                                                 # -100 will be reset later
                                                 linepos.LnPos(offset, offset)))

                    out_line = pdf_text_doc.doc_text[linex.lineinfo.start:linex.lineinfo.end]
                    block_lines.append(out_line)
                    span_se_list.append((linepos.LnPos(linex.lineinfo.start, linex.lineinfo.end),
                                         linepos.LnPos(offset, offset + len(out_line))))
                    offset += len(out_line) + 1  # to add eoln

                    not_gapped_line_nums.add(linex.lineinfo.line_num)

                    prev_linex = linex
                    prev_page_num = linex.page_num

                block_text = ' '.join(block_lines)
                out_line_list.append(block_text)
                offsets_line_list.append((span_se_list, pline_attrs))

            out_line_list.append('')
            span_se_list = [(linepos.LnPos(linex.lineinfo.end+2, linex.lineinfo.end+2),
                             linepos.LnPos(offset, offset))]
            # in future, maybe make this into a constant
            empty_pline_attrs = PLineAttrs()
            offsets_line_list.append((span_se_list, empty_pline_attrs))
            offset += 1

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

    # figure out the gap span, this has to be done at document level because
    # line sometimes are merged into the block in the previous page
    gap_span_list = []  # type: List[Tuple[int, int]]
    for page in pdf_text_doc.page_list:
        for linex in page.line_list:
            if linex.line_text:  # not empty line
                line_num = linex.lineinfo.line_num
                if line_num not in not_gapped_line_nums:
                    gap_span_list.append((linex.lineinfo.start, linex.lineinfo.end))

    # the last '\n' is for the last line
    nlp_text = '\n'.join(out_line_list) + '\n'

    if IS_DEBUG_MODE:
        pdf_nlp_txt_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.pdf.nlp.txt'))
        txtreader.dumps(nlp_text, pdf_nlp_txt_fn)
        print('wrote {}'.format(pdf_nlp_txt_fn), file=sys.stderr)

        pdf_nlp_debug_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt',
                                                                       '.pdf.nlp.debug.tsv'))
        with open(pdf_nlp_debug_fn, 'wt') as fout2:
            # for from_to_span_list, out_line, attr_list in offsets_line_list:
            for from_to_span_list, pline_attrs in offsets_line_list:
                print('{}\t{}'.format(from_to_span_list, pline_attrs), file=fout2)
        print('wrote {}'.format(pdf_nlp_debug_fn), file=sys.stderr)

    return offsets_line_list, nlp_text, gap_span_list


# pylint: disable=too-many-locals, too-many-statements
def init_pageinfo_list(doc_text: str,
                       nl_text: str,
                       paraline_fname: str,
                       page_offsets: List[Dict],
                       str_offsets: List[Dict]) \
                       -> List[PageInfo3]:
    lxid_strinfos_map = defaultdict(list)  # type: DefaultDict[int, List[StrInfo]]
    min_diff = float("inf")
    prev_y = 0
    all_diffs = []
    page_nums = {}
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
        y_diff = yStart - prev_y

        if y_diff < min_diff and y_diff > 0:
            all_diffs.append(y_diff)
        prev_y = yStart
        str_text = nl_text[start:end]
        if yStart < 100 and not str_text.strip():
            pass
        else:
            page_nums[line_num] = page_num
            lxid_strinfos_map[line_num].append(StrInfo(start, end, xStart, xEnd, yStart))

    pgid_pblockinfos_map = defaultdict(list)  # type: DefaultDict[int, List[PBlockInfo]]
    bxid_lineinfos_map = defaultdict(list)  # type: DefaultDict[int, List[LineInfo3]]
    # tmp_prev_end = 0
    block_num = 0
    mode_diff = int(max(set(all_diffs), key=all_diffs.count))
    tmp_end = 0
    start = None
    tmp_strinfos = []
    block_info_list = []
    max_line_num = max(lxid_strinfos_map.keys())
    for line_num in range(0, max_line_num+1):
        if not line_num in lxid_strinfos_map.keys():
            continue
        tmp_start = lxid_strinfos_map[line_num][0].start
        tmp_end = lxid_strinfos_map[line_num][0].end
        line_len = len(nl_text[tmp_start:tmp_end].split())

        # checks the difference in y val between this line and the next,
        # if below the mode, join into a block, otherwise add block to block_info
        if line_num+1 in lxid_strinfos_map.keys() and line_len > 0:
            y_diff = int(lxid_strinfos_map[line_num+1][0].yStart -
                         lxid_strinfos_map[line_num][0].yStart)
        else:
            y_diff = -1
        if tmp_start != tmp_end and (y_diff < 0 or y_diff > mode_diff+1):
            block_num += 1
            tmp_strinfos.extend(lxid_strinfos_map[line_num])
            if not start:
                start = tmp_start
            end = tmp_end
            page_num = page_nums[line_num]
            bxid_lineinfos_map[block_num].append(LineInfo3(start,
                                                           end,
                                                           line_num,
                                                           block_num,
                                                           tmp_strinfos))
            para_line, unused_is_multi_lines, unused_not_linebreaks = \
                pdfutils.para_to_para_list(nl_text[start:end])

            block_info = PBlockInfo(start,
                                    end,
                                    block_num,
                                    page_num,
                                    para_line,
                                    bxid_lineinfos_map[block_num],
                                    False)
            pgid_pblockinfos_map[page_num].append(block_info)
            block_info_list.append(block_info)
            tmp_strinfos = []
            start = None
        else:
            if not start:
                start = lxid_strinfos_map[line_num][0].start
            tmp_strinfos.extend(lxid_strinfos_map[line_num])

    pdfdocutils.save_nltext_as_paraline_file(nl_text,
                                             block_info_list,
                                             paraline_fname)

    pageinfo_list = []  # type: List[PageInfo3]
    for page_offset in page_offsets:
        #id, start, end
        start = page_offset['start']
        end = page_offset['end']
        page_num = page_offset['id']
        pblockinfo_list = pgid_pblockinfos_map[page_num]
        pinfo = PageInfo3(doc_text, start, end, page_num, pblockinfo_list)
        pageinfo_list.append(pinfo)

    return pageinfo_list


def parse_document(file_name: str,
                   work_dir: str) \
                   -> PDFTextDoc:
    base_fname = os.path.basename(file_name)

    doc_text = strutils.loads(file_name)
    # len_doc_text = len(doc_text)

    cpoint_cunit_mapper = TextCpointCunitMapper(doc_text)
    unused_doc_len, str_offsets, line_breaks, unused_pblock_offsets, page_offsets = \
        pdfutils.load_pdf_offsets(pdfutils.get_offsets_file_name(file_name),
                                  cpoint_cunit_mapper)
    # print('doc_len = {}, another {}'.format(doc_len, len(doc_text)))

    nl_text, linebreak_offset_list = \
        pdfdocutils.text_offsets_to_nl(base_fname,
                                       doc_text,
                                       line_breaks,
                                       work_dir=work_dir)

    linebreak_arr = array.array('i', linebreak_offset_list)  # type: ArrayType

    paraline_fn = pdfdocutils.get_paraline_fname(base_fname, work_dir)
    pageinfo_list = init_pageinfo_list(doc_text=doc_text,
                                       nl_text=nl_text,
                                       paraline_fname=paraline_fn,
                                       page_offsets=page_offsets,
                                       str_offsets=str_offsets)

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

    return pdf_text_doc


# TODO, 2018-08-24
# nobody is calling this???
# if so, remove
def merge_if_continue_to_next_page(prev_page, cur_page):
    if not prev_page.content_line_list or not cur_page.content_line_list:
        return
    last_line = prev_page.content_line_list[-1]
    words = last_line.line_text.split()
    last_line_block_num = last_line.block_num
    last_line_align = last_line.align

    first_line = cur_page.content_line_list[0]
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
    if first_line.is_centered or first_line.attrs.sechead:
        return


    # 8 because average word per sentence is known to be around 7
    if len(words) >= 8 and (words[-1][-1].islower() or strutils.is_not_sent_punct(words[-1][-1])):
        if not first_line.attrs.sechead:
            first_line_block_num = first_line.block_num
            for linex in cur_page.content_line_list:
                if linex.block_num == first_line_block_num:
                    linex.block_num = last_line_block_num
                else:
                    break

def reset_all_is_english(pdftxt_doc):
    block_list_map = defaultdict(list)
    page_special_attrs = ['signature', 'address']
    for apage in pdftxt_doc.page_list:
        for linex in apage.content_line_list:
            block_num = linex.block_num
            block_list_map[block_num].append(linex)
            # set up a page's special attrs for optimization later
            for special_attr in page_special_attrs:
                if getattr(linex.attrs, special_attr):
                    setattr(apage.attrs, 'has_{}'.format(special_attr), True)

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
                    if getattr(linex.attrs, special_attr):
                        special_attr_map[special_attr] = True

            # distribute special attribute to all linex
            for special_attr in special_attrs:
                if special_attr_map.get(special_attr):
                    for linex in linex_list:
                        setattr(linex.attrs, special_attr, True)

# pylint: disable=invalid-name
def merge_adjacent_line_with_special_attr(apage):
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


# break blocks if they are in the middle of header, english sents
def adjust_blocks_in_page(apage,
                          unused_pdftxt_doc: PDFTextDoc):
    tmp_block_list = docstructutils.line_list_to_block_list(apage.line_list)

    is_adjusted = False
    for block in tmp_block_list:
        if block[0].attrs.header:
            not_header_index = -1
            for line_seq, linex in enumerate(block[1:], 1):
                if not linex.attrs.header:
                    not_header_index = line_seq
                    break

            # this is for a block with 2 header, but followed by english
            # sentence.  Basically mixed up because of no space.
            if not_header_index != -1 and block[not_header_index].is_english:
                for after_linex in block[not_header_index:]:
                    after_linex.block_num += 10000  # to separate out line
                    is_adjusted = True
            elif docstructutils.is_block_all_not_english(block):
                # This is in ST-121 Form, NY State Exempt Use Certificate.
                # https://www.tax.ny.gov/pdf/current_forms/st/st121_fill_in.pdf
                # Top right header.
                """
                (3/10)
                Pages 1 and 2 must
                be completed by the
                purchaser and given
                to the seller
                """
                for linex in block:
                    linex.attrs.header = True
                    is_adjusted = True

    if not is_adjusted:
        return

    # now remove potential newly added toc lines
    tmp_list = []
    for linex in apage.content_line_list:
        if not linex.attrs.header:
            tmp_list.append(linex)
    apage.content_line_list = tmp_list

def update_page_removed_lines(pdftxt_doc: PDFTextDoc) -> None:
    rm_list = []  # type: List[LineWithAttrs]
    for apage in pdftxt_doc.page_list:
        for linex in apage.line_list:
            if linex.attrs.header or \
               linex.attrs.footer:
                pdftxt_doc.removed_lines.append(linex)
                rm_list.append(linex)
    # nothing to remove, nothing to update
    if not rm_list:
        return
    # jshaw, 2018-08-25
    # in the future, merge adjacent lines, based on
    # the text between the lines are all spaces or nl
    # Then, simply  have an set of offsets that should
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
    update_page_removed_lines(pdftxt_doc)

    if IS_DEBUG_MODE:
        pdftxt_doc.save_debug_lines('.paged.after.merge.tsv')
        pdfdocutils.save_removed_lines(pdftxt_doc, '.rmlines.tsv')
        pdfdocutils.save_exclude_lines(pdftxt_doc, '.exclude.lines.tsv')

    # now we have basic block_group with correct
    # is_english set.  Useful for merging
    # blocks with only 1 lines as table, or signature section

    for apage in pdftxt_doc.page_list:
        merge_adjacent_line_with_special_attr(apage)

    # Redo block info because they might be in different
    # pages.
    block_list_map = defaultdict(list)
    for apage in pdftxt_doc.page_list:
        for linex in apage.content_line_list:
            block_num = linex.block_num
            block_list_map[block_num].append(linex)

    # the block list is for the document, not a page
    paged_grouped_block_list = defaultdict(list)
    for block_num, line_list in sorted(block_list_map.items()):
        # take the page of the first line in a block as the page_num
        page_num = line_list[0].page_num
        paged_grouped_block_list[page_num].append(GroupedBlockInfo(page_num,
                                                                   block_num,
                                                                   line_list))
    # each page is a list of grouped_block
    pdftxt_doc.paged_grouped_block_list = []
    for page_num in range(1, pdftxt_doc.num_pages + 1):
        grouped_block_list = paged_grouped_block_list[page_num]
        pdftxt_doc.paged_grouped_block_list.append(grouped_block_list)


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
        if getattr(apage.attrs, 'has_{}'.format(special_attr)):
            # print("skip table_and_char_detection because of has_{}".format(special_attr))
            is_skip_table_and_chart_detection = True

    if is_skip_table_and_chart_detection:
        return

    # handle table and chart identification
    grouped_block_list = apage.grouped_block_list
    # if successful, markup_table_block_by_non_english will set 'has_table', a page attribute
    markup_table_block_by_non_english(grouped_block_list, apage)
    if not getattr(apage.attrs, 'has_table'):
        markup_table_block_by_columns(grouped_block_list, page_num)

    # create the annotation for table and chart from apage.content_line_list
    extract_tables_from_markups(apage, pdf_txt_doc)


# pylint: disable=invalid-name
def markup_table_block_by_non_english(grouped_block_list, apage):
    # print("\nmarkup_table_block_by_non_english, apage = {}".format(apage.page_num))
    for grouped_block in grouped_block_list:
        num_non_english_line = 0  # also short
        num_numeric_line = 0
        num_group_line = len(grouped_block.line_list)
        for linex in grouped_block.line_list:
            # pylint: disable=too-many-boolean-expressions
            if len(linex.line_text) < 30 and \
               not linex.is_english and linex.is_centered and \
               len(linex.line_text.split()) >= 2 and \
               not linex.attrs.sechead and \
               not secheadutils.is_line_sechead_prefix(linex.line_text):
                num_non_english_line += 1
            if not linex.is_english and len(strutils.extract_numbers(linex.line_text)) > 2:
                num_numeric_line += 1

        #print("num_non_english_line = {}, num_group_line = {}".format(num_non_english_line,
        #                                                              num_group_line))

        if (num_non_english_line >= 4 and float(num_non_english_line) / num_group_line >= 0.2) or \
           (num_numeric_line >= 2 and float(num_numeric_line) / num_group_line >= 0.4):
            # this is a tables
            apage.attrs.has_table = True  # so that other table routine doesn't have to fire

            merged_linex_list = grouped_block.line_list
            table_prefix = 'table'
            table_count = 300
            table_name = '{}-p{}-{}'.format(table_prefix, apage.page_num, table_count)
            for linex in merged_linex_list:
                linex.attrs.table = table_name

            block_first_linex = merged_linex_list[0]
            # merge blocks as we see fit
            # print("merge_centered_line_before_table(), markup_table_by_non_english...")
            merge_centered_lines_before_table(block_first_linex.lineinfo.line_num,
                                              block_first_linex.block_num,
                                              # page_linex_list,
                                              apage.content_line_list,
                                              table_prefix,
                                              table_name)


def extract_tables_from_markups(apage, pdf_txt_doc):
    tableid_lines_map = defaultdict(list)
    chartid_lines_map = defaultdict(list)
    for linex in apage.content_line_list:
        table_id = linex.attrs.table
        chart_id = linex.attrs.chart
        if table_id:
            tableid_lines_map[table_id].append(linex)
        elif chart_id:
            chartid_lines_map[chart_id].append(linex)
    for unused_tableid, table_lines in tableid_lines_map.items():
        pdf_txt_doc.special_blocks_map['table'] \
                   .append(pdfoffsets.lines_to_block_offsets(table_lines,
                                                             'table',
                                                             apage.page_num))
    for unused_chartid, chart_lines in chartid_lines_map.items():
        pdf_txt_doc.special_blocks_map['chart'] \
                   .append(pdfoffsets.lines_to_block_offsets(chart_lines,
                                                             'chart',
                                                             apage.page_num))

# pylint: disable=too-many-branches, too-many-statements, too-many-locals
def add_doc_structure_to_page(apage: PageInfo3,
                              pdf_txt_doc: PDFTextDoc):
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
                print("===323=1tocheading== line is toc, %d [%s]" %
                      (line.page_num, line.line_text))
            line.attrs.toc = True
            has_toc_heading = True
            num_toc_line += 10  # I know this is not true yet
            is_skip = True
            toc_block_list.append(line)
        elif docstructutils.is_line_toc(line.line_text):
            if IS_DEBUG_TOC:
                print("===323=2linetoc== line is toc, %d [%s]" %
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
            if line_seq >= first_sechead and line_seq <= last_sechead:
                if IS_DEBUG_TOC:
                    print("===323=7sechead33== line is toc, %d [%s]" %
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

    # we are here, so there must be toc lines
    first_toc_line, _ = tmp_toc_lines[0]
    last_toc_line, _ = tmp_toc_lines[-1]
    if len(tmp_toc_lines) >= 10:
        if table_of_content_line_idx != -1:
            for linex in apage.line_list[table_of_content_line_idx:last_toc_line+1]:
                if IS_DEBUG_TOC:
                    print("===323=3beforelasttoc1== line is toc, %d [%s]" %
                          (linex.page_num, linex.line_text))
                linex.attrs.toc = True
        else:
            for linex in apage.line_list[first_toc_line:last_toc_line+1]:
                if IS_DEBUG_TOC:
                    print("===323=4beforelasttoc1== line is toc, %d [%s]" %
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
                    print("===323=5sechead== line is toc, %d [%s]" %
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
                    print("===323=6too-small== line is toc, %d [%s]" %
                          (line.page_num, line.line_text))
                linex.attrs.toc = True

def get_longest_consecutive_line_group(linex_list):
    # linex_list = remove_linex_with_sechead(linex_list)
    # linex_list = list(filter(lambda linex: not linex.attrs.sechead, linex_list))
    group_list = []
    prev_block_num = linex_list[0].block_num
    prev_align = linex_list[0].align
    cur_group = [linex_list[0]]
    group_list.append(cur_group)
    for linex in linex_list[1:]:
        # print("prev_block_num = {}, linex.block_num = {}".format(prev_block_num, linex.block_num))
        # they are consecutive and aligned
        if linex.block_num == prev_block_num + 1 and linex.align == prev_align:
            cur_group.append(linex)
        else:
            cur_group = [linex]
            group_list.append(cur_group)
        prev_block_num = linex.block_num
        prev_align = linex.align

    group_list_by_size = sorted([(len(groupx), groupx) for groupx in group_list], reverse=True)
    """
    for group_num, (gsize, groupx) in enumerate(group_list_by_size):
        print("group #{}, size = {}".format(group_num, gsize))
        for linex in groupx:
            print("    linex: {}".format(linex.tostr4()))
    """
    return group_list_by_size[0][1]  # return longest linex_list


# pylint: disable=invalid-name, too-many-statements, too-many-locals
def collapse_similar_aligned_block_lines(grouped_block_list, page_num):

    # first try to see if tables should be formed from separated blocks
    num_table_one_line_block = 0
    num_line = 0
    blocks_with_one_line = []
    for grouped_block in grouped_block_list:
        num_line_in_block = len(grouped_block.line_list)
        if num_line_in_block == 1:
            linex = grouped_block.line_list[0]
            if len(linex.line_text) < 30 and not linex.is_english and \
               not linex.attrs.sechead and \
               not secheadutils.is_line_sechead_prefix(linex.line_text):
                num_table_one_line_block += 1
                blocks_with_one_line.append(linex)
        num_line += num_line_in_block

    if num_table_one_line_block < 3:
        return grouped_block_list
    else:
        #print('collapse_similar_aligned_block_lines, page = %d, num_table_one_line_block = %d' %
        #      (page_num, num_table_one_line_block))

        # find the longest consecutive one
        longest_consecutive_group = get_longest_consecutive_line_group(blocks_with_one_line)
        if len(longest_consecutive_group) < 3:
            return grouped_block_list

    # found a collapsable group
    #for linex in longest_consecutive_group:
    #    print("collapse_line = {}".format(linex.tostr4()))

    # now expand backward, the lines or block before
    first_merged_linex = longest_consecutive_group[0]
    last_merged_linex = longest_consecutive_group[-1]
    first_merged_line_num = first_merged_linex.lineinfo.line_num
    last_merged_line_num = last_merged_linex.lineinfo.line_num
    page_linex_list = []
    page_line_index = 0
    first_index, last_index = -1, -1
    last_index = -1
    for grouped_block in grouped_block_list:
        for linex in grouped_block.line_list:
            if linex.lineinfo.line_num == first_merged_line_num:
                first_index = page_line_index
            elif linex.lineinfo.line_num == last_merged_line_num:
                last_index = page_line_index
            page_linex_list.append(linex)
            page_line_index += 1
    # print('first_index = {}, last_index = {}'.format(first_index, last_index))

    line_index = first_index - 1
    while line_index >= 0:  # merge if align and not en
        linex = page_linex_list[line_index]
        if not linex.is_english and linex.align == first_merged_linex.align and \
           not (linex.attrs.sechead or \
                secheadutils.is_line_sechead_prefix(linex.line_text)):
            line_index -= 1
        else:
            break
    found_block_start_line_num = line_index + 1

    # now expand forward
    line_index = last_index + 1
    last_index = len(page_linex_list)
    while line_index < last_index:  # merge if align and not en
        linex = page_linex_list[line_index]
        if not linex.is_english and linex.align == first_merged_linex.align and \
           not (linex.attrs.sechead or secheadutils.is_line_sechead_prefix(linex.line_text)):
            line_index += 1
        else:
            break
    found_block_end_line_num = line_index

    num_line_with_number = 0
    merged_block_num = page_linex_list[found_block_start_line_num].block_num
    merged_linex_list = []
    for i in range(found_block_start_line_num, found_block_end_line_num):
        linex = page_linex_list[i]
        linex.block_num = merged_block_num
        merged_linex_list.append(linex)
        if strutils.find_number(linex.line_text):
            num_line_with_number += 1

    # check to see if the merged block should be a table
    merged_block_size = len(merged_linex_list)
    if float(num_line_with_number) / merged_block_size > 0.7:  # this is a table
        table_prefix = 'table'
        table_count = 200
        table_name = '{}-p{}-{}'.format(table_prefix, page_num, table_count)
        for linex in merged_linex_list:
            linex.attrs.table = table_name

        block_first_linex = merged_linex_list[0]
        # merge blocks as we see fit
        # print("merge_centered_line_before_table(), collapse_similar...")
        merge_centered_lines_before_table(block_first_linex.lineinfo.line_num,
                                          block_first_linex.block_num,
                                          page_linex_list,
                                          table_prefix,
                                          table_name)

    grouped_block_list = pdfoffsets.line_list_to_grouped_block_list(page_linex_list, page_num)

    return grouped_block_list


# the lines are sorted by xStart
# pylint: disable=invalid-name
def get_lines_from_block_nums_sorted_xStart(block_num_list, block_lines_map):
    xStart_line_list = []
    for block_num in block_num_list:
        for linex in block_lines_map[block_num]:
            xStart_line_list.append((linex.lineinfo.xStart, linex))
    return [x[1] for x in sorted(xStart_line_list)]


# pylint: disable=invalid-name
def get_lines_from_block_nums_sorted_obid(block_num_list, block_lines_map):
    obid_line_list = []
    for block_num in block_num_list:
        for linex in block_lines_map[block_num]:
            obid_line_list.append((linex.lineinfo.obid, linex))
    return [x[1] for x in sorted(obid_line_list)]


# pylint: disable=invalid-name, too-many-branches, too-many-statements, too-many-locals
def markup_table_block_by_columns(grouped_block_list, page_num):
    debug_mode = False
    # has_close_ydiffs = False
    blocks_with_similar_ys = set([])

    y_lines = []
    block_lines_map = defaultdict(list)
    for grouped_block in grouped_block_list:
        for linex in grouped_block.line_list:
            y_lines.append((linex.lineinfo.yStart, linex))
            block_lines_map[linex.block_num].append(linex)
    # pylint: disable=invalid-name
    prev_yStart = -100
    prev_block_num = -1
    page_linex_list = []  # sorted by yStart
    for yStart, linex in sorted(y_lines):
        # there is multiple lines in the same yStart
        cur_block_num = linex.block_num
        # TODO, MAYBE ISSUE: jshaw
        # there might be multiple lines with yStart <= 5.0
        if yStart - prev_yStart <= 5.0:
            # There are chart that the block_num are the same, but ydiff is similar.
            # Yet to finally decide.
            # A formula that's merged by pdfbox
            if prev_block_num != cur_block_num:
                # print("page_num = {}, ydiff {:.2f}, pre_block_num = {}, cur_block_num = {}".
                #    format(page_num, yStart - prev_yStart, prev_block_num, cur_block_num))
                # print("     prev_yStart = {:.1f}, yStart = {:.1f}".format(prev_yStart, yStart))
                blocks_with_similar_ys.add((prev_block_num, cur_block_num))
        page_linex_list.append(linex)
        prev_yStart = yStart
        prev_block_num = cur_block_num

    if not blocks_with_similar_ys:  # no overlap
        return

    # if there is blocks_with_similar_ys, but only 2 lines
    # if one of them is short, probably a mistake as table_by_columns.
    # Probably a pdfbox line separation mistake
    if len(blocks_with_similar_ys) == 1:
        # look through the set:
        # this is only 1 tuple in the set, so only iterate once
        for block_with_similar_ys in blocks_with_similar_ys:
            sorted_block_num_list = sorted(block_with_similar_ys)
            merged_linex_list = get_lines_from_block_nums_sorted_obid(sorted_block_num_list,
                                                                      block_lines_map)
            # one of them is very short
            if len(merged_linex_list) == 2:
                # only if one is a long line, the other is short
                if (len(merged_linex_list[0].line_text) < 20 and \
                    len(merged_linex_list[1].line_text) > 40) or \
                   (len(merged_linex_list[1].line_text) < 20 and \
                    len(merged_linex_list[0].line_text) > 40):
                    # print("only 2 lines are in 2 columns, reject as a table")
                    return

    if debug_mode:
        print("markup_table_block_by_columns(), page_num = {}".format(page_num))
        for yStart, linex in sorted(y_lines):
            print("ystart= {}, {} || {}".format(yStart, linex.tostr5(), linex.line_text))
    # now, the blocks are in pairs. Change them to groups or lists
    blocks_with_similar_ys = mathutils.pairs_to_sets(blocks_with_similar_ys)

    # now merge row first, then merge rows into table
    block_first_linex = None

    table_count = 1
    for block_num_set in blocks_with_similar_ys:
        sorted_block_num_list = sorted(block_num_set)
        min_block_num = sorted_block_num_list[0]

        # merged_linex_list = get_lines_from_block_nums_sorted_xStart(sorted_block_num_list,
        #                                                             block_lines_map)
        merged_linex_list = get_lines_from_block_nums_sorted_obid(sorted_block_num_list,
                                                                  block_lines_map)
        # merged_block_lines_map[sorted_block_num_list[0]] =
        # print("block_num_set, makrup_table_by_columns... {}".format(block_num_set))

        table_prefix = 'table'
        table_name = '{}-p{}-{}'.format(table_prefix, page_num, table_count)
        for block_num in block_num_set:
            block_lines = block_lines_map[block_num]
            for linex in block_lines:
                linex.attrs.table = table_name
                linex.block_num = min_block_num
        table_count += 1

    # merge tables that are adjacent to each other
    # basically each column is a row
    prev_table_label = ''
    prev_table_block_num = -1
    for linex in page_linex_list:
        cur_table_label = linex.attrs.table
        if cur_table_label and prev_table_label:
            if cur_table_label != prev_table_label:
                linex.attrs.table = prev_table_label
                linex.attrs.table_row = linex.block_num
                linex.block_num = prev_table_block_num
            else:  # if they are equal, do nothing
                linex.attrs.table_row = linex.block_num
        elif cur_table_label and not prev_table_label:  # found a new table
            prev_table_label = cur_table_label
            prev_table_block_num = linex.block_num
            linex.attrs.table_row = linex.block_num
            # current label is already correct
        elif not cur_table_label:  # both prev_table_label is empty or has value
            prev_table_label = ''
            prev_table_block_num = -1

    table_lines_map = defaultdict(list)
    for linex in page_linex_list:
        table_label = linex.attrs.table
        if table_label:
            table_lines_map[table_label].append(linex)

    for table_label, linex_list in table_lines_map.items():
        block_first_linex = None
        align_count_map = defaultdict(int)

        for linex in linex_list:
            # print("   linex {} || {}".format(linex.tostr5(), linex.line_text))
            if not block_first_linex:
                block_first_linex = linex
            align_count_map[linex.align[:2]] += 1

        #for align_st, count in align_count_map.items():
        #    print("align = {}, count = {}".format(align_st, count))

        align_count_gt_2_list = list(filter(lambda x: x[1] > 2, align_count_map.items()))
        #for align_st, count in align_count_gt_2_list:
        #    print("align = {}, count = {}".format(align_st, count))

        if len(align_count_gt_2_list) <= 1:  # 0 means short table
            # table_prefix = 'table'
            # no need to do anything, linex.attrs.table is correct already
            table_prefix = 'table'
            table_name = block_first_linex.attrs.table
        else:
            table_prefix = 'chart'
            for linex in linex_list:
                linex.attrs.chart = linex.attrs.table.replace('table', 'chart')
                linex.attrs.table = ''
            table_name = block_first_linex.attrs.chart

        # merge blocks as we see fit
        if block_first_linex:
            # print("merge_centered_line_before_table(), markup_table_by_columns...")
            merge_centered_lines_before_table(block_first_linex.lineinfo.line_num,
                                              block_first_linex.block_num,
                                              page_linex_list,
                                              table_prefix,
                                              table_name)


def get_merge_reason(linex: LineWithAttrs) -> str:
    special_attrs = ['signature', 'address', 'table', 'chart']
    for special_attr in special_attrs:
        if getattr(linex.attrs, special_attr):
            return special_attr
    return ''


# pylint: disable=invalid-name, too-many-statements, too-many-locals
def merge_centered_lines_before_table(line_num,
                                      block_num,
                                      content_linex_list,
                                      table_prefix,
                                      table_name):
    table_start_idx = -1
    debug_mode = False

    for line_idx, linex in enumerate(content_linex_list):
        if linex.lineinfo.line_num == line_num:
            if debug_mode:
                print("hello {} || {}".format(linex.tostr3(), linex.line_text))
            table_start_idx = line_idx
            break

    if debug_mode:
        print("merge_centered_lines_before_table(), xxx, table_start_idx = %d" %
              (table_start_idx, ))
    first_merge_reason = get_merge_reason(content_linex_list[0])
    if table_start_idx != -1:
        table_start_idx -= 1
        prev_merge_reason = first_merge_reason
        while table_start_idx >= 0:
            linex = content_linex_list[table_start_idx]
            merge_reason = get_merge_reason(linex)
            if debug_mode:
                print("jjj linex = {} || {}".format(linex.tostr4(), linex.line_text[:20]))
                print("   merge_reason = [{}], prev_merge_reason = [{}]" \
                      .format(merge_reason, prev_merge_reason))
            # previous is merged due to centered and now, we have either address, signature
            # table, or chart.  don't merge
            if not prev_merge_reason and merge_reason:
                if debug_mode:
                    print("break1")
                break
            # if not english, merge with the table by block
            if linex.attrs.sechead:
                if debug_mode:
                    print('linex is sechead....')
                # we take sechead before a table, but sechead block_num stays
                # linex.block_num = block_num
                setattr(linex.attrs, table_prefix, table_name)
                # there might be multiple sechead lines before
                table_start_idx -= 1
                while table_start_idx >= 0:
                    tmp_linex = content_linex_list[table_start_idx]
                    if tmp_linex.attrs.sechead:
                        setattr(tmp_linex.attrs, table_prefix, table_name)
                        table_start_idx -= 1
                    else:
                        break
                break
            elif merge_reason != '' and \
                 merge_reason != first_merge_reason and prev_merge_reason == '':
                if debug_mode:
                    print("break due to merge_reason, linex.block_num = {}" \
                          .format(linex.block_num))
                break
            elif linex.is_centered:
                if debug_mode:
                    print('linex is centered....')
                linex.block_num = block_num
                setattr(linex.attrs, table_prefix, table_name)
            elif not linex.is_english:
                if debug_mode:
                    print('linex is not english....')
                linex.block_num = block_num
                setattr(linex.attrs, table_prefix, table_name)
            else:
                if debug_mode:
                    print('linex is breaking....')
                break
            table_start_idx -= 1
            prev_merge_reason = merge_reason

        # check if everything before table is OK to include as a part of that table
        if debug_mode:
            print("table_start_idx = {}".format(table_start_idx))
        prev_block_linex_list_map = defaultdict(list)
        for line_idx, linex in enumerate(content_linex_list):
            block_num = linex.block_num
            prev_block_linex_list_map[block_num].append(linex)
            if line_idx == table_start_idx:
                break

        is_ok_table_heading = False
        prev_block_list = [linex_list for block_num, linex_list in
                           sorted(prev_block_linex_list_map.items())]

        # for debug
        if debug_mode:
            for blockx, linex_list in enumerate(prev_block_list):
                print("\npre-block: {}".format(blockx))
                for linex in linex_list:
                    print("     linex: {} || {}".format(linex.tostr5(), linex.line_text))

        if len(prev_block_list) <= 3:  # sechead + table head + short description
            num_is_header_block = 0
            max_english_lines_in_block = 0
            for linex_list in prev_block_list:
                first_linex = linex_list[0]
                if first_linex.is_centered or first_linex.attrs.sechead:
                    num_is_header_block += 1
                elif len(linex_list) <= 3:   # short descripton shouldn't be more than 3 lines
                    num_is_header_block += 1

                if len(linex_list) > max_english_lines_in_block:
                    max_english_lines_in_block = len(linex_list)
            if debug_mode:
                print("num_is_header_block = {}, len(prev_block_list)= {}" \
                      .format(num_is_header_block,
                              len(prev_block_list)))
            if max_english_lines_in_block <= 3 and \
               float(num_is_header_block) / len(prev_block_list) >= 0.5:
                is_ok_table_heading = True

        # if there is less than 4 lines in the page before tha table
        # all all those lines as a part of table for signal gathering
        if table_start_idx < 4:
            while table_start_idx >= 0:
                if debug_mode:
                    print("zzz table_start_idx = {}, len(content_linex_list) = {}" \
                          .format(table_start_idx,
                                  len(content_linex_list)))
                tmp_linex = content_linex_list[table_start_idx]
                if not tmp_linex.is_english:
                    setattr(tmp_linex.attrs, table_prefix, table_name)
                    table_start_idx -= 1
                else:
                    break
        elif is_ok_table_heading:
            while table_start_idx >= 0:
                if debug_mode:
                    print("zz2 table_start_idx = %d, len(content_linex_list) = %d",
                          (table_start_idx, len(content_linex_list)))
                tmp_linex = content_linex_list[table_start_idx]
                setattr(tmp_linex.attrs, table_prefix, table_name)
                table_start_idx -= 1


def main():
    parser = argparse.ArgumentParser(description='Parse a document into a document structure.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file', help='a file to be annotated')

    args = parser.parse_args()
    txt_fname = args.file

    work_dir = 'dir-work'
    pdf_txt_doc = parse_document(txt_fname, work_dir=work_dir)
    to_paras_with_attrs(pdf_txt_doc, txt_fname, work_dir=work_dir)

    pdf_txt_doc.save_debug_blocks(work_dir=work_dir, extension='.paged.blocks.tsv')

    pdf_txt_doc.save_debug_pages(work_dir=work_dir, extension='.paged.debug.tsv')

    logger.info('Done.')

if __name__ == '__main__':
    main()
