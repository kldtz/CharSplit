
from collections import defaultdict
import logging
import os
import sys
# pylint: disable=unused-import
from typing import DefaultDict, Dict, List, Tuple

from kirke.docstruct import docstructutils, pdfoffsets, secheadutils
from kirke.docstruct.pdfoffsets import LineWithAttrs, PageInfo3, PBlockInfo, PDFTextDoc
from kirke.utils import mathutils, strutils, txtreader


# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IS_DEBUG_MODE = False

def get_nl_fname(base_fname: str,
                 work_dir: str) -> str:
    return '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nl.txt'))


def get_paraline_fname(base_fname: str,
                       work_dir: str) -> str:
    return '{}/{}'.format(work_dir, base_fname.replace('.txt', '.paraline.txt'))


def text_offsets_to_nl(base_fname: str,
                       orig_doc_text: str,
                       line_breaks: List[Dict],
                       work_dir: str) \
                       -> Tuple[str, List[int]]:

    debug_mode = False
    # We allow only 1 diff, some old cached file might have the issue
    # where a value == len(orig_doc_text).
    # For example, BHI doc, cached 110464.txt have this property.
    len_doc_text = len(orig_doc_text)
    linebreak_offset_list = []  # type: List[int]
    for lbrk in line_breaks:
        lbrk_offset = lbrk['offset']
        if lbrk_offset < len_doc_text:
            linebreak_offset_list.append(lbrk_offset)
        elif lbrk_offset == len_doc_text:
            # logger.warning("text_offsets_to_nl(%s), len= %d, lnbrk_offset = %d",
            #                base_fname, len_doc_text, lbrk_offset)
            pass
        else:
            logger.warning("text_offsets_to_nl(%s), len= %d, lnbrk_offset = %d",
                           base_fname, len_doc_text, lbrk_offset)
    ch_list = list(orig_doc_text)
    for linebreak_offset in linebreak_offset_list:
        ch_list[linebreak_offset] = '\n'
    nl_text = ''.join(ch_list)
    if debug_mode:
        nl_fname = get_nl_fname(base_fname, work_dir)
        txtreader.dumps(nl_text, nl_fname)
        print('wrote {}, size= {}'.format(nl_fname, len(nl_text)), file=sys.stderr)
    return nl_text, linebreak_offset_list


def save_strinfo_list(pdf_text_doc: PDFTextDoc,
                      file_name: str,
                      work_dir: str) -> None:
    out_fname = '{}/{}'.format(work_dir, os.path.basename(file_name))

    doc_text = pdf_text_doc.doc_text
    with open(out_fname, 'wt') as fout:
        for page_num, pageinfo in enumerate(pdf_text_doc.page_list, 1):
            for pblock_num, pblockinfo in enumerate(pageinfo.pblockinfo_list):
                block_line_num = 0
                for lineinfo in pblockinfo.lineinfo_list:
                    block_line_num += 1
                    for str_num, strinfo in enumerate(lineinfo.strinfo_list):
                        text = doc_text[strinfo.start:strinfo.end]
                        print("{}\t{}\t{}\t{}\t{}\t[{}]".format(page_num,
                                                                pblock_num,
                                                                block_line_num,
                                                                str_num,
                                                                str(strinfo),
                                                                text),
                              file=fout)


def save_nltext_as_paraline_file(nl_text: str,
                                 block_info_list: List[PBlockInfo],
                                 paraline_fname: str) -> None:
    # save .paraline.txt, which has the exact same size
    # as .txt file.
    # Now, switch to array replacement.  This is not affected by the wrong block info.
    # It simply override everys block based on the indexes, so guarantees not to create
    # extra stuff.
    ch_list = list(nl_text)
    for block_info in block_info_list:
        block_text = block_info.text
        # block_text is already formatted correct because of above
        # pdfutils.para_to_para_list(nl_text[start:end])
        ch_list[block_info.start:block_info.end] = list(block_text)
    paraline_text = ''.join(ch_list)
    txtreader.dumps(paraline_text, paraline_fname)
    if IS_DEBUG_MODE:
        print('wrote {}, size= {}'.format(paraline_fname, len(paraline_text)),
              file=sys.stderr)


def save_removed_lines(pdftxt_doc: PDFTextDoc,
                       extension: str,
                       work_dir: str = 'dir-work') -> None:
    base_fname = os.path.basename(pdftxt_doc.file_name)
    out_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))

    doc_text = pdftxt_doc.doc_text
    with open(out_fname, 'wt') as fout:
        for linex in pdftxt_doc.removed_lines:
            line_text = doc_text[linex.lineinfo.start:linex.lineinfo.end]
            print('page {}, {}\t{}'.format(linex.page_num,
                                           linex.tostr5(),
                                           line_text),
                  file=fout)
    print('wrote {}'.format(out_fname), file=sys.stderr)


def save_exclude_lines(pdftxt_doc: PDFTextDoc,
                       extension: str,
                       work_dir: str = 'dir-work') -> None:
    base_fname = os.path.basename(pdftxt_doc.file_name)
    out_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))

    with open(out_fname, 'wt') as fout:
        for start, end in pdftxt_doc.exclude_offsets:
            print('exclude {} {}'.format(start, end),
                  file=fout)
    print('wrote {}'.format(out_fname), file=sys.stderr)


# we do not do our own block merging in pwc version
def save_page_list_by_lines(page_list: List[PageInfo3],
                            doc_text: str,
                            file_name: str,
                            extension: str,
                            work_dir: str = 'dir-work'):
    base_fname = os.path.basename(file_name)
    out_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))
    with open(out_fname, 'wt') as fout:
        for page in page_list:
            print('\n===== page #%d, start=%d, end=%d, len(lines)= %d' %
                  (page.page_num, page.start, page.end, len(page.line_list)), file=fout)

            prev_block_num = -1
            for linex in page.line_list:

                if linex.block_num != prev_block_num:  # this is not obid
                    print(file=fout)
                print('{}\t{}'.format(linex.tostr2(),
                                      doc_text[linex.lineinfo.start:linex.lineinfo.end]),
                      file=fout)
                prev_block_num = linex.block_num

    print('wrote {}'.format(out_fname), file=sys.stderr)


# ===============================================
# The code below might no longer be used

def get_gap_frto_list(prev_linex: LineWithAttrs,
                      linex: LineWithAttrs,
                      cur_page: PageInfo3,
                      next_page: PageInfo3) \
                      -> List[LineWithAttrs]:
    """Take the footer between the prev_line and the first line of next page.

    This code tries to recover the deleted lines inside a paragraph, or a block.
    The docstructure keeps lines from different pages inside the same block if
    it believes they are of the same paragraph.  As a result, some lines, such as
    page numbers or footer are deleted.  This seems to be trying to recover those
    deleted lines.  The logic was difficult if not realizing that the deletion is
    already done somewhere else when merging the blocks earlier.

    Return a list of footer lines.
    """
    start_index, end_index = -1, -1

    result = []

    prev_line_num = prev_linex.lineinfo.line_num
    for seq, xxx_linex in enumerate(cur_page.line_list):
        if xxx_linex.lineinfo.line_num == prev_line_num:
            if seq + 1 < len(cur_page.line_list):
                start_index = seq + 1  # next line is the gapped line
            #else:
            #    start_index = -1
            break
    # start_index can be -1 or other values here

    next_line_num = linex.lineinfo.line_num
    for seq, xxx_linex in enumerate(next_page.line_list):
        if xxx_linex.lineinfo.line_num == next_line_num:
            if seq - 1 >= 0:
                end_index = seq - 1  # prev line is the gapped line
            break

    # print("start_index = {}, end_index = {}".format(start_index, end_index))

    if start_index == -1 and end_index == -1:  # ?? everything is gapped?
        return []
    elif start_index == -1:
        # nothing from first page, so start from the first line of next page
        if linex.lineinfo.line_num != next_page.line_list[0].lineinfo.line_num:
            for tmp_linex in next_page.line_list:
                if tmp_linex != linex:
                    result.append(tmp_linex)
        else:
            # start_index == -1 means there is no gap from previous.
            # if we have no gap in the next page, then there is no gap
            return []
    elif end_index == -1:
        # nothing from next page, so end from the last line of current page
        if prev_linex.lineinfo.line_num != cur_page.line_list[-1].lineinfo.line_num:
            for i in range(start_index, len(cur_page.line_list)):
                result.append(cur_page.line_list[i])
        else:
            # end_index == -1 means there is no gap from next page
            # if we have no gap in the current page, then there is no gap
            return []
    else:  # start_index and end_index both are not -1
        # start_offset = cur_page.line_list[start_index].lineinfo.start
        # end_offset = next_page.line_list[end_index].lineinfo.end   # + 1  # 1 for eoln
        if prev_linex.lineinfo.line_num != cur_page.line_list[-1].lineinfo.line_num:
            for i in range(start_index, len(cur_page.line_list)):
                result.append(cur_page.line_list[i])
        if linex.lineinfo.line_num != next_page.line_list[0].lineinfo.line_num:
            for tmp_linex in next_page.line_list:
                if tmp_linex != linex:
                    result.append(tmp_linex)

    return result


# TODO, jshaw, 2018-08-25, maybe nobody is using this
# after it was commented out
def is_block_multi_line(linex_list):

    if len(linex_list) <= 1:
        return False
    if len(linex_list) == 2:  # if first line is a sentence
        if linex_list[0].is_english and linex_list[0].num_word >= 6:
            return False
    # if more than 3 lines, if most lines are english, then multi-line is False
    num_is_english = 0
    num_not_english = 0
    for linex in linex_list:
        if linex.is_english:
            num_is_english += 1
        else:
            num_not_english += 1
    return num_is_english < num_not_english



# break blocks if they are in the middle of header, english sents
# This was commented out
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


# jshaw, 2018-08-27
# Nobody calls this.
def reset_all_is_english(pdftxt_doc):
    block_list_map = defaultdict(list)  # type: DefaultDict[int, List[LineWithAttrs]]
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


## ==================================================
# These seems to be related to tables
# but nobody calls them?
# The top level seems to be
#    add_sections_to_page(apage, pdf_txt_doc):
#        collapse_similar_aligned_block_lines
#           merge_centered_lines_before_table
#        markup_table_block_by_non_english
#        markup_table_block_by_column
#        extract_tables_from_markups
# pylint: disable=invalid-name, too-many-statements, too-many-locals


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


# jshaw, 2018-08-27
# nobody is calling this?
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
