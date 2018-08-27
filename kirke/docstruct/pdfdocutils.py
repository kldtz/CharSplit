import os
import logging
import sys
from typing import Dict, List, Tuple

from kirke.docstruct.pdfoffsets import LineWithAttrs, PageInfo3, PBlockInfo, PDFTextDoc
from kirke.utils import txtreader

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
    print('wrote {}, size= {}'.format(paraline_fname, len(paraline_text)),
          file=sys.stderr)


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
