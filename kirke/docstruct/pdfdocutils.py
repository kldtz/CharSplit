
from collections import defaultdict
import logging
import os
import sys
from typing import DefaultDict, Dict, List, Tuple

from kirke.docstruct import pdfutils
from kirke.docstruct.pdfoffsets import PageInfo3, PBlockInfo, PDFTextDoc, LineWithAttrs, LineInfo3
from kirke.utils import txtreader


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


def is_block_multi_line(linex_list: List[LineWithAttrs]) -> bool:

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


# we do not do our own block merging in pwc version
def save_nlp_paras_with_attrs(pdftxt_doc: PDFTextDoc,
                              extension: str,
                              work_dir: str = 'dir-work') -> None:
    base_fname = os.path.basename(pdftxt_doc.file_name)
    out_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))

    # List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]],
    #            PLineAttrs]],

    doc_text = pdftxt_doc.doc_text
    with open(out_fname, 'wt') as fout:
        para_seq = 1
        for para_with_attrs in pdftxt_doc.nlp_paras_with_attrs:
            from_to_lnpos_list, para_attrs = para_with_attrs

            if len(from_to_lnpos_list) == 1:
                from_lnpos, to_lnpos = from_to_lnpos_list[0]
                if from_lnpos.start == from_lnpos.end:
                    # an empty line, this is a paragraph break
                    continue

            print('\n----- para_with_attr #{}'.format(para_seq), file=fout)
            print('   para_attrs: {}'.format(para_attrs), file=fout)
            for from_lnpos, to_lnpos in from_to_lnpos_list:
                print('    ({}, {}), ({}, {}) [{}]'.format(from_lnpos.start,
                                                           from_lnpos.end,
                                                           to_lnpos.start,
                                                           to_lnpos.end,
                                                           doc_text[from_lnpos.start:
                                                                    from_lnpos.end]),
                      file=fout)
            para_seq += 1
    print('wrote {}'.format(out_fname), file=sys.stderr)


def save_nlp_paras_with_attrs_v2(pdftxt_doc: PDFTextDoc,
                                 extension: str,
                                 work_dir: str = 'dir-work') -> None:
    base_fname = os.path.basename(pdftxt_doc.file_name)
    out_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))

    # List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]],
    #            PLineAttrs]],

    nlp_text = pdftxt_doc.nlp_doc_text
    with open(out_fname, 'wt') as fout:
        # for from_to_span_list, out_line, attr_list in offsets_line_list:
        para_seq = 1
        for para_with_attrs in pdftxt_doc.nlp_paras_with_attrs:
            from_to_lnpos_list, para_attrs = para_with_attrs

            if len(from_to_lnpos_list) == 1:
                from_lnpos, to_lnpos = from_to_lnpos_list[0]
                if from_lnpos.start == from_lnpos.end:
                    # an empty line, this is a paragraph break
                    continue

            print('\n----- para_with_attr #{}'.format(para_seq), file=fout)
            for unused_from_lnpos, to_lnpos in from_to_lnpos_list:
                ln_text = nlp_text[to_lnpos.start:to_lnpos.end]
                if ln_text:
                    print("ln_text: [{}]".format(ln_text), file=fout)
            print('    {}\t{}'.format(from_to_lnpos_list, para_attrs), file=fout)
            para_seq += 1

        print('wrote {}'.format(out_fname), file=sys.stderr)


def is_title_page(apage: PageInfo3) -> bool:
    """Determine is a page is a title page, only based on apage.content_linex_list."""
    content_lines = apage.content_linex_list
    if not content_lines:
        return False
    num_lines_in_page = len(content_lines)
    num_centered_line = 0
    num_short_line = 0
    num_not_english = 0

    if apage.page_num > 3:
        return False
    if num_lines_in_page >= 20:
        return False

    for linex in content_lines:
        if linex.is_centered:
            num_centered_line += 1
        if linex.num_word <= 6:
            num_short_line += 1
        if not linex.is_english:
            num_not_english += 1

    # print("num_lines_in_page: %d" % num_lines_in_page)
    # print("num_centered_line: %d" % num_centered_line)
    # print("num_short_line: %d" % num_short_line)
    # print("num_not_english: %d" % num_not_english)

    if num_centered_line / num_lines_in_page >= 0.8:
        return True
    if num_short_line / num_lines_in_page >= 0.8 or \
       num_not_english / num_lines_in_page >= 0.8:
        return True
    return False

"""
def linex_list_to_pblock_info(lineinfo_list: List[LineInfo3],
                              block_id: int,
                              page_num: int,
                              nl_text: str) -> PBlockInfo:
    block_start = lineinfo_list[0].start
    block_end = lineinfo_list[-1].end
    paraline_chunk_text = nl_text[block_start:block_end]

    unused_para_line, is_multi_lines, unused_not_linebreaks = \
        pdfutils.para_to_para_list(paraline_chunk_text)
    # print('is_multi_lines = {}'.format(is_multi_lines))

    # print("is_multi_lines = {}, paraline: [{}]\n".format(is_multi_lines,
    #                                                      para_line))
    # print("\nblock_chunk_text: [{}] is_multi={}".format(paraline_chunk_text,
    #                                                     is_multi_lines))
    if not is_multi_lines:
        paraline_chunk_text = paraline_chunk_text.replace('\n', ' ')

    # print('page: {}, block {}'.format(page_num, doc_block_id))
    # print(paraline_chunk_text)
    # print()

    # for lineinfo in lineinfo_list:
    #     lineinfo.ybid = doc_block_id

    # print("is_multi_lines = {}, paraline: [{}]\n".format(is_multi_lines, para_line))
    block_info = PBlockInfo(block_start,
                            block_end,
                            block_id,
                            page_num,
                            paraline_chunk_text,
                            # bxid_lineinfos_map[pblock_id],
                            lineinfo_list,
                            is_multi_lines)
    return block_info
"""


def linex_list_to_multi_line_pblock_info(linex_list: List[LineWithAttrs],
                                         block_id: int,
                                         page_num: int,
                                         doc_text: str) -> PBlockInfo:
    block_start = linex_list[0].lineinfo.start
    block_end = linex_list[-1].lineinfo.end
    is_multi_lines = True

    # assume the line breaks are correctly done here
    paraline_chunk_text = doc_text[block_start:block_end]

    # print('page: {}, block {}'.format(page_num, doc_block_id))
    # print(paraline_chunk_text)
    # print()

    lineinfo_list = []  # type: List[LineInfo3]
    for linex in linex_list:
        lineinfo_list.append(linex.lineinfo)

    # print("is_multi_lines = {}, paraline: [{}]\n".format(is_multi_lines, para_line))
    block_info = PBlockInfo(block_start,
                            block_end,
                            block_id,
                            page_num,
                            paraline_chunk_text,
                            # bxid_lineinfos_map[pblock_id],
                            lineinfo_list,
                            is_multi_lines)
    return block_info



def adjust_title_page_blocks(apage: PageInfo3,
                             doc_text: str) -> None:
    """After found that this page is a title page, use original PDFBOx's
       block id instead of Kirke's version based on ydiff.

    This applies to all the lines in a page, not just the content_line_list.
    """

    for linex in apage.line_list:
        linex.block_num = linex.lineinfo.obid

    # first set all block_num of the lines in this page to pdfbox's obid
    # obid_lines_map = defaultdict(list)  # type: DefaultDict[int, List[LineWithAttrs]]
    # obid_list = []  # type: List[int]
    # for linex in apage.line_list:
    #     linex.block_num = linex.lineinfo.obid
    #     obid_list.append(linex.lineinfo.obid)
    #     obid_lines_map[linex.lineinfo.obid].append(linex)
    #
    # There is no point adjusting apage's pblock_list
    # because it is NO LONGER used by the rest of the system.
    # pblock_list = []
    # for obid in obid_list:
    #     pblock = linex_list_to_multi_line_pblock_info(obid_lines_map[obid],
    #                                                   obid,
    #                                                   apage.page_num,
    #                                                   doc_text)
    #     pblock_list.append(pblock)
