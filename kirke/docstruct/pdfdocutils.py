
import logging
import os
import sys
from typing import Dict, List, Tuple

from kirke.docstruct.pdfoffsets import PageInfo3, PBlockInfo, PDFTextDoc, LineWithAttrs
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


# @deprecated
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
