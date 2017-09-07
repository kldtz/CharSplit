import os
import re
import sys
from collections import defaultdict
from typing import List

from kirke.utils import strutils, txtreader, mathutils
from kirke.docstruct import pdfoffsets
from kirke.docstruct.pdfoffsets import StrInfo, LineInfo3, PageInfo, PageInfo3, PDFTextDoc, LineWithAttrs, PBlockInfo, GroupedBlockInfo
from kirke.docstruct import pdfutils, docstructutils


def get_nl_fname(base_fname, work_dir):
    return '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nl.txt'))


def get_paraline_fname(base_fname, work_dir):
    return '{}/{}'.format(work_dir, base_fname.replace('.txt', '.paraline.txt'))


def text_offsets_to_nl(base_fname, orig_doc_text, line_breaks, work_dir, debug_mode=False):
    linebreak_offset_list = [lbrk['offset'] for lbrk in line_breaks]
    ch_list = list(orig_doc_text)
    for linebreak_offset in linebreak_offset_list:
        ch_list[linebreak_offset] = '\n'
    nl_text = ''.join(ch_list)
    nl_fname = get_nl_fname(base_fname, work_dir)
    txtreader.dumps(nl_text, nl_fname)
    if debug_mode:
        print('wrote {}, size= {}'.format(nl_fname, len(nl_text)), file=sys.stderr)
    return nl_text, nl_fname


def to_nl_paraline_texts(file_name, offsets_file_name, work_dir, debug_mode=True):
    base_fname = os.path.basename(file_name)

    if debug_mode:
        print('reading text doc: [{}]'.format(file_name), file=sys.stderr)
    orig_doc_text = strutils.loads(file_name)
    doc_len, str_offsets, line_breaks, pblock_offsets, page_offsets = pdfutils.load_pdf_offsets(offsets_file_name)
    # print('doc_len = {}, another {}'.format(doc_len, len(doc_text)))

    nl_text, nl_fname = text_offsets_to_nl(base_fname, orig_doc_text, line_breaks,
                                           work_dir=work_dir, debug_mode=debug_mode)

    lxid_strinfos_map = defaultdict(list)
    for str_offset in str_offsets:
        start = str_offset['start']
        end = str_offset['end']
        # page_num = str_offset['pageNum']
        line_num = str_offset['lineNum']
        xStart = str_offset['xStart']
        xEnd = str_offset['xEnd']
        yStart = str_offset['yStart']
        lxid_strinfos_map[line_num].append(StrInfo(start, end,
                                                   xStart, xEnd, yStart))

    bxid_lineinfos_map = defaultdict(list)
    tmp_prev_end = 0
    for i, break_offset in enumerate(line_breaks):
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

    pgid_pblockinfos_map = defaultdict(list)
    block_info_list = []
    for pblock_offset in pblock_offsets:
        pblock_id = pblock_offset['id']
        start = pblock_offset['start']
        end = pblock_offset['end']
        page_num = pblock_offset['pageNum']

        while start <= end - 1 and strutils.is_nl(nl_text[end -1]):
            end -= 1

        if start != end:
            para_line, is_multi_lines = pdfutils.para_to_para_list(nl_text[start:end])
            # print("para_line=======================")
            # print(para_line)
            # print("para_line==================end==")
            # xStart, xEnd, yStart are initizlied in here
            block_info = PBlockInfo(start,
                                    end,
                                    pblock_id,
                                    page_num,
                                    para_line,
                                    bxid_lineinfos_map[pblock_id],
                                    is_multi_lines)
            pgid_pblockinfos_map[page_num].append(block_info)
            block_info_list.append(block_info)

    # the code below DOESN'T ALWAYS produces the .paraline.txt with same size as
    # .txt.  This is due to block offset overlaps in "\s\n\n" => "\s\n\s\s",
    # block (start 0, end 1) followed by block (start 1, end 42).  Normally
    # 2nd block start doesn't overlap with first block end.  2 out of 30 has this
    # issue.
    """
    paraline_list = []
    prev_end_offset = 0
    for block_info in block_info_list:
        diff_prev_end_offset = block_info.start - prev_end_offset
        # output eoln
        for i in range(max(diff_prev_end_offset - 1, 0)):
            paraline_list.append('')
        # print('df= {}'.format(diff_prev_end_offset), file=fout)
        # print('', file=fout)
        # block_text = doc_text[block_info.start:block_info.end]
        block_text = block_info.text  # because of pblock might have multiple paragraphs; sad.
        if block_info.is_multi_lines:
            paraline_list.append(block_text)
        else:
            paraline_list.append(block_text.replace('\n', ' '))
        # print('', file=fout)
        prev_end_offset = block_info.end
    for i in range(doc_len - block_info.end):
        paraline_list.append('')
    paraline_text = '\n'.join(paraline_list)
    """
    # Now, switch to array replacement.  This is not affected by the wrong block info.
    # It simply override everys block based on the indexes, so guarantees not to create
    # extra stuff.
    ch_list = list(nl_text)
    for block_info in block_info_list:
        block_text = block_info.text  # because of pblock might have multiple paragraphs; sad.
        if block_info.is_multi_lines:
            ch_list[block_info.start:block_info.end] = list(block_text)
        else:
            ch_list[block_info.start:block_info.end] = list(block_text.replace('\n', ' '))
    paraline_text = ''.join(ch_list)

    # save the result
    paraline_fn = get_paraline_fname(base_fname, work_dir)
    txtreader.dumps(paraline_text, paraline_fn)
    if debug_mode:
        print('wrote {}, size= {}'.format(paraline_fn, len(paraline_text)), file=sys.stderr)

    return orig_doc_text, nl_text, paraline_text, nl_fname, paraline_fn

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


def to_paras_with_attrs(pdf_text_doc, file_name, work_dir):
    base_fname = os.path.basename(file_name)

    cur_attr = []
    gap_span_list = []
    omit_line_set = []
    offset = 0
    out_line_list = []
    offsets_line_list = []
    with open('ashxx.check.offsets.txt', 'wt') as fout1, open('ashxx.with.offsets.txt', 'wt') as fout2:

        for page_num, grouped_block_list in enumerate(pdf_text_doc.paged_grouped_block_list, 1):
            apage = pdf_text_doc.page_list[page_num - 1]
            attr_list = sorted(apage.attrs.items())
            #if attr_list:
            #    print('  attrs: {}'.format(', '.join(attr_list)))
            for grouped_block in grouped_block_list:

                is_multi_line = is_block_multi_line(grouped_block.line_list)

                if is_multi_line:
                    for linex in grouped_block.line_list:
                        out_line = pdf_text_doc.doc_text[linex.lineinfo.start:linex.lineinfo.end]
                        attr_list = linex.to_attrvals()  # sorted(linex.attrs.items())

                        out_line_list.append(out_line)
                        offsets_line_list.append(((linex.lineinfo.start, linex.lineinfo.end),
                                                  (offset, offset + len(out_line)),
                                                  out_line, attr_list))
                        offset += len(out_line) + 1  # to add eoln
                else:
                    block_lines = []
                    attr_list = grouped_block.line_list[0].to_attrvals()  # sorted(linex.attrs.items())
                    block_start = offset
                    for linex in grouped_block.line_list:
                        out_line = pdf_text_doc.doc_text[linex.lineinfo.start:linex.lineinfo.end]
                        block_lines.append(out_line)
                        offset += len(out_line) + 1  # to add eoln

                    block_text = ' '.join(block_lines)
                    out_line_list.append(block_text)
                    offsets_line_list.append(((grouped_block.line_list[0].lineinfo.start, grouped_block.line_list[-1].lineinfo.end),
                                              (block_start, offset - 1),
                                              block_text, attr_list))

                out_line_list.append('')
                offsets_line_list.append(((linex.lineinfo.end+2, linex.lineinfo.end+2),
                                          (offset, offset),
                                          '', {}))
                offset += 1

        for out_line in out_line_list:
            print(out_line, file=fout1)
        for x, y, out_line, attr_list in offsets_line_list:
            print('{}, {}\t{}\t[{}]'.format(x, y, sorted(attr_list.items()), out_line), file=fout2)

        print('wrote {}'.format('ashxx.check.offsets.txt'))off
        print('wrote {}'.format('ashxx.with.offsets.txt'))

    # return lineinfos_paras, paras_doc_text, gap_span_list


def parse_document(file_name, work_dir, debug_mode=True):
    base_fname = os.path.basename(file_name)

    doc_text = strutils.loads(file_name)
    doc_len, str_offsets, line_breaks, pblock_offsets, page_offsets = \
        pdfutils.load_pdf_offsets(pdfutils.get_offsets_file_name(file_name))
    # print('doc_len = {}, another {}'.format(doc_len, len(doc_text)))

    nl_text, nl_fname = text_offsets_to_nl(base_fname, doc_text, line_breaks,
                                           work_dir=work_dir, debug_mode=debug_mode)
    
#    if debug_mode:
#        save_debug_txt_files(work_dir, base_fname, nl_text,
#                             linebreak_offset_list,
#                             doc_len, str_offsets, line_breaks, pblock_offsets, page_offsets)

    lxid_strinfos_map = defaultdict(list)
    for str_offset in str_offsets:
        start = str_offset['start']
        end = str_offset['end']
        # page_num = str_offset['pageNum']
        line_num = str_offset['lineNum']
        xStart = str_offset['xStart']
        xEnd = str_offset['xEnd']
        yStart = str_offset['yStart']
        lxid_strinfos_map[line_num].append(StrInfo(start, end,
                                                   xStart, xEnd, yStart))

    bxid_lineinfos_map = defaultdict(list)
    tmp_prev_end = 0
    for i, break_offset in enumerate(line_breaks):
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

    pgid_pblockinfos_map = defaultdict(list)
    block_info_list = []
    for pblock_offset in pblock_offsets:
        pblock_id = pblock_offset['id']
        start = pblock_offset['start']
        end = pblock_offset['end']
        page_num = pblock_offset['pageNum']

        while start <= end - 1 and strutils.is_nl(nl_text[end -1]):
            end -= 1

        if start != end:
            para_line, is_multi_lines = pdfutils.para_to_para_list(nl_text[start:end])

            linex_list = bxid_lineinfos_map[pblock_id]
            # xStart, xEnd, yStart are initizlied in here
            block_info = PBlockInfo(start,
                                    end,
                                    pblock_id,
                                    page_num,
                                    para_line,
                                    bxid_lineinfos_map[pblock_id],
                                    is_multi_lines)
            pgid_pblockinfos_map[page_num].append(block_info)
            block_info_list.append(block_info)

    pageinfo_list = []
    nlp_offset = 0
    for page_offset in page_offsets:
        start = page_offset['start']
        end = page_offset['end']
        page_num = page_offset['id']
        pblockinfo_list = pgid_pblockinfos_map[page_num]
        pinfo = PageInfo3(doc_text, start, end, page_num, pblockinfo_list)
        pageinfo_list.append(pinfo)

    pdf_text_doc = PDFTextDoc(doc_text, pageinfo_list)

    add_doc_structure_to_doc(pdf_text_doc)

    return pdf_text_doc

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
    if (last_line_align != first_line_align and last_line_align[:2] == first_line_align[:2] and
        # or any type of sechead prefix
        # TODO, jshaw, implement a better prefix detection in secheadutil
        first_line.line_text[0] == '('):
        return

    # dont' join sechead or anything that's centered
    if first_line.is_centered or first_line.attrs.get('sechead'):
        return


    # 8 because average word per sentence is known to be around 7
    if len(words) >= 8 and (words[-1][-1].islower() or strutils.is_not_sent_punct(words[-1][-1])):
        if not first_line.attrs.get('sechead'):
            first_line_block_num = first_line.block_num
            for linex in cur_page.content_line_list:
                if linex.block_num == first_line_block_num:
                    linex.block_num = last_line_block_num
                else:
                    break

# this should be moved to pdfutils
def lines_to_block_offsets(linex_list: List[LineWithAttrs], block_type: str, pagenum: int):
    if linex_list:
        start = linex_list[0].lineinfo.start
        end = linex_list[-1].lineinfo.end
        return (start, end, {'block-type': block_type, 'pagenum': pagenum})
    # why would this happen?
    return (0, 0, {'block-type': block_type})

def line_to_block_offsets(linex, block_type: str, pagenum: int):
    start = linex.lineinfo.start
    end = linex.lineinfo.end
    return (start, end, {'block-type': block_type, 'pagenum': pagenum})

def lines_to_blocknum_map(linex_list):
    result = defaultdict(list)
    for linex in linex_list:
        block_num = linex.block_num
        result[block_num].append(linex)
    return result


def add_doc_structure_to_doc(pdftxt_doc):
    # first remove obvious non-content lines, such
    # toc, page-num, header, footer
    # Also add section heads
    # page_attrs_list is to store table information?
    for page in pdftxt_doc.page_list:
        add_doc_structure_to_page(page, pdftxt_doc)

    # merge paragraphs that are across pages
    prev_page = pdftxt_doc.page_list[0]
    for apage in pdftxt_doc.page_list[1:]:
        merge_if_continue_to_next_page(prev_page, apage)
        prev_page = apage

    # categorize blocks, such as signature, address
    # xxx
    block_list_map = defaultdict(list)
    for apage in pdftxt_doc.page_list:
        for linex in apage.content_line_list:
            block_num = linex.block_num
            block_list_map[block_num].append(linex)

    # the block list is for the document, not a page
    paged_grouped_block_list = defaultdict(list)
    for block_num, line_list in sorted(block_list_map.items()):
       page_num = line_list[0].page_num
       paged_grouped_block_list[page_num].append(GroupedBlockInfo(page_num, block_num, line_list))
       # pdftxt_doc.grouped_block_list.append(line_list)

    for page_num in range(1, pdftxt_doc.num_pages + 1):
        grouped_block_list = paged_grouped_block_list[page_num]
        pdftxt_doc.paged_grouped_block_list.append(grouped_block_list)


def add_doc_structure_to_page(apage, pdf_txt_doc):
    num_line_in_page = len(apage.line_list)
    page_num = apage.page_num
    prev_line_text = ''
    # take out lines that are clearly not useful for annotation extractions:
    #   - toc
    #   - page_num
    #   - header, footer
    content_line_list = []
    toc_block_list = []
    for line_num, line in enumerate(apage.line_list, 1):
        is_skip = False
        if docstructutils.is_line_toc(line.line_text):
            line.attrs['toc'] = True
            is_skip = True
            apage.attrs['has_toc'] = True
            toc_block_list.append(line)
        elif docstructutils.is_line_page_num(line.line_text, line_num, num_line_in_page, line.linebreak, line.lineinfo.yStart, line.is_centered):
            line.attrs['page_num'] = True
            # so we can detect footers after page_num, 1-based
            apage.attrs['page_num_index'] = line_num
            pdf_txt_doc.special_blocks_map['pagenum'].append(line_to_block_offsets(line, 'pagenum', page_num))
            is_skip = True
        elif docstructutils.is_line_header(line.line_text,
                                           line.lineinfo.yStart,
                                           line_num,
                                           line.is_english,
                                           num_line_in_page):
            line.attrs['header'] = True
            pdf_txt_doc.special_blocks_map['header'].append(line_to_block_offsets(line, 'header', page_num))
            is_skip = True

        # 2nd stage of rules
        is_footer, score = docstructutils.is_line_footer(line.line_text,
                                                         line_num,
                                                         num_line_in_page,
                                                         line.linebreak,
                                                         apage.attrs.get('page_num_index', -1),  # 1-based
                                                         line.is_english,
                                                         line.align,
                                                         line.lineinfo.yStart)
        # if score != -1.0:
        #    print("        is_footer = {}\t{}".format(line.tostr2(), line.line_text))
        if is_footer:
            line.attrs['footer'] = True
            is_skip = True
            pdf_txt_doc.special_blocks_map['footer'].append(line_to_block_offsets(line, 'footer', page_num))

        prev_line_text = line.line_text

        if not is_skip:
            content_line_list.append(line)


    apage.content_line_list = content_line_list

    # remove secheads after toc section
    # sechead info is not available until now.  Cannot remove
    # those earlier.
    deactivate_toc_detection = False
    non_sechead_count = 0
    for line_num, line in enumerate(apage.content_line_list, 1):
        # sechead detection is applied later
        # sechead, prefix, head, split_idx
        sechead_tuple = docstructutils.extract_line_sechead(line.line_text, prev_line_text)
        if sechead_tuple:
            if apage.attrs.get('has_toc') and not deactivate_toc_detection:
                line.attrs['toc'] = True
            line.attrs['sechead'] = sechead_tuple
        else:
            non_sechead_count += 1

        if non_sechead_count >= 3:
            deactivate_toc_detection = True

    # now remove potential newly added toc lines
    tmp_list = []
    for linex in apage.content_line_list:
        if not linex.attrs.get('toc'):
            tmp_list.append(linex)
        else:
            toc_block_list.append(linex)

    if toc_block_list:
        pdf_txt_doc.special_blocks_map['toc'].append(lines_to_block_offsets(toc_block_list, 'toc', page_num))
    # TODO, jshaw, take the begin and end of TOC
    # take any tmp_list that's falls inside TOC and drop them
    # from regular text.  Probably TOC recognition errors.
    # NOT DONE YET.
    apage.content_line_list = tmp_list

    grouped_block_list = pdfoffsets.line_list_to_grouped_block_list(apage.content_line_list, page_num)

    apage.grouped_block_list = markup_signature_block(grouped_block_list, page_num)
    for grouped_block in apage.grouped_block_list:
        # GroupedBlockInfo(page_num, block_num, linex_list))
        if grouped_block.line_list and grouped_block.line_list[0].attrs.get('is_signature'):
            pdf_txt_doc.special_blocks_map['signature'].append(lines_to_block_offsets(grouped_block.line_list, 'signature', page_num))

    table_with_number_list = markup_table_block2(grouped_block_list, page_num)
    print("table_with_number_list: {}, page_num = {}".format(table_with_number_list, page_num))
    if table_with_number_list:
        blocknum_lines_map = lines_to_blocknum_map(apage.content_line_list)
        if apage.attrs.get('table_blockset_list'):
            apage.attrs['table_blockset_list'].extends(table_with_number_list)
        else:
            apage.attrs['table_blockset_list'] = table_with_number_list

        print("page #{}, table_blockset_list = {}".format(page_num, apage.attrs['table_blockset_list']))
        for table_blockset in table_with_number_list:
            table_lines = []
            for blocknum in sorted(table_blockset):
                for linex in blocknum_lines_map[blocknum]:
                    table_lines.append(linex)
            print("blocknum {}, len(lines) = {}".format(blocknum, len(blocknum_lines_map[blocknum])))
            pdf_txt_doc.special_blocks_map['table'].append(lines_to_block_offsets(table_lines, 'table', page_num))

    # handle table and chart identification
    grouped_block_list = apage.grouped_block_list
    table_blockset_list, chart_blockset_list = markup_table_block(grouped_block_list, page_num)

    blocknum_lines_map = lines_to_blocknum_map(apage.content_line_list)
    if table_blockset_list:
        if apage.attrs.get('table_blockset_list'):
            apage.attrs['table_blockset_list'].extends(table_with_number_list)
        else:
            apage.attrs['table_blockset_list'] = table_with_number_list

        # pdf_txt_doc.special_blocks_map['table'].append(lines_to_block_offsets(table_block_list))
        print("page #{}, table_blockset_list = {}".format(page_num, apage.attrs['table_blockset_list']))
        for table_blockset in table_blockset_list:
            table_lines = []
            for blocknum in sorted(table_blockset):
                for linex in blocknum_lines_map[blocknum]:
                    table_lines.append(linex)
            print("blocknum {}, len(lines) = {}".format(blocknum, len(blocknum_lines_map[blocknum])))
            pdf_txt_doc.special_blocks_map['table'].append(lines_to_block_offsets(table_lines, 'table', page_num))
    if chart_blockset_list:
        apage.attrs['chart_blockset_list'] = chart_blockset_list
        print("page #{}, chart_blockset_list = {}".format(page_num, apage.attrs['chart_blockset_list']))
        for table_blockset in chart_blockset_list:
            table_lines = []
            for blocknum in sorted(table_blockset):
                for linex in blocknum_lines_map[blocknum]:
                    table_lines.append(linex)
            pdf_txt_doc.special_blocks_map['chart'].append(lines_to_block_offsets(table_lines, 'chart', page_num))

def markup_table_block2(grouped_block_list, page_num):
    maybe_table_with_numbers = []

    # first try to see if tables should be formed from separated blocks
    num_table_one_line_block = 0
    num_line = 0
    for grouped_block in grouped_block_list:
        num_line_in_block = len(grouped_block.line_list)
        if num_line_in_block == 1:
            linex = grouped_block.line_list[0]
            if len(linex.line_text) < 30 and not linex.is_english and linex.is_centered:
                num_table_one_line_block += 1
        num_line += num_line_in_block

    if num_line > 7 and num_table_one_line_block >= 3:
        print('markup_table_block2, page = {}'.format(page_num))
        print("should collapse......................")

        # for debug purpose only
        print("for debug purpose:  35234")
        for grouped_block in grouped_block_list:
            print()
            for linex in grouped_block.line_list:
                print("    linex: {}\t[{}...]".format(linex.tostr5(), linex.line_text[:15]))

        # find the most common
        line_aligned_map = defaultdict(list)
        linex_list = []
        for grouped_block in grouped_block_list:
            for linex in grouped_block.line_list:
                linex_list.append(linex)
                prefix = 'align={}, cn={}, en={}'.format(linex.align, linex.is_centered, linex.is_english)
                line_aligned_map[prefix].append(linex)
        freq_align_list = sorted([(len(alist), prefix, alist) for prefix, alist in line_aligned_map.items()], reverse=True)
        lenx, prefix, maybe_collapse_lines = freq_align_list[0]
        #for lenx, prefix, alist in freq_align_list:
        #    print("lenx= {}, prefix = {}".format(lenx, prefix))
        #    for linex in alist:
        #        print("      linex: {}".format(linex.tostr3()))
        if lenx >= 3:  # as long as this is more than 3
            maybe_collapse_lines = sorted(maybe_collapse_lines)
            linex0 = maybe_collapse_lines[0]
            bnum0 = linex0.block_num
            # linex_size = len(maybe_collapse_lines)
            prev_bnum = bnum0
            is_collpased = False
            # we assume there is only 1 such table in a page
            is_collapsed = False
            for i, linex in enumerate(maybe_collapse_lines[1:]):
                cur_block_num = linex.block_num
                if linex.block_num == prev_bnum:  # already in the same block
                    break
                elif linex.block_num - prev_bnum <= 2:
                    linex.block_num = bnum0
                    is_collapsed = True
                else:
                    break  # stop once we cannot find 3 consecutive diff
                prev_bnum = cur_block_num

            if is_collapsed:
                print("collapsed......................")
            else:
                print("NOT collapsed......................")
            grouped_block_list = pdfoffsets.line_list_to_grouped_block_list(linex_list, page_num)

    for grouped_block in grouped_block_list:
        num_line_in_block = len(grouped_block.line_list)
        num_line_with_number = 0
        for linex in grouped_block.line_list:
            if strutils.find_number(linex.line_text) and len(linex.line_text) < 30 and not linex.is_english:
                num_line_with_number += 1
        # print('num_line_in_block = {}, num_line_with_number = {}'.format(num_line_in_block, num_line_with_number))
        if num_line_in_block >= 3 and 1.0 * num_line_with_number / num_line_in_block > 0.7:
            maybe_table_with_numbers.append(grouped_block)

    table_blockset_list = []
    if maybe_table_with_numbers:
        print("mabye_table_with_number, pagenum = {}".format(page_num))
        y_lines = []
        for grouped_block in grouped_block_list:
            for linex in grouped_block.line_list:
                y_lines.append((linex.lineinfo.yStart, linex))
                # print("len(linex.lineinfo.str_list) = {}".format(len(linex.lineinfo.strinfo_list)))
                # print("linex.lineinfo.str_list = {}".format(linex.lineinfo.strinfo_list))

        table_prefix = 'table'
        table_count = 100
        for grouped_block in maybe_table_with_numbers:
            table_name = '{}-p{}-{}'.format(table_prefix, page_num, table_count)
            for linex in grouped_block.line_list:
                linex.attrs[table_prefix] = table_name

            block_first_linex = grouped_block.line_list[0]
            # merge blocks as we see fit
            if block_first_linex:
                merge_centered_lines_before_table(block_first_linex.lineinfo.line_num,
                                                  block_first_linex.block_num,
                                                  y_lines,
                                                  table_prefix,
                                                  table_name)
            else:
                print("block_first_line is empty, page {}".format(page_num))

            table_count += 1
            table_blockset_list.append({block_first_linex.block_num,})
    return table_blockset_list


def markup_table_block(grouped_block_list, page_num):
    has_close_ydiffs = False
    blocks_with_similar_ys = set([])

    y_lines = []
    block_lines_map = defaultdict(list)
    for grouped_block in grouped_block_list:
        for linex in grouped_block.line_list:
            y_lines.append((linex.lineinfo.yStart, linex))
            block_lines_map[linex.block_num].append(linex)
    prev_yStart = -100
    prev_block_num = -1
    for yStart, linex in sorted(y_lines):
        # there is multiple lines in the same yStart
        cur_block_num = linex.block_num
        if yStart - prev_yStart <= 5.0:
            # print("page_num = {}, ydiff {}".format(page_num, yStart - prev_yStart))
            blocks_with_similar_ys.add((prev_block_num, cur_block_num))
        prev_yStart = yStart
        prev_block_num = cur_block_num

    # now, the blocks are in pairs. Change them to groups or lists
    blocks_with_similar_ys = mathutils.pairs_to_sets(blocks_with_similar_ys)

    table_blockset_list, chart_blockset_list = [], []
    table_count = 1
    for block_num_set in blocks_with_similar_ys:
        block_align_set_list = []
        block_first_linex = None
        for block_num in block_num_set:
            block_lines = block_lines_map[block_num]
            align_set = set([])
            for linex in block_lines:
                # print("block {} align = {}".format(block_num, linex.align))
                if not block_first_linex:
                    block_first_linex = linex
                align_set.add(linex.align[:2])
            block_align_set_list.append(align_set)

        is_all_align_size_one = True
        for block_align_set in block_align_set_list:
            if len(block_align_set) != 1:
                is_all_align_size_one = False
                break

        if is_all_align_size_one:
            table_prefix = 'table'
            table_blockset_list.append(block_num_set)
        else:
            table_prefix = 'chart'
            chart_blockset_list.append(block_num_set)

        table_name = '{}-p{}-{}'.format(table_prefix, page_num, table_count)
        for block_num in block_num_set:
            block_lines = block_lines_map[block_num]
            for linex in block_lines:
                linex.attrs[table_prefix] = table_name

        # merge blocks as we see fit
        if block_first_linex:
            merge_centered_lines_before_table(block_first_linex.lineinfo.line_num,
                                              sorted(block_num_set)[0],
                                              y_lines,
                                              table_prefix,
                                              table_name)
        else:
            print("block_first_line is empty, page {}".format(page_num))
        table_count += 1

    return table_blockset_list, chart_blockset_list


def merge_centered_lines_before_table(line_num, block_num, y_lines, table_prefix, table_name):
    table_start_idx = -1
    for line_idx, (_, linex) in enumerate(y_lines):
        if linex.lineinfo.line_num == line_num:
            # print("hello {}".format(linex.tostr3()))
            table_start_idx = line_idx
            break
    if table_start_idx != -1:
        table_start_idx -= 1
        while table_start_idx >= 0:
            _, linex = y_lines[table_start_idx]
            # if not english, merge with the table by block
            if linex.attrs.get('sechead'):
                # we take sechead before a table
                linex.block_num = block_num
                linex.attrs[table_prefix] = table_name
                break
            elif linex.is_centered:
                linex.block_num = block_num
                linex.attrs[table_prefix] = table_name
            elif not linex.is_english:
                linex.block_num = block_num
                linex.attrs[table_prefix] = table_name
            else:
                break
            table_start_idx -= 1



SIGNATURE_PREFIX_PAT = re.compile(r'(By|Name|Title)\s*:')

def markup_signature_block(grouped_block_list, page_num):
    has_signature_line = False
    result = []
    # first merge blocks that are likely to be signature blocks
    for grouped_block in grouped_block_list:
        if len(grouped_block.line_list) == 1:
            mat = SIGNATURE_PREFIX_PAT.match(grouped_block.line_list[0].line_text)
            if mat:
                has_signature_line = True
                grouped_block.line_list[0].attrs['is_signature'] = True
                grouped_block.attrs['is_signature'] = True
        else:
            for linex in grouped_block.line_list:
                mat = SIGNATURE_PREFIX_PAT.match(linex.line_text)
                if mat:
                    has_signature_line = True
                    linex.attrs['is_signature'] = True
                    grouped_block.attrs['is_signature'] = True
        result.append(grouped_block)

    if has_signature_line:
        # merge blocks with one signature lines only, not across multiple-line
        # blocks.
        block_line_list_map = defaultdict(list)
        for grouped_block in grouped_block_list:
            block_num = grouped_block.bid
            is_signature_line = grouped_block.attrs.get('is_signature')
            if is_signature_line and len(grouped_block.line_list) == 1:
                linex = grouped_block.line_list[0]
                if is_prev_block_signature:
                    linex.block_num = prev_signature_block_num
                    block_line_list_map[prev_signature_block_num].append(linex)
                else:
                    block_line_list_map[linex.block_num].append(linex)
                    is_prev_block_signature = True
                    prev_signature_block_num = linex.block_num
            else:
                for linex in grouped_block.line_list:
                    block_line_list_map[block_num].append(linex)
                is_prev_block_signature = False

        tmp_list = []
        for block_num, linex_list in sorted(block_line_list_map.items()):
            tmp_list.append(linex_list)
            # GroupedBlockInfo(page_num, block_num,

        block_line_list_map = defaultdict(list)
        prev_block_num = -1
        is_prev_maybe_signature = False
        for linex_list in tmp_list:
            block_has_signature = False
            block_num = linex_list[0].block_num
            for linex in linex_list:
                if linex.attrs.get('is_signature'):
                    block_has_signature = True

            if block_has_signature and is_prev_maybe_signature:
                for linex in linex_list:
                    block_line_list_map[prev_block_num].append(linex)
                # now set all line in this block signature
                for linex in block_line_list_map[prev_block_num]:
                    linex.attrs['is_signature'] = True
                    linex.block_num = prev_block_num
            else:
                for linex in linex_list:
                    block_line_list_map[block_num].append(linex)

            is_prev_maybe_signature = False
            if len(linex_list) <= 3 and not linex_list[0].is_english:
                is_prev_maybe_signature = True
                prev_block_num = block_num

        result = []
        for block_num, linex_list in sorted(block_line_list_map.items()):
            result.append(GroupedBlockInfo(page_num, block_num, linex_list))

    return result


def save_debug_files(pdf_text_doc, base_fname, work_dir):
    doc_text = pdf_text_doc.doc_text
    paged_para_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.paged_para.txt'))
    with open(paged_para_fn, 'wt') as fout:
        for pageinfo in pdf_text_doc.page_list:
            for pblockinfo in pageinfo.pblockinfo_list:
                # print('[{}]'.format(pblockinfo.text), file=fout)
                print(pblockinfo.text, file=fout)
                print('', file=fout)
        print('wrote {}'.format(paged_para_fn))

    paged_debug_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.paged_debug.txt'))
    with open(paged_debug_fn, 'wt') as fout:
        for pageinfo in pdf_text_doc.page_list:
            for line4nlp in pageinfo.line4nlp_list:
                print('orig=(%d, %d), nlp=(%d, %d), ydiff= %.2f, xStart= %.2f, xEnd= %.2f, yStart= %.2f, yEnd= %.2f, linebreak= %d' %
                      (line4nlp.orig_start, line4nlp.orig_end,
                       line4nlp.nlp_start, line4nlp.nlp_end,
                       line4nlp.ydiff,
                       line4nlp.xStart,
                       line4nlp.xEnd,
                       line4nlp.yStart,
                       line4nlp.yEnd,
                       line4nlp.linebreak),
                      file=fout)
                print('[{}]'.format(doc_text[line4nlp.orig_start:line4nlp.orig_end].replace('\n', ' ')), file=fout)
                print('', file=fout)
        print('wrote {}'.format(paged_debug_fn))
    """    
        # probably not being called
        nlp_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nlpXXX333.txt'))
        with open(nlp_fn, 'wt') as fout:
            for pageinfo in pageinfo_list:
                for line4nlp in pageinfo.line4nlp_list:
                    print(doc_text[line4nlp.orig_start:line4nlp.orig_end].replace('\n', ' '), file=fout)
                    print('', file=fout)
            print('wrote {}'.format(nlp_fn))


        sep_fn = get_paraline_fname(base_fname, work_dir)
        with open(sep_fn, 'wt') as fout:
            prev_end_offset = 0
            for block_info in block_info_list:
                diff_prev_end_offset = block_info.start - prev_end_offset
                # output eoln
                for i in range(max(diff_prev_end_offset - 1, 0)):
                    print('', file=fout)
                # print('df= {}'.format(diff_prev_end_offset), file=fout)
                # print('', file=fout)
                block_text = doc_text[block_info.start:block_info.end]
                if block_info.is_multi_lines:
                    print(block_text, file=fout)
                else:
                    print(block_text.replace('\n', ' '), file=fout)
                # print('', file=fout)
                prev_end_offset = block_info.end
            for i in range(doc_len - block_info.end -1):
                print('', file=fout)
            print('wrote {}'.format(sep_fn))
"""
    
