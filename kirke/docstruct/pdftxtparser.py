import os
import re
import sys
from collections import defaultdict

from kirke.utils import strutils, txtreader
from kirke.docstruct.pblockinfo import PBlockInfo, GroupedBlockInfo
from kirke.docstruct.pdfoffsets import StrInfo, LineInfo3, PageInfo, PageInfo3, PDFTextDoc
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



def parse_document(file_name, work_dir, debug_mode=True):
    base_fname = os.path.basename(file_name)
    offsets_file_name = pdfutils.get_offsets_file_name(file_name)

    doc_text = strutils.loads(file_name)
    doc_len, str_offsets, line_breaks, pblock_offsets, page_offsets = \
        pdfutils.load_pdf_offsets(offsets_file_name)
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

    # xxx
    add_doc_structure_to_doc(pdf_text_doc)

    return pdf_text_doc

def merge_if_continue_to_next_page(prev_page, cur_page):
    if not prev_page.content_line_list or not cur_page.content_line_list:
        return
    last_line = prev_page.content_line_list[-1]
    words = last_line.line_text.split()
    last_line_block_num = last_line.lineinfo.block_num
    last_line_align = last_line.align

    first_line = cur_page.content_line_list[0]
    first_line_align = first_line.align

    # if the last line is not even toward the lower portion of the page, don't
    # bother merging
    if last_line.lineinfo.yStart < 600:
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
            first_line_block_num = first_line.lineinfo.block_num
            for linex in cur_page.content_line_list:
                if linex.lineinfo.block_num == first_line_block_num:
                    linex.lineinfo.block_num = last_line_block_num
                else:
                    break

def add_doc_structure_to_doc(pdftxt_doc):
    # first remove obvious non-content lines, such
    # toc, page-num, header, footer
    # Also add section heads
    for page in pdftxt_doc.page_list:
        add_doc_structure_to_page(page)

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
            block_num = linex.lineinfo.block_num
            block_list_map[block_num].append(linex)

    # the block list is for the document, not a page
    paged_grouped_block_list = defaultdict(list)
    for block_num, line_list in sorted(block_list_map.items()):
       page_num = line_list[0].page_num
       paged_grouped_block_list[page_num].append(GroupedBlockInfo(page_num, block_num, line_list))
       # pdftxt_doc.grouped_block_list.append(line_list)

    for page_num in range(1, pdftxt_doc.num_pages + 1):
        grouped_block_list = paged_grouped_block_list[page_num]
        grouped_block_list = infer_block_type(grouped_block_list, page_num)
        pdftxt_doc.paged_grouped_block_list.append(grouped_block_list)


def add_doc_structure_to_page(apage):
    num_line_in_page = len(apage.line_list)
    prev_line_text = ''
    # take out lines that are clearly not useful for annotation extractions:
    #   - toc
    #   - page_num
    #   - header, footer
    content_line_list = []
    for line_num, line in enumerate(apage.line_list, 1):
        is_skip = False
        if docstructutils.is_line_toc(line.line_text):
            line.attrs['toc'] = True
            is_skip = True
            apage.attrs['has_toc'] = True
        elif docstructutils.is_line_page_num(line.line_text, line.is_centered):
            line.attrs['page_num'] = True
            # so we can detect footers after page_num, 1-based
            apage.attrs['page_num_index'] = line_num
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
    apage.content_line_list = tmp_list

SIGNATURE_PREFIX_PAT = re.compile(r'(By|Name|Title)\s*:')

def infer_block_type(grouped_block_list, page_num):
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
        prev_line = None
        # merge blocks with one signature lines only, not across multiple-line
        # blocks.
        block_line_list_map = defaultdict(list)
        for grouped_block in grouped_block_list:
            block_num = grouped_block.bid
            is_signature_line = grouped_block.attrs.get('is_signature')
            if is_signature_line and len(grouped_block.line_list) == 1:
                linex = grouped_block.line_list[0]
                if is_prev_block_signature:
                    linex.lineinfo.block_num = prev_signature_block_num
                    block_line_list_map[prev_signature_block_num].append(linex)
                else:
                    block_line_list_map[linex.lineinfo.block_num].append(linex)
                    is_prev_block_signature = True
                    prev_signature_block_num = linex.lineinfo.block_num
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
        is_prv_maybe_signature = False
        for linex_list in tmp_list:
            block_has_signature = False
            block_num = linex_list[0].lineinfo.block_num
            for linex in linex_list:
                if linex.attrs.get('is_signature'):
                    block_has_signature = True

            if block_has_signature and is_prev_maybe_signature:
                for linex in linex_list:
                    block_line_list_map[prev_block_num].append(linex)
                # now set all line in this block signature
                for linex in block_line_list_map[prev_block_num]:
                    linex.attrs['is_signature'] = True
            else:
                for linex in linex_list:
                    block_line_list_map[block_num].append(linex)

            is_prev_maybe_signature = False
            if len(linex_list) <= 3 and linex_list[0].is_english == False:
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
    
