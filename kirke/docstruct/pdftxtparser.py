import os
import sys
from collections import defaultdict

from kirke.utils import strutils, txtreader
from kirke.docstruct.pblockinfo import PBlockInfo
from kirke.docstruct.pdfoffsets import StrInfo, LineInfo3, PageInfo, PageInfo3
from kirke.docstruct import pdfutils

class EbPage:

    def __init__(self, start, end, xStart, xEnd, yStart):
        self.start = start
        self.end = end    
        self.page_num = page_num
        self.attrs = {}        
        

class EbBlock:

    def __init__(self, start, end, xStart, xEnd, yStart):
        self.start = start
        self.end = end
        self.xStart = xStart
        self.yStart = yStart
        self.page_num = page_num
        self.is_multi_line = xxx
        self.lines = lines
        self.attrs = {}


class EbLine:
    
    def __init__(self, start, end, xStart, xEnd, yStart):
        self.start = start
        self.end = end
        self.xStart = xStart
        self.xEnd = xEnd
        self.yStart = yStart
        self.attrs = {}
        
class PDFTextDoc:

    def __init__(self, doc_text, page_list):
        self.doc_text = doc_text
        self.page_list = page_list
        self.nlp_block_list = []

    def print_debug_lines(self):
        paged_fname = 'dir-work/paged-line.txt'
        for page in self.page_list:
            print('===== page #{}, start={}, end={}\n'.format(page.page_num,
                                                              page.start,
                                                              page.end))
            print('page_line_list.len= {}'.format(len(page.line_list)))

            for linex in page.line_list:
                print('{}\t{}'.format(linex.tostr2(), self.doc_text[linex.lineinfo.start:linex.lineinfo.end]))

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
            bxid_lineinfos_map[block_num].append(LineInfo3(start, end, line_num, block_num, lxid_strinfos_map[line_num]))
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
            print("xxxx len(linex_list)= {}".format(len(linex_list)))
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
        print("doing Pageinfo3, page_num = {}".format(page_num))
        pinfo = PageInfo3(doc_text, start, end, page_num, pblockinfo_list)
        pageinfo_list.append(pinfo)

    pdf_text_doc = PDFTextDoc(doc_text, pageinfo_list)

    return pdf_text_doc


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
                print('orig=({}, {}), nlp=({}, {}), ydiff= {}, xStart= {}, xEnd= {}, yStart= {}, yEnd= {}, linebreak= {}'.format(line4nlp.orig_start, line4nlp.orig_end,
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
    
