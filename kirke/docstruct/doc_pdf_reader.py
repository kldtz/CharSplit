import bisect
from collections import namedtuple, defaultdict
import itertools
import json
import logging
import os
import re
import sys
import time
from typing import List

# from kirke.docstruct.ebdoc import EbSentence, EbPageNumber
from kirke.utils import mathutils, strutils, txtreader

from kirke.docstruct import jenksutils

from kirke.docstruct.sentv2 import SentV2
from kirke.docstruct.lxlineinfo import LineInfo
from kirke.docstruct import lxlineinfo
from kirke.docstruct import sentv2 as lxsentv2
from kirke.docstruct import secheadutils, tablefinder

from kirke.docstruct import secheadutils, spanfinder
from kirke.docstruct import sent4nlp
from kirke.docstruct.doctree import DTreeDoc, DTreeSegment

from kirke.docstruct import tocfinder, parav2finder, footerutils

from kirke.docstruct.pblockinfo import PBlockInfo
from kirke.docstruct.pdfoffsets import StrInfo, LineInfo3, PageInfo
                                       
DEBUG_MODE = False

"""
def print_format_document(text, sentV2or4_list):
    for i, sentV2or4 in enumerate(sentV2or4_list):
        start = sentV2or4.start
        end = sentV2or4.end
        text2 = text[start:end]

        # print('[ {} {} ] {}'.format(sentV2.category, sentV2.align_list, sentV2.text))
        if isinstance(sentV2or4, SentV2):
            print('[{} {} {} ] {}'.format(i, sentV2or4.category, sentV2or4.align_list, text2))
        else:
            print('[{} {} ] {}'.format(i, sentV2or4.category, text2))            
        print()
"""


# we only merge paragraph across page boundaries
def sentV2_list_to_clean_text_v1(doc_text, sentV2_list):

    prev_para_last_char = '.'
    prev_para_sentV2 = None

    isin_page_boundary_mode = False
    for i, sentV2 in enumerate(sentV2_list):
        start = sentV2.start
        end = sentV2.end
        text2 = doc_text[start:end]

        if sentV2.category not in {'footer', 'pagenum'}:
            if isin_page_boundary_mode:
                print('\nprev [ {} {} ] [{}]'.format(prev_para_sentV2.category, prev_para_sentV2.align_list, prev_para_sentV2.text))
                print("xxx last_para_last_char = [{}]".format(prev_para_last_char))
                print("first_char = [{}]".format(text2[0]))
                print('cur [ {} {} ] [{}]'.format(sentV2.category, sentV2.align_list, sentV2.text))

                if not strutils.is_eosent(prev_para_last_char) and strutils.is_lc(text2[0]):
                    print("\n****** start here")
                    print(prev_para_sentV2)
                    print("****** merge here")
                    print(sentV2)
            isin_page_boundary_mode = False

            # sometimes [-1] get spaces because of the way we do sentence boundary in Java
            prev_para_sentV2 = sentV2
            prev_para_last_char = text2[-1] # text2[-2:].strip()[-1]
            if strutils.is_space(prev_para_last_char):
                prev_para_last_char = text2[-2]
        else:
            isin_page_boundary_mode = True

            # print('{}\t{}'.format(i, text2))
            # print()

        # print("last_para_last_char = [{}]".format(prev_para_last_char))



def save_sent_nlp_offsets(file_name, sentV4_list):
    #for sentV2 in sentV4_list:
    #    print("xxxx3534\tsentV2\t[{}]".format(sentV2.text))
    #    # result = [sentV2.fromtospan.to_dict() for sentV2 in sentV4_list if sentV2.fromtospan]
    result = [sentV2.fromtospan.to_dict() for sentV2 in sentV4_list if sentV2.fromtospan]
    out_st = json.dumps(result)
    strutils.dumps(out_st, file_name)

    
def save_sentv2_offsets(file_name, sentV2_list):
    result = []
    for sentV2 in sentV2_list:
        adict = {'start': sentV2.start,
                 'end': sentV2.end}
        result.append(adict)
    out_st = json.dumps(result)
    strutils.dumps(out_st, file_name)


def text2sentV2_list(doc_text: str, line_offsets, doc_tree: DTreeDoc) -> List[SentV2]:

    lineinfo_list, page_lineinfos_list = lxlineinfo.text2page_lineinfos_list(doc_text, line_offsets)

    if DEBUG_MODE:
        lineinfo_fname = doc_tree.file_name.replace('.txt', '.lineinfo.tsv')
        with open(lineinfo_fname, 'wt') as lineinfo_out:
            for lineinfo in lineinfo_list:
                print("{}".format(lineinfo), file=lineinfo_out)
        print("wrote {}".format(lineinfo_fname))

        lineinfo2_fname = doc_tree.file_name.replace('.txt', '.lineinfo2.tsv')
        with open(lineinfo2_fname, 'wt') as lineinfo_out:
            prev_pagenum = -1
            for lineinfo in lineinfo_list:
                if lineinfo.page != prev_pagenum:
                    print("\n\n=== page #{} =====================".format(lineinfo.page), file=lineinfo_out)
                if not lineinfo.is_close_prev_line:
                    print("", file=lineinfo_out)
                print("{}".format(lineinfo.tostr2(doc_text)), file=lineinfo_out)
                prev_pagenum = lineinfo.page
        print("wrote {}".format(lineinfo2_fname))
            
    pagenum_list, footer_list = footerutils.find_pagenum_footer(page_lineinfos_list)
    pheader_list = []

    # toc_parav2 are grouped lineinfos
    toc_parav2_list, tocline_list= tocfinder.extract_toc_parav2s(page_lineinfos_list, doc_text, pheader_list, footer_list, pagenum_list)

    doc_tree.toc = toc_parav2_list
    if DEBUG_MODE:
        toc_fname = doc_tree.file_name.replace('.txt', '.toc.tsv')
        with open(toc_fname, 'wt') as toc_out:
            for parav2 in toc_parav2_list:
                print("{}\n".format(parav2), file=toc_out)
        print("wrote {}".format(toc_fname))
                
    skip_lineinfo_set = set(tocline_list) | set(footer_list) | set(pagenum_list) | set(pheader_list)

    # secheadx_list, sechead_lineinfo_list = secheadutils.find_section_header(lineinfo_list, skip_lineinfo_set)
    secheadx_list, sechead_lineinfo_list = secheadutils.find_paged_section_header(page_lineinfos_list, skip_lineinfo_set)

    if DEBUG_MODE:
        for tocline in tocline_list:
            print("found tocline: page= %d, line_num= %d, text=%s" %
                  (tocline.page, tocline.sid, tocline.text))
        for pnum in pagenum_list:
            print("found page_num: page= %d, line_num= %d, text=%s" %
                  (pnum.page, pnum.sid, pnum.text))
        for footer in footer_list:
            print("found footer: page= %d, line_num= %d, footer=%s" %
                  (footer.page, footer.sid, footer.text))
        #for sechead in sechead_list:
        #    print("found sechead: page= %d, line_num= %d, sechead=%s" %
        #          (sechead.page, sechead.sid, sechead.text))
        for sechead in sechead_lineinfo_list:
            print("found sechead: page= %d, line_num= %d, sechead=%s" %
                  (sechead.page, sechead.sid, sechead.text))

        sechead_fname = doc_tree.file_name.replace('.txt', '.sechead.tsv')
        with open(sechead_fname, 'wt') as sechead_out:
            for sechead in secheadx_list:
                is_top_sechead, top_sechead_num = secheadutils.verify_sechead_prefix(sechead.prefix)
                print("sechead_prefix\t{} {}\t[{}]\t[{}]".format(is_top_sechead,
                                                                 top_sechead_num, sechead.prefix, sechead.title),
                      file=sechead_out)
            print("wrote {}".format(sechead_fname))

    skip_lineinfo_set |=  set(sechead_lineinfo_list)

    # parav2_list is NOT used!
    # parav2_list = parav2finder.find_parav2s(page_lineinfos_list, skip_lineinfo_set, doc_text)

    graph_segment_list, graph_lineinfos_list = (
        spanfinder.find_graph_spans(page_lineinfos_list, skip_lineinfo_set, doc_text))

    if DEBUG_MODE:
        segment_fname = doc_tree.file_name.replace('.txt', '.segment.tsv')
        with open(segment_fname, 'wt') as segment_out:
            for graph_segment in graph_segment_list:
                print("\n===found segment =====================:", file=segment_out)
                print("    {}".format(graph_segment), file=segment_out)
                print("    {}".format(doc_text[graph_segment.start:graph_segment.end]), file=segment_out)
            print("wrote {}".format(segment_fname))

    # all those lineinfos to skip line
    segment_lineinfo_list = []
    for graph_lineinfos in graph_lineinfos_list:
        for lineinfo in graph_lineinfos:
            skip_lineinfo_set.add(lineinfo)
        segment_lineinfo_list.extend(graph_lineinfos)

    # paged_sentV2s_list cannot have page not presented (no gaps though there can be blank pages)
    # page_lineinfos_list doesn't need this.
    # collect EbPDFTextStripper's linebreaks
    sentV2_list, paged_sentV2s_list = lxsentv2.init_sentV2s(doc_text,
                                                            lineinfo_list,
                                                            pagenum_list,
                                                            footer_list,
                                                            tocline_list,
                                                            sechead_lineinfo_list,
                                                            segment_lineinfo_list)

    # paged_sentv2_fname = doc_tree.file_name.replace('.txt', '.paged_sentv2.log')
    # lxsentv2.save_paged_sentv2s(paged_sentv2_fname, paged_sentV2s_list)
    # print("wrote {}".format(paged_sentv2_fname))
    
    # now extract segments that won't be parsed by CoreNLP
    if pagenum_list:
        doc_tree.pagenum_list = pagenum_list
    if footer_list:
        doc_tree.pfooter_list = footer_list
    if pheader_list:
        doc_tree.pheader_list = pheader_list        
        
    if DEBUG_MODE:
        dbsentv2_fname = doc_tree.file_name.replace('.txt', '.dbsentv2.tsv')
        with open(dbsentv2_fname, 'wt') as dbsentv2_out:        
            for i, sentV2 in enumerate(sentV2_list):
                print("sentV2 #{}: {}".format(i, sentV2), file=dbsentv2_out)
        print("wrote {}".format(dbsentv2_fname))

    if DEBUG_MODE:
        paged_sentv2_pretable_fname = doc_tree.file_name.replace('.txt', '.paged_sentv2_pretable.log')
        lxsentv2.save_paged_sentv2s(paged_sentv2_pretable_fname, paged_sentV2s_list)
        print("wrote {}".format(paged_sentv2_pretable_fname))

    table_list = tablefinder.find_table(paged_sentV2s_list)

    if DEBUG_MODE:
        table_fname = doc_tree.file_name.replace('.txt', '.tablex3.tsv')
        with open(table_fname, 'wt') as table_out:
            for table in table_list:
                print("\n====================================", file=table_out)
                print("table\tpage={} start={} end={}\n{}".format(table.pagenum, table.start, table.end,
                                                                  doc_text[table.start:table.end]), file=table_out)
                print("wrote {}".format(table_fname))

        table_fname = doc_tree.file_name.replace('.txt', '.table.tsv')
        doc_table_count = 0
        with open(table_fname, 'wt') as table_out:
            for pnumx, page_sentV2s in enumerate(paged_sentV2s_list, 1):

                cur_table_sentV2s = []
                table_list = []
                prev_sent_category = '---'
                for sentV2 in page_sentV2s:
                    if sentV2.category in set(['table', 'graph']):
                        if prev_sent_category in set(['table', 'graph']):
                            cur_table_sentV2s.append(sentV2)
                        else:
                            cur_table_sentV2s = [sentV2]
                            table_list.append(cur_table_sentV2s)
                    else:
                        pass
                    prev_sent_category = sentV2.category

                if table_list:  # print only if we have tables
                    print("\n======= page #{} =============================".format(pnumx), file=table_out)
                    for ti, table_sentV2s in enumerate(table_list, 1):
                        print("\ntable #{}-{}".format(doc_table_count + 1, ti), file=table_out)
                        doc_table_count += 1
                        for sentV2 in table_sentV2s:
                            print("\t{}\t{}".format(sentV2.category, sentV2.text), file=table_out)

            print("wrote {}".format(table_fname))

#    for graph_segment in graph_segment_list:
#        lxsentv2.fix_sentV2_list_in_segment(sentV2_list, graph_segment.category,
#                                            graph_segment.start, graph_segment.end)

    # fix_sentv2_category_issues(paged_sentV2s_list)

    if True:
        paged_sentv2_table_fname = doc_tree.file_name.replace('.txt', '.paged_sentv2_table.log')
        lxsentv2.save_paged_sentv2s(paged_sentv2_table_fname, paged_sentV2s_list)
        print("wrote {}".format(paged_sentv2_table_fname))

    # st_list here is not really used by anyone
    st_list = [sentV2.text for sentV2 in sentV2_list]

    return '\n\n'.join(st_list), sentV2_list, paged_sentV2s_list

# a para is a list of sentV4
# sepcial case in test4.txt
# exhibit   CONTRACT RATE \ ^ 3^)
# table	    Re l A
# exhibit   EXHIBIT A . \ r
# table     apSiifeippBiiliB
# need to collapse them into the same table
def extract_table_list(para_list):
    table_list = []
    prev_exhibit_para = []
    prev_exhibit_para_pagenum = 0
    prev_para_idx = -1
    prev_para_category = '---'
    for para_idx, para in enumerate(para_list):
        # sck, remove
        # print("para = {}".format(para))
        if not para:  # why sometimes they are empty?
            continue
        # print("para[0] = {}".format(type(para[0])))
        para_category = para[0].category
        para_pagenum = para[0].pagenum

        # print("extract_table_list: {}".format(para_category))

        if para_category == 'exhibit':
            if para_pagenum == prev_exhibit_para_pagenum and prev_exhibit_para_idx - para_idx < 4:
                prev_exhibit_para.append(para)
            else:
                prev_exhibit_para = []
                prev_exhibit_para.append(para)
                cur_table = []  # found a new exhibit, must start a new table
                table_list.append(cur_table)
            prev_exhibit_para_idx = para_idx
            prev_exhibit_para_pagenum = para_pagenum

        if para_category in set(['table', 'graph']):
            if prev_para_category in set(['table', 'graph']):
                pass
            elif prev_para_category in set(['exhibit']):  # this is for the special case above
                pass
            else:
                cur_table = []
                table_list.append(cur_table)
            if prev_exhibit_para and prev_exhibit_para[-1][0].pagenum == para_pagenum:
                for exhibit_para in prev_exhibit_para:
                    for sentV4 in exhibit_para:
                        cur_table.append(sentV4)
                prev_exhibit_para = []
            for sentV4 in para:
                cur_table.append(sentV4)
        prev_para_category = para_category

    # make sure we do not take empty tables due to "exhibit"
    table_list = [table for table in table_list if table]

    return table_list


# return the tuple (text, is_multi_lines)
def para_to_para_list(line):
    fake_line = ''
    if re.search("\n\s*\n", line):
        # print("weird double line in para..........")
        fake_line = re.sub('([\n\s]+)([\n\s][\n\s][\n\s])', r'\1'.replace('\n', ' ') + " XX253x", line)
        fake_line = fake_line.replace('\n', ' ').replace(' XX253x', ' \n\n')
        # print("fake line = {}".format(fake_line))

        # print("len(line) = {}, len(fake_line)= {}".format(len(line), len(fake_line)))
        return fake_line, True

    line_list = line.split('\n')
    max_line_len = 0
    num_notempty_line = 0
    for lx in line_list:
        lx_len = len(lx)
        if lx_len > max_line_len:
            max_line_len = lx_len
        if lx_len != 0:
            num_notempty_line += 1
    if num_notempty_line <= 1:
        return line, False  # not multi-line
    # print("line = [{}]".format(line))
    # print("max_line = {}".format(max_line_len))
    if max_line_len > 60:
        line = line.replace('\n', ' ')
        return line, False

    # num_period = line.count('.')
    nonl_line = line.replace('\n', ' ')
    words = nonl_line.split(' ')
    num_cap, num_lc, num_other = 0, 0, 0
    num_words = len(words)
    for word in words:
        if word:
            if word[0].islower():
                num_lc += 1
            elif word[0].isupper():
                num_cap += 1
            else:
                num_other += 1
    # print("num_lc {} / num_words {} = {}".format(num_lc, num_words, num_lc / float(num_words)))
    if num_words >= 8 and num_lc / float(num_words) >= 0.7:
        line = nonl_line
        return line, False
    return line, num_notempty_line > 1


def save_debug_txt_files(work_dir, base_fname, doc_text,
                         linebreak_offset_list,
                         doc_len, str_offsets, line_breaks, pblock_offsets, page_offsets):
    stred_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.str.txt'))
    breaked_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.each.nl.txt'))
    para_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.para.txt'))
    page_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.page.txt'))

    nl_set = set(linebreak_offset_list)

    # we don't really care about str, but it's x, y coordinates
    with open(stred_fn, 'wt') as fout:
        for i, str_offset in enumerate(str_offsets):
            start = str_offset['start']
            end = str_offset['end']
            
            if start != end:
                if end in nl_set:
                    line = doc_text[start:end] + '_NL_'
                else:
                    line = doc_text[start:end]
                print(line, file=fout)
            else:
                print('', file=fout)
        print('wrote {}'.format(stred_fn))

    with open(breaked_fn, 'wt') as fout:
        tmp_prev_end = 0
        for i, break_offset in enumerate(line_breaks):
            start = tmp_prev_end
            end = break_offset['offset']
            # adjust the start to exclude nl or space
            # print("start = {}, end= {}".format(start, end))
            while start < end and strutils.is_space_or_nl(doc_text[start]):
                # print("start2 = {}".format(start))
                start += 1
            while start <= end - 1 and strutils.is_nl(doc_text[end -1]):
                end -= 1
            if start != end:
                line = doc_text[start:end]
                # print("[{}]".format(line), file=fout)
                if end == doc_len:
                    # print("[{}]".format(line), file=fout)
                    print(line, file=fout, end='')
                    # print('{}\t{}\t[{}]'.format(start, end, line), file=fout, end='')
                else:
                    print(line, file=fout)
                    # print('{}\t{}\t[{}]'.format(start, end, line), file=fout)
            else:
                # we can ignore blank lines, which are usually at the
                # end of a page
                # print('', file=fout)
                pass
            tmp_prev_end = end + 1
        print('wrote {}'.format(breaked_fn))

    with open(para_fn, 'wt') as fout:
        for i, pblock_offset in enumerate(pblock_offsets, 1):
            start = pblock_offset['start']
            end = pblock_offset['end']
            if start != end:
                line = doc_text[start:end]
                # print(line, file=fout)
                print('{}\t{}\t[{}]'.format(start, end, line), file=fout)
            else:
                print('', file=fout)
            print('', file=fout)
        print('wrote {}'.format(para_fn))

    with open(page_fn, 'wt') as fout:
        for i, page_offset in enumerate(page_offsets, 1):
            start = page_offset['start']
            end = page_offset['end']
            print('==================== page #{} ===================='.format(i), file=fout)
            line = doc_text[start:end]
            print(line, file=fout)
        print('wrote {}'.format(page_fn))

def to_nl_paraline_texts(file_name, offsets_file_name, work_dir):
    base_fname = os.path.basename(file_name)

    orig_doc_text = strutils.loads(file_name)
    doc_len, str_offsets, line_breaks, pblock_offsets, page_offsets = load_pdf_offsets(offsets_file_name)
    # print('doc_len = {}, another {}'.format(doc_len, len(doc_text)))

    linebreak_offset_list = [lbrk['offset'] for lbrk in line_breaks]
    ch_list = list(orig_doc_text)
    for linebreak_offset in linebreak_offset_list:
        ch_list[linebreak_offset] = '\n'
    nl_text = ''.join(ch_list)

    # save the result
    nl_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nl.txt'))
    txtreader.dumps(nl_text, nl_fn)
    # print('wrote {}, size= {}'.format(nl_fn, len(nl_text)))

    doc_text = nl_text

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
        while start < end and strutils.is_space_or_nl(doc_text[start]):
            # print("start2 = {}".format(start))
            start += 1
        while start <= end - 1 and strutils.is_nl(doc_text[end -1]):
            end -= 1
        if start != end:
            bxid_lineinfos_map[block_num].append(LineInfo3(start, end, line_num, lxid_strinfos_map[line_num]))
        tmp_prev_end = end + 1

    pgid_pblockinfos_map = defaultdict(list)
    block_info_list = []
    for pblock_offset in pblock_offsets:
        pblock_id = pblock_offset['id']
        start = pblock_offset['start']
        end = pblock_offset['end']
        page_num = pblock_offset['pageNum']

        while start <= end - 1 and strutils.is_nl(doc_text[end -1]):
            end -= 1

        if start != end:
            para_line, is_multi_lines = para_to_para_list(doc_text[start:end])
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

    """
    pageinfo_list = []
    nlp_offset = 0
    for page_offset in page_offsets:
        start = page_offset['start']
        end = page_offset['end']
        page_num = page_offset['id']
        pblockinfo_list = pgid_pblockinfos_map[page_num]
        pinfo = PageInfo(start, end, page_num, pblockinfo_list)
        nlp_offset = pinfo.init_line4nlp_list(nlp_offset)
        pageinfo_list.append(pinfo)
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

    # save the result
    paraline_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.paraline.txt'))
    txtreader.dumps(paraline_text, paraline_fn)
    # print('wrote {}, size= {}'.format(paraline_fn, len(paraline_text)))

    return orig_doc_text, nl_text, paraline_text, nl_fn, paraline_fn


def parse_document(file_name, offsets_file_name, work_dir):
    debug_mode = False

    base_fname = os.path.basename(file_name)

    orig_doc_text = strutils.loads(file_name)
    doc_len, str_offsets, line_breaks, pblock_offsets, page_offsets = load_pdf_offsets(offsets_file_name)
    # print('doc_len = {}, another {}'.format(doc_len, len(doc_text)))

    nl_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nl.txt'))
    linebreak_offset_list = [lbrk['offset'] for lbrk in line_breaks]
    ch_list = list(orig_doc_text)
    for linebreak_offset in linebreak_offset_list:
        ch_list[linebreak_offset] = '\n'
    nl_text = ''.join(ch_list)
    txtreader.dumps(nl_text, nl_fn)
    print('wrote {}'.format(nl_fn))

    doc_text = nl_text

    if debug_mode:
        save_debug_txt_files(work_dir, base_fname, doc_text,
                             linebreak_offset_list,
                             doc_len, str_offsets, line_breaks, pblock_offsets, page_offsets)

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
        while start < end and strutils.is_space_or_nl(doc_text[start]):
            # print("start2 = {}".format(start))
            start += 1
        while start <= end - 1 and strutils.is_nl(doc_text[end -1]):
            end -= 1
        if start != end:
            bxid_lineinfos_map[block_num].append(LineInfo3(start, end, line_num, lxid_strinfos_map[line_num]))
        tmp_prev_end = end + 1

    pgid_pblockinfos_map = defaultdict(list)
    block_info_list = []
    for pblock_offset in pblock_offsets:
        pblock_id = pblock_offset['id']
        start = pblock_offset['start']
        end = pblock_offset['end']
        page_num = pblock_offset['pageNum']

        while start <= end - 1 and strutils.is_nl(doc_text[end -1]):
            end -= 1

        if start != end:
            para_line, is_multi_lines = para_to_para_list(doc_text[start:end])
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
        pinfo = PageInfo(start, end, page_num, pblockinfo_list)
        nlp_offset = pinfo.init_line4nlp_list(nlp_offset)
        pageinfo_list.append(pinfo)

        
    """
    for pagenum, pageinfo in enumerate(pageinfo_list):
        strinfo_list = paged_strinfos_map.get(pagenum + 1, [])
        pageinfo.set_strinfo_list(strinfo_list)
        # for debugging purpose

        print('page num: {}'.format(pagenum))
        for i, strinfo in enumerate(strinfo_list):
            print("strinfo {}: {}".format(i, strinfo))
            print("      text: {}".format(doc_text[strinfo['start']:strinfo['end']]))
        for i, pblockinfo in enumerate(pageinfo.pblockinfo_list):
            print("pblockinfo {}: {} {}".format(i, pblockinfo.start, pblockinfo.end))

        pageinfo.init_pblockinfos()
    """

    paged_para_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.paged_para.txt'))
    with open(paged_para_fn, 'wt') as fout:
        for pageinfo in pageinfo_list:
            for pblockinfo in pageinfo.pblockinfo_list:
                # print('[{}]'.format(pblockinfo.text), file=fout)
                print(pblockinfo.text, file=fout)
                print('', file=fout)
        print('wrote {}'.format(paged_para_fn))

    paged_debug_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.paged_debug.txt'))
    with open(paged_debug_fn, 'wt') as fout:
        for pageinfo in pageinfo_list:
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

    nlp_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nlp.txt'))
    with open(nlp_fn, 'wt') as fout:
        for pageinfo in pageinfo_list:
            for line4nlp in pageinfo.line4nlp_list:
                print(doc_text[line4nlp.orig_start:line4nlp.orig_end].replace('\n', ' '), file=fout)
                print('', file=fout)
        print('wrote {}'.format(nlp_fn))


    sep_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.paraline.txt'))
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


def parse_document_old(file_name, offsets_file_name, work_dir):
    debug_mode = False

    base_fname = os.path.basename(file_name)

    doc_text = strutils.loads(file_name)
    doc_len, str_offsets, line_breaks, para_offsets, page_offsets = load_pdf_offsets(offsets_file_name)
    doc_tree = DTreeDoc(file_name, 0, len(doc_text), doc_text)

    text4nlp_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nlp.txt'))
    text4nlp_offsets_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nlp.offsets.json'))
    # print("text4nlp_fn: [{}]".format(text4nlp_fn))
    # print("text4nlp_offsets_fn: [{}]".format(text4nlp_offsets_fn))

    # offsets_fname = file_name.replace(".txt", ".lineinfo.json")
    text_sentV2, sentV2_list, page_sentV2s_list = text2sentV2_list(doc_text, line_offsets, doc_tree)
    if debug_mode:
        strutils.dumps(text_sentV2, file_name.replace('.txt', '.sentv2.txt'))
        print("wrote {}".format(file_name.replace('.txt', '.sentv2.txt')))
        offsets_sentv2_fname = file_name.replace(".txt", ".offsets.sentv2.json")
        save_sentv2_offsets(offsets_sentv2_fname, sentV2_list)
        print("wrote {}".format(offsets_sentv2_fname))

    # collect pagenum and footer, so can remove them in annotation in the future
    gap_span_list = get_gap_span_list(sentV2_list)

    debug_para_fn = file_name.replace('.txt', '.para.debug.txt')
    # para_list = sent4nlp.page_sentV2s_list_to_para_list(doc_text, page_sentV2s_list, file_name=None)
    para_list = sent4nlp.page_sentV2s_list_to_para_list(doc_text, page_sentV2s_list,
                                                        file_name=debug_para_fn)

    debug_group_fn = file_name.replace('.txt', '.para.group.txt')
    para_sentv2_list = sent4nlp.page_sentV2s_list_to_group_list(page_sentV2s_list,
                                                                file_name=debug_group_fn)

    debug_g2_fn = file_name.replace('.txt', '.group2.debug.txt')
    para_sentv2_to_text4nlp, g2_sentV4_list = sent4nlp.para_sentv2_to_text4nlp(para_sentv2_list, file_name=debug_g2_fn)

    strutils.dumps(para_sentv2_to_text4nlp, text4nlp_fn)
    logging.info("wrote {}".format(text4nlp_fn))
    save_sent_nlp_offsets(text4nlp_offsets_fn, g2_sentV4_list)
    logging.info("wrote {}".format(text4nlp_offsets_fn))

    return doc_text, gap_span_list, text4nlp_fn, text4nlp_offsets_fn, para_list


def parse_document_no_skip(file_name, offsets_file_name, work_dir):
    debug_mode = False

    base_fname = os.path.basename(file_name)

    doc_text = strutils.loads(file_name)
    # line_offsets = strutils.load_json_list(offsets_file_name)
    # line_offsets, line_breaks = load_pdf_offsets(offsets_file_name)
    doc_len, str_offsets, line_breaks, para_offsets, page_offsets = load_pdf_offsets(offsets_file_name)
    doc_tree = DTreeDoc(file_name, 0, len(doc_text), doc_text)

    text4lined_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.lined.txt'))
    text4lined_offsets_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.lined.offsets.json'))
    print("text4lined_fn: [{}]".format(text4lined_fn))
    print("text4lined_offsets_fn: [{}]".format(text4lined_offsets_fn))

    # offsets_fname = file_name.replace(".txt", ".lineinfo.json")
    text_sentV2, sentV2_list, page_sentV2s_list = text2sentV2_list(doc_text, str_offsets, doc_tree)
    if debug_mode:
        strutils.dumps(text_sentV2, file_name.replace('.txt', '.sentv2.txt'))
        print("wrote {}".format(file_name.replace('.txt', '.sentv2.txt')))
        offsets_sentv2_fname = file_name.replace(".txt", ".offsets.sentv2.json")
        save_sentv2_offsets(offsets_sentv2_fname, sentV2_list)
        print("wrote {}".format(offsets_sentv2_fname))

    # collect pagenum and footer, so can remove them in annotation in the future
    gap_span_list = get_gap_span_list(sentV2_list)

    # text4lined, sentV4_list = sent4nlp.sentV2_list_to_sent4nlp_list(doc_text, sentV2_list)
    # debug_lined_fn = file_name.replace('.txt', '.lined.debug1.txt')
    # text4lined, sentV4_list = sent4nlp.page_sentV2s_list_to_sent4lined_list(doc_text, page_sentV2s_list, debug_lined_fn, is_debug_mode=True)
    debug_lined_fn = file_name.replace('.txt', '.lined.debug2.txt')
    text4lined, sentV4_list = sent4nlp.page_sentV2s_list_to_sent4lined_list(doc_text, page_sentV2s_list, file_name=None)
    strutils.dumps(text4lined, text4lined_fn)
    logging.info("wrote {}".format(text4lined_fn))
    
    save_sent_nlp_offsets(text4lined_offsets_fn, sentV4_list)
    logging.info("wrote {}".format(text4lined_offsets_fn))

    # the offsets in para_list is for doc_text
    debug_para_fn = file_name.replace('.txt', '.para.debug.txt')
    para_list = sent4nlp.page_sentV2s_list_to_para_list(doc_text, page_sentV2s_list, file_name=None)

    return doc_text, gap_span_list, para_list


# return a list of 2-tuples (start, end)
def get_gap_span_list(sentV2_list: List[SentV2]):
    cur_skip_sents = []
    skip_list = []
    is_prev_skip = False
    for sentV2 in sentV2_list:
        if sentV2.category in set(['pagenum', 'footer', 'toc']):
            if is_prev_skip:
                cur_skip_sents.append(sentV2)
            else:
                cur_skip_sents = [sentV2]
                skip_list.append(cur_skip_sents)
            is_prev_skip = True
        else:
            is_prev_skip = False

    result = []            
    for skip_sentv2s in skip_list:
        first_sentv2 = skip_sentv2s[0]
        last_sentv2 = skip_sentv2s[-1]
        start_end_tuple = (first_sentv2.start, last_sentv2.end)
        # print("gap_span({}, {})".format(start_end_tuple[0], start_end_tuple[1]))
        result.append(start_end_tuple)
            
    # the result is sorted
    return sorted(result)


def update_ant_spans(ant_list, gap_span_list, doc_text):

    se_ant_list_map = defaultdict(list)
    for ant in ant_list:
        se = (ant['start'], ant['end'])
        se_ant_list_map[se].append(ant)
    ant_se_list = sorted(se_ant_list_map.keys())
    # print("  ant_se_list: {}".format(ant_se_list))
    # print("gap_span_list: {}".format(gap_span_list))

    min_possible_j, jmax = 0, len(gap_span_list)    
    
    for ant_se in ant_se_list:

        overlap_spans = []
        for j in range(min_possible_j, jmax):
            gap_span = gap_span_list[j]
            if gap_span[1] < ant_se[0]:
                min_possible_j = j+1
            if mathutils.start_end_overlap(ant_se, gap_span):
                overlap_spans.append(gap_span)
                # print('overlap {}, {}'.format(se_ant_list_map[ant_se][0], gap_span))
            # because annotations can overlap,
            # this guarantee is false
            # if gap_span[0] > ant_end:
            #    break

        overlap_spans2 = []
        for gap_span in gap_span_list:
            if mathutils.start_end_overlap(ant_se, gap_span):
                overlap_spans2.append(gap_span)
                # print('overlap2 {}, {}'.format(se_ant_list_map[ant_se][0], gap_span))

        #if overlap_spans2 != overlap_spans:
        #    print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", file=sys.stderr)
        #    print("overlap_span = {}".format(overlap_spans), file=sys.stderr)
        #    print("overlap_span2 = {}".format(overlap_spans2), file=sys.stderr)
            
        if overlap_spans:
            # adjusted_spanst_list = []
            endpoint_list = [ant_se[0], ant_se[1]]
            max_end = len(doc_text)
            for gap_span in overlap_spans:
                tmp_start = gap_span[0]
                # find the first space after an non-space to the left of gap
                while tmp_start > 0 and strutils.is_space(doc_text[tmp_start-1]):
                    tmp_start -= 1
                tmp_end = gap_span[1]
                
                # find the first non-space to the right of the gap
                while tmp_end < max_end and strutils.is_space(doc_text[tmp_end]):
                    tmp_end += 1

                # adjusted_spanst_list.append("{}:{}".format(tmp_start, tmp_end))
                endpoint_list.append(tmp_start)
                endpoint_list.append(tmp_end)
                # print("adjusted span {}:{} -> {}:{}".format(gap_span[0], gap_span[1], tmp_start, tmp_end))
            endpoint_list.sort()

            endpoints_st_list = ['{}:{}'.format(endpoint_list[i], endpoint_list[i+1]) for i in range(0, len(endpoint_list), 2)]
            # spans_st = ','.join(adjusted_spanst_list)
            # spans_st = ','.join([str(endpoint) for endpoint in endpoint_list])
            spans_st = ','.join(endpoints_st_list)
        else:
            spans_st = '{}:{}'.format(ant_se[0], ant_se[1])

        se_ant_list = se_ant_list_map[ant_se]
        for antx in se_ant_list:
            antx['start_end_span_list'] = spans_st


def read_fromto_json(file_name: str):
    fromto_list = strutils.load_json_list(file_name)

    alist = []
    for fromto in fromto_list:
        from_start = fromto['from_start']
        # from_end = fromto['from_end']
        to_start = fromto['to_start']
        # to_end = fromto['to_end']
        alist.append((from_start, to_start))
        # alist.append((from_end, to_end))

    sorted_alist = sorted(alist)
    
    from_list = [a for a,b in sorted_alist]
    to_list = [b for a,b in sorted_alist]    
    return from_list, to_list

def read_fromto_json_pairs(file_name: str):
    fromto_list = strutils.load_json_list(file_name)

    from_list = []
    to_list = []
    for fromto in fromto_list:
        from_start = fromto['from_start']
        from_end = fromto['from_end']
        to_start = fromto['to_start']
        to_end = fromto['to_end']
        from_list.append((from_start, from_end))
        to_list.append((to_start, to_end))        
        
    return from_list, to_list


# binary search version
def find_offset_to(fromx: int, from_list, to_list):

    # find rightmost value less than or equal to fromx
    found_i = bisect.bisect_right(from_list, fromx)
    if found_i:
        if fromx == from_list[found_i-1]:
            return to_list[found_i-1]
        diff = fromx - from_list[found_i-1]
        return to_list[found_i-1] + diff

    return -1


# linear version
def find_offset_to_linear(fromx: int, from_list, to_list):
    found_i = -1
    for i, val in enumerate(from_list):
        if val >= fromx:
            found_i = i
            break

    if found_i != -1:
        if fromx == from_list[found_i]:
            return to_list[found_i]
        diff = fromx - from_list[found_i]
        return to_list[found_i] + diff

    return -1


def txt_to_linedtxtant(base_txt_fn, base_lineinfo_fn, base_ant_fn, upload_dir, work_dir):
    time1 = time.time()

    txt_fn = '{}/{}'.format(upload_dir, base_txt_fn)
    lineinfo_fn = '{}/{}'.format(upload_dir, base_lineinfo_fn)
    ant_fn = '{}/{}'.format(upload_dir, base_ant_fn)

    doc_text, gap_span_list, para_list = parse_document_no_skip(txt_fn, lineinfo_fn, work_dir)
    # Currently, we don't do anything with para_list for custom_train, which
    # is the only workflow that calls this.
    # now file_name.lined.txt and file_name.lined.offsets.json are created

    text4nlp_fn = '{}/{}'.format(work_dir, base_txt_fn.replace('.txt', '.lined.txt'))
    text4nlp_offsets_fn = '{}/{}'.format(work_dir, base_txt_fn.replace('.txt', '.lined.offsets.json'))

    # save the prov_labels_map
    ant_list = []
    if ant_fn.endswith('.ant'):
        nlp_ants_fn = '{}/{}'.format(work_dir, base_txt_fn.replace('.txt', '.lined.ant'))
        with open(ant_fn, 'rt') as handle:
            parsed = json.load(handle)
            for ajson in parsed:
                ant_list.append(ajson)
    else:
        nlp_ants_fn = '{}/{}'.format(work_dir, base_txt_fn.replace('.txt', '.lined.ebdata'))
        with open(ant_fn, 'rt') as handle:
            parsed = json.load(handle)
            prov_ants = parsed['ants']
            for prov, ajsons in prov_ants.items():
                print("prov = {}".format(prov))
                for ajson in ajsons:
                    ant_list.append(ajson)

    # translate the offsets
    from_list, to_list = read_fromto_json(text4nlp_offsets_fn)
    for antx in ant_list:
        # print("ant start = {}, end = {}".format(antx['start'], antx['end']))
        xstart = antx['start']
        xend = antx['end']
        antx['txt_start'] = xstart
        antx['txt_end'] = xend
        antx['start'] = find_offset_to(xstart, to_list, from_list)
        antx['end'] = find_offset_to(xend, to_list, from_list)

    ant_out_st = json.dumps(ant_list)
    strutils.dumps(ant_out_st, nlp_ants_fn)

    time2 = time.time()
    logging.info('txt_to_nlptxtant(%s, %s, %s) took %0.2f sec', txt_fn, lineinfo_fn, ant_fn, (time2 - time1))
    return text4nlp_fn, nlp_ants_fn


# lineinfo has its data structure
# lineOffsets are just integers
# blockOffsets has 'start' and 'end' attribute
# pageOffsets has 'start' and 'end' attribute
def load_pdf_offsets(file_name: str):
    atext = strutils.loads(file_name)
    ajson = json.loads(atext)
    return (ajson.get('docLen'), ajson.get('strOffsets'), ajson.get('lineOffsets'),
            ajson.get('blockOffsets'), ajson.get('pageOffsets'))