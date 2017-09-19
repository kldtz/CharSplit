import argparse
from collections import defaultdict, namedtuple
import logging
import os
import re
import sys
from typing import List, Dict


from kirke.docstruct import docstructutils
from kirke.docstruct import pdfutils, pdfoffsets, secheadutils
from kirke.docstruct.pdfoffsets import GroupedBlockInfo, LineInfo3, PageInfo3, PBlockInfo
from kirke.docstruct.pdfoffsets import PDFTextDoc, StrInfo
from kirke.utils import strutils, txtreader, mathutils


class PageInfo6:

    def __init__(self, start, end, page_num, linex_list):
        self.start = start
        self.end = end
        self.page_num = page_num
        self.linex_list = linex_list


class PDFTextDoc6:

    def __init__(self, file_name: str, doc_text: str,
                 page_list: List[PageInfo6], num_char_per_line: int):
        self.file_name = file_name
        self.doc_text = doc_text
        self.page_list = page_list
        self.num_pages = len(page_list)
        self.pageinfo6_list = []
        self.special_blocks_map = defaultdict(list)
        self.num_char_per_line = num_char_per_line

    def print_debug_blocks(self):
        for page_num, paged_linexes in enumerate(self.paged_linexes_list, 1):
            print("\n===== page #{}".format(page_num))
            for line_seq, linex in enumerate(paged_linexes, 1):
                print("linex #{} {} ||\t{}".format(line_seq, linex.tostr(),
                                                   linex.line_text))

class Line6WithAttrs:

    def __init__(self,
                 start, end,
                 page_line_num, line_text, page_num,
                 str_list, attrs):
        self.start = start
        self.end = end
        self.page_line_num = page_line_num  # start from 1, not 0
        self.line_text = line_text
        self.length = len(line_text)
        self.page_num = page_num
        self.str_list = str_list
        self.attrs = attrs
        self.is_english = docstructutils.is_line_english(line_text)
        self.table_cell_list = []
        self.col_dividers = []

    def __lt__(self, other):
        return (self.start, ).__lt__((other.start, ))

    def __str__(self):
        # print("    jjj line {}, {} spaces={} ||\t [{}]".format(start, end, find_large_spaces(line), line))
        alist = []
        alist.append('pg=%d' % self.page_num)
        alist.append('ln=%d' % self.page_line_num)
        alist.append('se=(%d, %d)' % (self.start, self.end))
        alist.append(', '.join(['{}={}'.format(attr, value) for attr, value in sorted(self.attrs.items())]))

        return '{} ||\t[{}]'.format(', '.join(alist), self.line_text)

    def tostr2(self):
        # print("    jjj line {}, {} spaces={} ||\t [{}]".format(start, end, find_large_spaces(line), line))
        alist = []
        alist.append('pg=%d' % self.page_num)
        alist.append('ln=%d' % self.page_line_num)
        alist.append('se=(%d, %d)' % (self.start, self.end))
        alist.append(', '.join(['{}={}'.format(attr, value) for attr, value in sorted(self.attrs.items())]))

        if self.table_cell_list:
            tout = []
            for tcell in self.table_cell_list:
                tout.append('[col={}: {}]'.format(tcell.col_num, tcell.str6.text))
            return '{} ||\t[{}]'.format(', '.join(alist), '\t'.join(tout))

        return '{} ||\t[{}]'.format(', '.join(alist), self.line_text)

    def add_table_columns(self, col_dividers):
        table_cell_list = []
        len_dividers = len(col_dividers)
        for str6 in self.str_list:
            for idx in range(len_dividers):
                if str6.xstart < col_dividers[idx]:
                    table_cell_list.append(TableCell(idx, -1, str6))
                    # tabbed_out.append('[col={}: {}]'.format(idx, str6.text))
                    break
        self.table_cell_list = table_cell_list
        self.col_dividers = col_dividers
        # print("self.table_cell_list = {}, len= {}".format(self.table_cell_list, len(table_cell_list)))

    def table_columns(self, col_dividers):
        alist = []
        alist.append('pg=%d' % self.page_num)
        alist.append('ln=%d' % self.page_line_num)
        alist.append('se=(%d, %d)' % (self.start, self.end))
        alist.append(', '.join(['{}={}'.format(attr, value) for attr, value in sorted(self.attrs.items())]))

        tabbed_out = []
        len_dividers = len(col_dividers)
        for str6 in self.str_list:
            for idx in range(len_dividers):
                if str6.xstart < col_dividers[idx]:
                    tabbed_out.append('[col={}: {}]'.format(idx, str6.text))
                    break
                    
        return '{} ||\t[{}]'.format(', '.join(alist), '\t'.join(tabbed_out))
        

Str6 = namedtuple('Str6', ['start', 'end', 'text', 'xstart', 'xend'])

class TableCell:

    def __init__(self, col_num, line_seq, str6):
        self.col_num = col_num
        self.line_seq = line_seq
        self.str6 = str6


# full character per line is 160
# This is computed per document now.
# Stored PdfTextDoc6
# NUM_CHAR_PER_LINE = 160

def is_line_centered(line: str, num_char_per_line: int, str_start: int):
    num_right_space = num_char_per_line - len(line)
    if num_right_space <= 0:
        # print('str_start = {}, num_right_space= {}'.format(float(str_start), num_right_space))
        num_right_space = 1
    left_to_right_ratio = round(float(str_start) / num_right_space, 2)
    return (left_to_right_ratio >= 0.83 and left_to_right_ratio <= 1.17, left_to_right_ratio)


def line2str6_list(line:str, line_start: int, num_char_per_line: int) -> (List[Str6], Dict):
    space_offset_list = find_large_spaces(line)
    str_list = []
    num_column = 0
    is_centered = False
    lr_ratio = -1  # left_to_right_ratio
    is_sechead_prefix = False
    is_phone_number = False
    is_email = False
    attrs = {}

    if len(space_offset_list) == 0:
        if len(line) != 0:
            space_offset_list.append((line_start + 0, line_start + len(line), line))
    elif len(space_offset_list) == 1:  # empty or just left aligned text
        str_start = space_offset_list[0][1]
        str_line = line[str_start:]
        str_list.append(Str6(line_start + str_start, line_start + len(line), str_line, str_start, len(line)))

        """
        num_right_space = NUM_CHAR_PER_LINE - len(line)
        if num_right_space <= 0:
            # print('str_start = {}, num_right_space= {}'.format(float(str_start), num_right_space))
            num_right_space = 1
        left_to_right_ratio = float(str_start) / num_right_space
        if left_to_right_ratio >= 0.9  and left_to_right_ratio <= 1.1:
            is_centered = True
        """
        is_centered, lr_ratio = is_line_centered(line, num_char_per_line, str_start)
        if secheadutils.is_line_sechead_prefix(str_line):
            is_sechead_prefix = True
            sechead_tuple = docstructutils.extract_line_sechead(str_line)
            if sechead_tuple:
                attrs['sechead'] = sechead_tuple 

        if docstructutils.is_line_phone_number(str_line):
            is_phone_number = True

        if docstructutils.is_line_email(str_line):
            is_email = True            

    else:  # multiple long spaces, likely columns in a table, or header or footer
        prev_end = 0
        for space_offset in space_offset_list:
            if space_offset[0] == 0:  # begin of a line
                prev_end = space_offset[1]
            else:
                str_start = prev_end
                str_end = space_offset[0]
                str_list.append(Str6(line_start + str_start, line_start + str_end, line[str_start:str_end], str_start, str_end))
                prev_end = space_offset[1]
        # for the last str in str_list
        str_start = prev_end
        str_list.append(Str6(line_start + str_start, line_start + len(line), line[str_start:], str_start, len(line)))

    attrs['num_col'] = len(str_list)
    if is_centered:
        attrs['cn'] = True
    if is_sechead_prefix:
        attrs['sec'] = True
    if is_phone_number:
        attrs['pho'] = True
    if is_email:
        attrs['eml'] = True
    if is_line_page_num(line):
        attrs['pagenum'] = True

    attrs['lr_ratio'] = lr_ratio

    if str_list:
        attrs['str_xstarts'] = [str6.xstart for str6 in str_list]

    #for strx in str_list:
    #    print("strx {}".format(strx))
    #if str_list:
    #    print(', '.join(['{}={}'.format(attr, value) for attr, value in sorted(attrs.items())]))
        # print("num_column = {}, is_centered = {}, is_sechead = {}, left_to_right_ratio = {}".format(len(str_list), is_centered, is_sechead_prefix, left_to_right_ratio))
        # print("num_column={}, cn={}, sec={}, lr_ratio= {:.1f}, pho={}, em={}".format(len(str_list), is_centered, is_sechead_prefix, left_to_right_ratio, is_phone_number, is_email))
        
    return str_list, attrs
    

"""
# remove all begin and end spaces for lines
# 'be' = begin_end
def load_paged_text_with_offsets(file_name: str):
    offsets_lines = txtreader.load_lines_with_offsets(file_name)

    cur_page_linex_list = []
    page_list = [cur_page_linex_list]
    page_offset_list = [0]

    for start, end, line_text in offsets_lines:
        if line_text and ord(line_text[0]) == 12:
            page_offset_list.append(start)
        print("{}\t{}\t{}".format(start, end, line_text))
    # take the last page
    page_offset_list.append(end)

    print("page_offsets = {}".format(page_offset_list))
    return []
"""

FOUR_OR_MORE_SPACES = re.compile(r'(^\s*|\s{4}\s*)')


def find_large_spaces(line: str):
    alist = []
    for mat in FOUR_OR_MORE_SPACES.finditer(line):
        start, end = mat.start(), mat.end() 
        alist.append((start, end, end - start))
    return alist


def remove_ocr_error_23(line: str) -> str:
    return strutils.replace_dot3plus_with_spaces(line)

def parse_document(file_name, work_dir, debug_mode=True):
    base_fname = os.path.basename(file_name)

    doc_text = strutils.loads(file_name)

    page_offsets, page_list = txtreader.load_page_lines_with_offsets(file_name)

    # compute the num_char_per_line
    line_len_list = []
    for paged_line_list in page_list:
        for start, end, line in paged_line_list:
            line_len = end - start
            if line_len != 0:
                line_len_list.append(line_len)
    # figure3 out the length of a full line for computing center
    sorted_line_len_list = sorted(line_len_list)
    if not sorted_line_len_list:
        num_char_per_line = 0
    elif len(sorted_line_len_list) <= 10:
        num_char_per_line = sorted_line_len_list[-1]  # max
    else:
        # take the average of the last 2 number
        #num_char_per_line = int((sorted_line_len_list[-2] +
        #                         sorted_line_len_list[-1]) / 2)
        # take 90%
        top90 = int(len(sorted_line_len_list) * 0.90)
        num_char_per_line = sorted_line_len_list[top90]
    print("sorted_line_len_list = {}".format(sorted_line_len_list))
    print("num_char_per_line = {}".format(num_char_per_line))

    pageinfo6_list = []
    for page_num, (page_offset, paged_line_list) in enumerate(zip(page_offsets, page_list), 1):
        page_start, page_end = page_offset
        linex_list = []
        for line_seq, (start, end, line) in enumerate(paged_line_list, 1):

            line = remove_ocr_error_23(line)
            # print("    jjj line {}, {} spaces={} ||\t [{}]".format(start, end, find_large_spaces(line), line))
            # need num_char_per_line to computer is_centered()
            str_list, line_attrs = line2str6_list(line, start, num_char_per_line)

            linex = Line6WithAttrs(start, end, line_seq, line, page_num, str_list, line_attrs)
            linex_list.append(linex)
            print(linex)
        pageinfo6_list.append(PageInfo6(page_start, page_end, page_num, linex_list))

    pdf_text_doc = PDFTextDoc6(file_name, doc_text, pageinfo6_list, num_char_per_line)

    add_doc_structure_to_doc(pdf_text_doc)

    # add_doc_structure_to_doc(pdf_text_doc)
    return pdf_text_doc

NUM_CHAR_PER_LINE = 160

def add_doc_structure_to_doc(pdftxt_doc):
    # first remove obvious non-content lines, such
    # toc, page-num, header, footer
    # Also add section heads
    # page_attrs_list is to store table information?
    for page in pdftxt_doc.page_list:
        add_doc_structure_to_page(page, pdftxt_doc)

    # now we have basic block_group with correct
    # is_english set.  Useful for merging
    # blocks with only 1 lines as table, or signature section
    #for apage in pdftxt_doc.page_list:
    #    merge_adjacent_line_with_special_attr(apage)
    # pdftxt_doc.save_debug_pages(extension='.debug6.mergepage.tsv')

def find_column_dividers(linex_list: List[Line6WithAttrs]):
    xxx = linex_list
    linex_list = [linex for linex in linex_list if not linex.attrs.get('table-not-column-divider-calc')]
    
    max_col = max([linex.length for linex in linex_list])
    col_isspace_count = defaultdict(int)
    for linex in linex_list:
        line_text = linex.line_text
        for idx in range(max_col):
            if ((idx < linex.length and line_text[idx] == ' ') or
                (idx >= linex.length)):
                col_isspace_count[idx] += 1

    max_space_count = len(linex_list)
    # consecutive_space_count
    consec_space = 0
    divide_list = []
    for idx in range(max_col):
        if col_isspace_count[idx] == max_space_count:
            consec_space += 1
        else:
            if idx <= 4 and consec_space == idx:
                divide_list.append(idx)
            elif consec_space > 4:
                divide_list.append(idx)
            consec_space = 0

    # the last one
    divide_list.append(max_col)
    return divide_list
    
def no_english_in_list(alist: List[str]) -> bool:
    if not alist:
        return True
    # for strx in alist:
        # if docstructutils.is_line_english(strx):
        # return False
    first_strx = alist[0]
    words = first_strx.split()
    if len(words) > 7:
        return False
    return True

def find_lines_with_same_cols(linex_list: List[Line6WithAttrs]):
    first_line = linex_list[0]
    num_cols = len(first_line.str_list)
    result = [first_line]
    for linex in linex_list[1:]:
        if len(linex.str_list) == num_cols:
            result.append(linex)
        # tables with rows with slightly off xstart.
        # now continue until we found a row with lots of words (> 7)
        elif (len(linex.str_list) < num_cols and
            no_english_in_list([strx.text for strx in linex.str_list])):
            result.append(linex)            
        else:
            break
    if len(result) < 4:    # in case that it's a 3 column table, with distinct columns (2 + 1 + 2)
        result = linex_list[:4]
    return result


def separate_page_nums(block_list):
    out_block_list = []
    for block in block_list:
        is_prev_pagenum = False
        cur_block = []
        out_block_list.append(cur_block)
        for line_seq, linex in enumerate(block):
            if not linex.attrs.get('pagenum'):
                if line_seq == 0 or not is_prev_pagenum:
                    cur_block.append(linex)
                else:
                    # non-page after page-num
                    cur_block = [linex]
                    out_block_list.append(cur_block)
                is_prev_pagenum = False
            else:
                if line_seq == 0 or is_prev_pagenum:  # for first line and other
                    # already in pagenum mode, no change
                    cur_block.append(linex)
                else:
                    # pagenum in after non-page-numbs
                    cur_block = [linex]
                    out_block_list.append(cur_block)
                is_prev_pagenum = True

    return out_block_list


def separate_secheads(block_list):
    out_block_list = []
    for block in block_list:
        cur_block = []
        out_block_list.append(cur_block)
        for line_seq, linex in enumerate(block):
            if linex.attrs.get('sechead'):
                if cur_block:  # already has something
                    cur_block = [linex]
                    out_block_list.append(cur_block)
                    cur_block = []
                    out_block_list.append(cur_block)
                else:  # nothing in cur_block
                    cur_block.append(linex)
                    cur_block = []
                    out_block_list.append(cur_block)
            else:
                cur_block.append(linex)

    # filter out empty blocks
    return [block for block in out_block_list if block]


def markup_table(linex_list: List[Line6WithAttrs]):
    lines_with_same_cols = find_lines_with_same_cols(linex_list)
    # only 1 column and is centered, most likely not a table
    # but still can be the first line of a table, but centered.  :-(
    if lines_with_same_cols and lines_with_same_cols[0].attrs.get('cn'):
        return []
    col_dividers = find_column_dividers(lines_with_same_cols)

    if len(col_dividers) > 2:
        chopped_divs = col_dividers[1:]
        for linex in linex_list:
            linex.add_table_columns(chopped_divs)
        return chopped_divs

    return []

"""
def markup_table_old(linex_list: List[Line6WithAttrs]):

    strlist_count_map = defaultdict(int)
    xstart_count_map = defaultdict(int)
    for linex in linex_list:
        num_str = len(linex.str_list)
        strlist_count_map[num_str] += 1
        for str6 in linex.str_list:
            xstart_count_map[str6.xstart] += 1

    # print("len(strlist_count_map) = {}".format(len(strlist_count_map)))

    is_table = False
    first_line = linex_list[0]
    expected_col_num = -1
    if first_line.attrs.get('table-header'):
        is_table = True
        expected_col_num = first_line.attrs.get('table-col-num', -1)
    elif len(strlist_count_map) == 1:  # all same number of column, eaiser
        # print("strlist_count_map[0] = {}".format(strlist_count_map[0]))
        if strlist_count_map[0] == 1:
            is_table = False
        else:  # there are more than 1 column
            is_table = True

    col_dividers = find_column_dividers(linex_list[:5])
    print('column dividers: {}'.format(col_dividers))
    
    num_line_in_block = len(linex_list)
    if is_table:  # likely a table
        column_xstart_list = []
        for xstart, count in xstart_count_map.items():
            if float(count) / num_line_in_block >= 0.4:
                column_xstart_list.append((count, xstart))

        # missed by just one
        if len(column_xstart_list) == expected_col_num -1:
            # add the last one from the first line, which is table-header
            column_xstart_list.append((1, first_line.str_list[-1].xstart))
        print("column_xstart_list = {}".format(column_xstart_list))
            
            
    return is_table
"""

def is_column_customer(linex: Line6WithAttrs):
    if (len(linex.str_list) >= 2 and linex.str_list[0].text.lower().startswith("customer") and
        linex.str_list[1].xstart >= 50):
        # print("is_column_customer_plus =========== {}".format(linex))
        return True
    return False

def is_column_country_address(linex: Line6WithAttrs):
    if (len(linex.str_list) >= 2 and linex.str_list[0].text.lower().startswith("country") and
        linex.str_list[1].text.lower().startswith("address")):
        return True
    return False

def is_column_discount_type(linex: Line6WithAttrs):
    if (len(linex.str_list) >= 2 and linex.str_list[0].text.lower().startswith("discount type")):
        return True
    return False

"""
def is_column_service_component(linex: Line6WithAttrs):
    if (len(linex.str_list) >= 2 and
        (linex.str_list[0].text.lower().startswith("service component") or
         linex.str_list[0].text.lower().startswith("prior to completion"))):
        return True
    return False

def is_column_service(linex: Line6WithAttrs):
    if (len(linex.str_list) >= 2 and
        linex.str_list[0].text.lower().startswith("service") and
        linex.str_list[1].text.lower().startswith("service")):
        return True
    return False
"""


def is_column_pricing_schedule(linex: Line6WithAttrs):
    if (len(linex.str_list) >= 2 and
        # for both 'Pricing Schedule Term' and
        # 'Pricing Schedule Term Start Date' 
        linex.str_list[0].text.lower().startswith("pricing schedule term")):
        return True
    return False        


def is_known_table_header(linex: Line6WithAttrs):
    if is_column_country_address(linex):
        return True
    if is_column_customer(linex):
        return True
    if is_column_discount_type(linex):
        return True
    #if is_column_service(linex):
    #    return True
    #if is_column_service_component(linex):
    #    return True
    
    return False

def is_known_table_header_to_sep(linex: Line6WithAttrs):
    if is_column_pricing_schedule(linex):
        # special case, don't want to use it for column divider computation
        # because it is a 2 column heading for a 3 column table
        return True
    return False


def fix_known_block_errors(block_list):
    # "custom * |\t| anything are new tables

    out_block_list = []
    is_to_merge_next_block = False
    for block in block_list:
        if not is_to_merge_next_block:
            cur_block = []
        for linex in block:
            # to separate out rows with 'pricing schedule' as its own
            # table because of different columns
            if is_known_table_header_to_sep(linex):
                linex.attrs['table-header'] = True
                # linex.attrs['table-col-num'] = len(linex.str_list)
                if cur_block:  # there is already something, do a break
                    out_block_list.append(cur_block)                    
                    cur_block = [linex]
                    # linex.attrs['is_merge_next_block'] = True
                    # is_to_merge_next_block = True
                else:
                    cur_block.append(linex)
                out_block_list.append(cur_block)
                cur_block = []
                is_to_merge_next_block = False  # regardless both case
            elif is_known_table_header(linex):
                linex.attrs['table-header'] = True
                # linex.attrs['table-col-num'] = len(linex.str_list)
                if cur_block:  # there is already something, do a break
                    out_block_list.append(cur_block)                    
                    cur_block = [linex]
                    # linex.attrs['is_merge_next_block'] = True
                else:
                    cur_block.append(linex)
                is_to_merge_next_block = True
            else:
                cur_block.append(linex)
                is_to_merge_next_block = False
        if not is_to_merge_next_block:
            if cur_block:
                out_block_list.append(cur_block)

    # now every line in a table is row1
    for block in out_block_list:
        first_line = block[0]
        if first_line.attrs.get('table-header'):
            for other_linex in block[1:]:
                other_linex.attrs['table-row-num'] = 1

    # for tables that should be adjacent, such as addresses blocks
    # without headers
    tmp_block_list = out_block_list
    out_block_list = []
    len_block_list = len(tmp_block_list)
    block_seq = 0
    while block_seq < len_block_list:
        block = tmp_block_list[block_seq]
        #if not block:
        #    print('block_empty..................................................')
        #    block_seq += 1
        #    continue
        
        first_line = block[0]
        # print('first_line = {}'.format(first_line))
        is_merged_address = False
        if first_line.attrs.get('table-header'):
            next_block = None
            while block_seq + 1 < len_block_list:
                last_block_line = tmp_block_list[block_seq][-1]
                next_block = tmp_block_list[block_seq + 1]
                if next_block and is_block_address(next_block):
                    for nlinex in next_block:
                        nlinex.attrs['table-row-num'] = last_block_line.attrs.get('table-row-num', 1) + 1
                    block.extend(next_block)
                    block_seq += 1
                    is_merged_address = True
                else:
                    break

        out_block_list.append(block)
        block_seq += 1

    return out_block_list

# 'By ....', so cannot really require ':;' at the end
ADDRESS_PREFIX_PAT = re.compile(r'\s*(attention|attn|by|city|country|email|name|phone|s[lt]reet address|title|zip|zip\s*code)\s*[:;]?', re.I)

def is_line_address_prefix(line: str):
    return ADDRESS_PREFIX_PAT.match(line)


def is_block_address(linex_list: List[Line6WithAttrs]):
    if linex_list and linex_list[0].attrs.get('table-header'):
        return False
    num_non_address_line = 0
    for linex in linex_list:
        for strx in linex.str_list:
            if is_line_address_prefix(linex.line_text):
                return True

        if linex.is_english:
            return False
        num_non_address_line += 1
        if num_non_address_line >= 5:
            return False
    return False

def print_blocks(msg: str, page_num: int, block_list: List[List[Line6WithAttrs]]):
    print("\n\n===== page #{}, {}".format(page_num, msg))
    for block_seq, block in enumerate(block_list, 0):

        print("\n  === block #{}".format(block_seq))

        is_table = block[0].table_cell_list

        for linex in block:
            print("  line6 {}".format(linex))

        if is_table:
            print()
            if block:
                print('col_dividers: {}'.format(block[0].col_dividers))
            for linex in block:
                print("  table.line {}".format(linex.tostr2()))


def add_doc_structure_to_page(apage: PageInfo6, pdf_txt_doc):
    num_line_in_page = len(apage.linex_list)
    page_num = apage.page_num
    prev_line_text = ''

    debug_mode = True

    block_list = separate_block_by_empty_lines(apage.linex_list)
    if debug_mode:
        print_blocks("after block_by_empty_lines", apage.page_num, block_list)

    # non_empty_block_list = [block for block in block_list if len(block) != 0]
    # non_empty_block_list = block_list

    block_list = separate_page_nums(block_list)
    block_list = separate_secheads(block_list)
    if debug_mode:
        print_blocks("after separate_page_nums", apage.page_num, block_list)

    # split tables with known table_headers
    block_list = fix_known_block_errors(block_list)
    if debug_mode:
        print_blocks("after fix_known_block_errors", apage.page_num, block_list)

    for block_seq, block in enumerate(block_list, 0):
        col_dividers = markup_table(block)

    # print("block_list = {}".format(len(block_list)))
    block_list = merge_adjacent_tables(block_list)

    # print("block_list2 = {}".format(len(block_list)))
    print_blocks("final xxxxxxxxxxxxxxxxxxxxxxx", apage.page_num, block_list)


def is_next_block_has_number(last_block_line_num, last_block_num_cols, next_block):
    first_line = next_block[0]
    if first_line.attrs.get('table-header'):
        return False
    if (first_line.page_line_num <= last_block_line_num + 2 and
        last_block_num_cols == len(first_line.col_dividers)):
        #for linex in next_block[:5]:
        #    if strutils.strlist_has_number([str6.text for str6 in linex.str_list]):
        return True
    return False


def separate_block_by_empty_lines(linex_list: List[Str6]):
    cur_block = []
    block_list = [cur_block]

    is_prev_line_empty = False
    for line_num, linex in enumerate(linex_list, 1):
        if linex.line_text:
            cur_block.append(linex)
            is_prev_line_empty = False
        else:  # not linex.line_text
            if is_prev_line_empty:
                pass
            else:  # prev_line not empty, and current line empty
                cur_block = []
                block_list.append(cur_block)
            is_prev_line_empty = True
    # filter out empty block
    return [block for block in block_list if block]




def merge_adjacent_tables(block_list):

    # for tables that should be adjacent, such as addresses blocks
    # without headers
    out_block_list = []
    len_block_list = len(block_list)
    block_seq = 0
    while block_seq < len_block_list:
        block = block_list[block_seq]
        #if not block:
        #    print('block_empty..................................................')
        #    block_seq += 1
        #    continue
        
        first_line = block[0]
        is_table = first_line.table_cell_list
        # print('first_line = {}'.format(first_line))
        if is_table:
            next_block = None
            while block_seq + 1 < len_block_list:
                last_block_line = block_list[block_seq][-1]
                next_block = block_list[block_seq + 1]
                if next_block and is_next_block_has_number(last_block_line.page_line_num, len(last_block_line.col_dividers), next_block):
                    for nlinex in next_block:
                        nlinex.attrs['table-row-num'] = last_block_line.attrs.get('table-row-num', 1) + 1
                    block.extend(next_block)
                    block_seq += 1
                else:
                    break

        out_block_list.append(block)
        block_seq += 1
        
    return out_block_list

IGNORE_LINE_LIST = [r'at&t and customer confidential information',
                    r'asap!',
                    r'page\s*\d+\s*of\s*\d+',
                    # this is the first|last page of
                    r'this is the \S+\s*page of.*']

ATT_PAGE_NUM_PAT = re.compile(r'^\s*({})\s*$'.format('|'.join(IGNORE_LINE_LIST)),
                              re.I)


def is_line_page_num(line: str) -> bool :
    return ATT_PAGE_NUM_PAT.match(line)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse a document into a document structure.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file', help='a file to be annotated')

    args = parser.parse_args()
    txt_fname = args.file

    work_dir = 'dir-work'
    pdf_txt_doc = parse_document(txt_fname, work_dir=work_dir)
    # to_paras_with_attrs(pdf_txt_doc, txt_fname, work_dir=work_dir)

    # pdf_txt_doc.print_debug_lines()
    # pdf_txt_doc.print_debug_blocks()

    # pdf_txt_doc.save_debug_pages(work_dir=work_dir, extension='.paged.debug.tsv')

    logging.info('Done.')
