#!/usr/bin/env python3

import argparse
import logging
from collections import defaultdict, OrderedDict
import pprint
import sys
import warnings
import re
import operator

import json

from typing import Any, DefaultDict, Dict, List, Optional, Union

# from kirke.abbyxml import AbbyLine, AbbyPar, AbbyTextBlock, AbbyTableBlock, AbbyXmlDoc
from kirke.abbyxml.pdfoffsets import AbbyCell, AbbyLine, AbbyPar, AbbyRow
from kirke.abbyxml.pdfoffsets import AbbyTextBlock, AbbyTableBlock, AbbyPage, AbbyXmlDoc
from kirke.abbyxml.pdfoffsets import print_text_block_meta, to_html_tables, block_to_text

from kirke.abbyxml import abbyutils

IS_DISPLAY_ATTRS = False
# IS_DISPLAY_ATTRS = True


def add_infer_line_attrs(attr_dict: Dict):
    infer_attr_dict = {}
    b_attr = attr_dict['@b']
    l_attr = attr_dict['@l']
    # if l_attr > 1800 and \
    #    b_attr < 350:
    #    infer_attr_dict['header'] = True
    infer_attr_dict['x'] = l_attr
    infer_attr_dict['y'] = b_attr
    return infer_attr_dict


def add_infer_par_attrs(attr_dict: Dict):
    infer_attr_dict = {}
    align_attr = attr_dict.get('@align')
    left_indent_attr = attr_dict.get('@leftIndent')
    start_indent_attr = attr_dict.get('@startIndent')
    if align_attr and align_attr != 'Justified':
        infer_attr_dict['align'] = align_attr

    if left_indent_attr and start_indent_attr:
        left_val = left_indent_attr
        start_val = start_indent_attr
        if left_val > 3000 and left_val < 6000:
            infer_attr_dict['indent_1'] = left_val
            infer_attr_dict['start_1'] = start_val
        if left_val > 9000 and left_val < 11000:
            infer_attr_dict['indent_2'] = left_val
            infer_attr_dict['start_2'] = start_val
        if left_val > 18000 and left_val < 21000:
            infer_attr_dict['Center_2.1'] = left_val
            infer_attr_dict['start_2.2'] = start_val
    elif left_indent_attr:
        left_val = left_indent_attr
        if left_val > 3000 and left_val < 6000:
            infer_attr_dict['indent_1.1'] = left_val
        if left_val > 9000 and left_val < 11000:
            infer_attr_dict['indent_2.1'] = left_val
        if left_val > 18000 and left_val < 21000:
            infer_attr_dict['Center2'] = left_val
    return infer_attr_dict


# only applicable to text blocks, not lines
def add_infer_header_footer(attr_dict: Dict) -> None:
    infer_attr_dict = {}
    b_attr = attr_dict['@b']
    l_attr = attr_dict['@l']
    t_attr = attr_dict['@t']
    r_attr = attr_dict['@r']
    align_attr = attr_dict.get('@align')
    if l_attr > 1800 and \
       b_attr < 350:
        infer_attr_dict['header'] = True
    if t_attr >= 3000:
        infer_attr_dict['footer'] = True
    if align_attr and align_attr != 'Justified':
        infer_attr_dict['align'] = align_attr
    return infer_attr_dict


def add_infer_text_block_attrs(text_block: AbbyTextBlock) -> None:
    par_with_indent_list = []
    indent_count_map = defaultdict(int)
    num_line = 0
    total_line_len = 0
    for ab_par in text_block.ab_pars:
        abbyutils.count_indent_attr(ab_par.infer_attr_dict, indent_count_map)
        for ab_line in ab_par.ab_lines:
            total_line_len += len(ab_line.text)
            num_line += 1
    avg_line_len = total_line_len / num_line
    # print("avg_line_len = {}".format(avg_line_len))

    num_indent_1 = indent_count_map['indent_1']
    num_indent_2 = indent_count_map['indent_2']
    num_indent = num_indent_1 + num_indent_2
    perc_indent_pars = num_indent / len(text_block.ab_pars)
    # print("perc_indent_pars = {}".format(perc_indent_pars))
    if num_indent_1 > 0 and num_indent_2 > 0 and \
       perc_indent_pars >= 0.75 and \
       avg_line_len < 50:
        text_block.infer_attr_dict['column_blobs'] = perc_indent_pars


def add_infer_table_block_attrs(table_block: AbbyTableBlock) -> None:
    pass


def parse_abby_line(ajson) -> AbbyLine:
    line_attr_dict = {}

    for attr, val in sorted(ajson.items()):
        if attr.startswith('@'):
            line_attr_dict[attr] = abbyutils.abby_attr_str_to_val(attr, val)

        if attr == 'formatting':
            abby_line = AbbyLine(val['#text'], line_attr_dict)
            abby_line.infer_attr_dict = add_infer_line_attrs(line_attr_dict)
            return abby_line

    raise ValueError


def parse_abby_cell(ajson) -> AbbyCell:
    cell_attr_dict = {}

    # print("\n\nparse_abby_cell:")
    # pprint.pprint(ajson)

    for attr, val in sorted(ajson.items()):
        if attr.startswith('@'):
            cell_attr_dict[attr] = abbyutils.abby_attr_str_to_val(attr, val)

        if attr == 'text':
            # print('\n    text_block ------ {}'.format(block_attr_dict))

            if isinstance(val, list):
                par_list = []
                for tmp_val in val:
                    par_list.extend(parse_abby_par(tmp_val))
            elif isinstance(val, dict):
                # print('par: {}'.format(val))
                par_list = parse_abby_par(val)
            else:
                raise ValueError

            abby_cell = AbbyCell(par_list, cell_attr_dict)
            # abby_cell.infer_attr_dict = add_infer_xxxattrs(cell_attr_dict)
            return abby_cell

    raise ValueError


def parse_abby_rows(ajson) -> List[AbbyRow]:

    ab_row_list = []  # List[AbbyRow]

    # print("\n\najson in parse_abby_row:")
    # pprint.pprint(ajson)
    # [{'cell': ...}, {'cell': }]

    for cell_dict in ajson:
        ab_cell_list = []  # type: List[AbbyCell]
        if isinstance(cell_dict, dict):
            # cell_attr, cell_val = cell_dict
            cell_val = cell_dict['cell']
            if isinstance(cell_val, list):
                for tmp_val in cell_val:
                    ab_cell_list.append(parse_abby_cell(tmp_val))
            elif isinstance(cell_val, dict):
                ab_cell_list.append(parse_abby_cell(cell_val))
            else:
                raise ValueError
        else:
            raise ValueError

        if ab_cell_list:
            ab_row = AbbyRow(ab_cell_list, {})
            ab_row_list.append(ab_row)

    return ab_row_list



def add_ydiffs_in_lines(ab_lines: List[AbbyLine]) -> None:
    prev_b_attr = ab_lines[0].attr_dict['@b']
    ab_lines[0].infer_attr_dict['ydiff'] = -1
    for ab_line in ab_lines[1:]:
        b_attr = ab_line.attr_dict['@b']
        ydiff = b_attr - prev_b_attr
        ab_line.infer_attr_dict['ydiff'] = ydiff
        prev_b_attr = b_attr

def add_ydiffs_in_text_block(ab_text_block: AbbyTextBlock) -> None:
    line_list = []  # type: List[AbbyLine]
    # collect all the abby_lines
    par_list = ab_text_block.ab_pars
    for par in par_list:
        for line in par.ab_lines:
            line_list.append(line)
    # calculate the ydiff for all the lines in this block
    add_ydiffs_in_lines(line_list)


# ajson is {'par': ... }
def parse_abby_par(ajson) -> List[AbbyPar]:
    par_attr_dict = {}
    par_json_list = []

    for attr, val in sorted(ajson.items()):
        # there is no attribute for 'par'
        # if attr.startswith('@'):
        #     par_attr_dict[attr] = abbyutils.abby_attr_str_to_val(attr, val)

        if attr == 'par':
            if isinstance(val, list):
                par_json_list.extend(val)
            elif isinstance(val, dict):
                par_json_list.append(val)

    ab_par_list = []  # type: List[AbbyLine]
    # print("        par_attrs: {}".format(par_attr_dict))
    for par_json in par_json_list:

        par_attr_dict = {}  # type: Dict
        ab_line_list = []  # type: List[AbbyLine]
        for attr, val in sorted(par_json.items()):
            if attr.startswith('@'):
                par_attr_dict[attr] = abbyutils.abby_attr_str_to_val(attr, val)

            if attr == 'line':
                # print('\n            par\t{}'.format(par_attr_dict))
                if isinstance(val, list):
                    for tmp_val in val:
                        abby_line = parse_abby_line(tmp_val)
                        ab_line_list.append(abby_line)
                else:
                    abby_line = parse_abby_line(val)
                    ab_line_list.append(abby_line)

        # it is possible that a par has no line
        # {'@b': '3425',
        #  '@blockType': 'Text',
        #  'region': ...
        #  'text': {'par': {'@lineSpacing': '-1'}}}
        if ab_line_list:
            abby_par = AbbyPar(ab_line_list, par_attr_dict)
            abby_par.infer_attr_dict = add_infer_par_attrs(par_attr_dict)
            ab_par_list.append(abby_par)

    return ab_par_list


def parse_abby_page(ajson) -> AbbyPage:
    text_block_jsonlist = []
    table_block_list = []

    # print("parse_abby_page")
    # print(ajson)

    if isinstance(ajson, dict):
        page_attr_dict = {}
        for attr, val in sorted(ajson.items()):
            if attr.startswith('@'):
                page_attr_dict[attr] = abbyutils.abby_attr_str_to_val(attr, val)

            if attr == 'block':
                if isinstance(val, list):
                    text_block_jsonlist.extend(val)
                elif isinstance(val, dict):
                    text_block_jsonlist.append(val)
    elif isinstance(ajson, list):
        raise ValueError
    else:
        raise ValueError

    # print("page_attrs: {}".format(page_attr_dict))

    # ab_text_block_list = []  # type: List[AbbyTextBlock]
    # ab_table_block_list = []  # type: List[AbbyTableBlock]
    ab_block_list = []  # type: List[Union[AbbyTableBlock, AbbyTextBlock]]
    prev_block_battr = -1
    for text_block in text_block_jsonlist:

        block_attr_dict = {}
        for attr, val in sorted(text_block.items()):
            if attr.startswith('@'):
                block_attr_dict[attr] = abbyutils.abby_attr_str_to_val(attr, val)
            # print("attr = [{}], val= [{}]".format(attr, val))
            if attr == '@blockType' and \
               val not in set(['Text', 'Table', 'Barcode']):
                continue

            # we will take both 'Text' and 'Barcode', not sure about 'Table' yet
            # for 'Barcode', all attribute b, t, etc will be 0
            if attr == 'text':
                # print('\n    text_block ------ {}'.format(block_attr_dict))

                if isinstance(val, list):
                    par_list = []
                    for tmp_val in val:
                        par_list.extend(parse_abby_par(tmp_val))
                elif isinstance(val, dict):
                    # print('par: {}'.format(val))
                    par_list = parse_abby_par(val)
                else:
                    raise ValueError

                if par_list:
                    text_block = AbbyTextBlock(par_list, block_attr_dict)
                    # maybe this should be performed later
                    text_block.infer_attr_dict = add_infer_header_footer(block_attr_dict)

                    block_battr = block_attr_dict['@b']
                    block_tattr = block_attr_dict['@t']
                    block_ydiff = block_tattr - prev_block_battr
                    if prev_block_battr != -1:
                        text_block.infer_attr_dict['ydiff'] = block_ydiff
                    prev_block_battr = block_battr

                    # try:
                    add_ydiffs_in_text_block(text_block)
                    #except Exception as exc:
                    #    print("text_block ============================")
                    #    print(text_block)

                    add_infer_text_block_attrs(text_block)
                    # ab_text_block_list.append(text_block)
                    ab_block_list.append(text_block)
            elif attr == 'row':
                if isinstance(val, list):
                    row_list = parse_abby_rows(val)
                elif isinstance(val, dict):  # val is a dictionary
                    # print('par: {}'.format(val))
                    row_list = parse_abby_rows([val])
                else:
                    raise ValueError

                if row_list:
                    table_block = AbbyTableBlock(row_list, block_attr_dict)
                    # maybe this should be performed later
                    table_block.infer_attr_dict = add_infer_header_footer(block_attr_dict)

                    block_battr = block_attr_dict['@b']
                    block_tattr = block_attr_dict['@t']
                    block_ydiff = block_tattr - prev_block_battr
                    if prev_block_battr != -1:
                        table_block.infer_attr_dict['ydiff'] = block_ydiff
                    prev_block_battr = block_battr

                    # try:
                    # add_ydiffs_in_block(table_block)
                    #except Exception as exc:
                    #    print("table_block ============================")
                    #    print(table_block)

                    add_infer_table_block_attrs(table_block)
                    # ab_table_block_list.append(table_block)
                    ab_block_list.append(table_block)


    apage = AbbyPage(ab_block_list, page_attr_dict)
    # apage.infer_attr_dict = infer_page_attrs(page_attr_dict)
    return apage


def docjson_to_abby_page_list(ajson) -> List[AbbyPage]:
    doc_val = ajson['document']
    page_val = doc_val['page']
    page_json_list = []
    if isinstance(page_val, dict):
        page_json_list.append(page_val)
    elif isinstance(page_val, list):
        for val in page_val:
            page_json_list.append(val)
    else:
        pass

    # This is the more concse version of lines below.
    # But, later is easier to print debug info for now.
    # ab_page_list = [parse_abby_page(page_json)
    #                 for page_json in page_json_list]
    ab_page_list = []
    for pcount, page_json in enumerate(page_json_list):
        # print("===== page seq: {} =====".format(pcount))
        # if pcount == 17:
        #     pprint.pprint(page_json)
        ab_page_list.append(parse_abby_page(page_json))

    return ab_page_list


def parse_document(file_name: str,
                   work_dir: str,
                   debug_mode: bool = False) \
                   -> AbbyXmlDoc:

    # pprint.pprint(ajson, width=140)
    ajson = abbyutils.abbyxml_to_json(file_name)

    ajson_fname = file_name.replace('.pdf.xml', '.pdf.json')
    with open(ajson_fname, 'wt') as fout:
        pprint.pprint(ajson, stream=fout)
        print('wrote {}'.format(ajson_fname))

    abby_page_list = docjson_to_abby_page_list(ajson)

    ab_xml_doc = AbbyXmlDoc(file_name, abby_page_list)

    tmp_file = file_name.replace('.pdf.xml', '.tmp')
    with open(tmp_file, 'wt') as fout:
        ab_xml_doc.print_debug_text(fout)
        print('wrote {}'.format(tmp_file))

    # adjust the blocks of document according to our interpretation
    # based what we have seen in contracts
    remake_abby_xml_doc(ab_xml_doc)

    # infer_header_footer_doc(ab_xml_doc)

    merge_haligned_block_as_table(ab_xml_doc)
    # set page number block number at the end
    set_abby_page_numbers(ab_xml_doc)

    return ab_xml_doc

    """
    li_map = defaultdict(int)
    left_indent_count_map = count_left_indent(ajson, li_map)
    print('left_indent_count_map:')
    pprint.pprint(OrderedDict(sorted(left_indent_count_map.items(), key=operator.itemgetter(1), reverse=True)))

    doc_attrs = defaultdict(int)
    print_text(ajson, doc_attrs)
    """

def set_abby_page_numbers(ab_doc: AbbyXmlDoc) -> None:
    block_id = 0
    par_id = 0
    lid = 0
    table_id = 0
    # Page number starts with 1, to be consistent with UI.
    for pnum, abby_page in enumerate(ab_doc.ab_pages, 1):
        # print("\n\npage #{} ========== {}".format(pnum, abby_page.infer_attr_dict))
        abby_page.num = pnum

        for ab_block in abby_page.ab_blocks:
            if isinstance(ab_block, AbbyTextBlock):
                abby_page.ab_text_blocks.append(ab_block)
            elif isinstance(ab_block, AbbyTableBlock):
                abby_page.ab_table_blocks.append(ab_block)
            else:
                raise ValueError

        for ab_block in abby_page.ab_blocks:
            # print("\n    block #{} -------- {}".format(bid, ab_text_block.infer_attr_dict))
            ab_block.num = block_id
            block_id += 1
            if isinstance(ab_block, AbbyTextBlock):
                ab_text_block = ab_block
                for ab_par in ab_text_block.ab_pars:
                    # print("        par #{} {}".format(par_id, ab_par.infer_attr_dict))
                    ab_par.num = par_id
                    par_id += 1
                    for ab_line in ab_par.ab_lines:
                        #print("            line #{} [{}] {}".format(lid, ab_line.text, ab_line.infer_attr_dict))
                        ab_line.num = lid
                        lid += 1
            elif isinstance(ab_block, AbbyTableBlock):
                ab_table_block = ab_block
                ab_table_block.table_id = table_id
                ab_table_block.page_num = pnum
                table_id += 1
            else:
                raise ValueError

        # infer_header_footer_blocks(abby_page)
        infer_header_footer_table_blocks(abby_page)


def split_indent_1_2(ab_text_block: AbbyTextBlock) -> None:
    """Group all indent_2 para into 1 par.
    """
    ab_par_list = ab_text_block.ab_pars
    out_par_list = []
    cur_line_list = []
    cur_attr_dict, cur_infer_attr_dict = {}, {}
    for ab_par in ab_par_list:
        if abbyutils.has_indent_2_attr(ab_par.infer_attr_dict):
            if cur_line_list:
                cur_line_list.extend(ab_par.ab_lines)
                cur_attr_dict['@b'] = ab_text_block.attr_dict['@b']
            else:
                cur_line_list = list(ab_par.ab_lines)
                cur_attr_dict = dict(ab_par.attr_dict)
                cur_infer_attr_dict = dict(ab_par.infer_attr_dict)
        else:
            if cur_line_list:
                tmp_par = AbbyPar(cur_line_list, cur_attr_dict)
                tmp_par.infer_attr_dict = cur_infer_attr_dict
                out_par_list.append(tmp_par)
                cur_line_list, cur_attr_dict, cur_infer_attr_dict = [], {}, {}
            out_par_list.append(ab_par)
    if cur_line_list:
        tmp_par = AbbyPar(cur_line_list, cur_attr_dict)
        tmp_par.infer_attr_dict = cur_infer_attr_dict
        out_par_list.append(tmp_par)

    ab_text_block.ab_pars = out_par_list

def get_text_block_num_words(ab_block: Union[AbbyTextBlock, AbbyTableBlock]) -> int:
    block_text = block_to_text(ab_block)
    words = block_text.split()
    return len(words)


def infer_header_footer_doc(ab_doc: AbbyXmlDoc) -> None:
    for ab_page in ab_doc.ab_pages:
        infer_header_footer_blocks(ab_page)


def infer_header_footer_blocks(ab_page: AbbyPage) -> None:
    for ab_block in ab_page.ab_blocks:
        battr = ab_block.attr_dict.get('@b', -1)
        tattr = ab_block.attr_dict.get('@t', -1)

        block_num_words = get_text_block_num_words(ab_block)

        # if tattr < 200:
        if tattr < 340:
            print("tattr < 340, tattr = {}".format(tattr))
            block_text = block_to_text(ab_block)
            print('block_num_words = {} [{}]'.format(block_num_words, block_text))

        if tattr < 340 and \
           block_num_words <= 20 and \
           battr < 500:
            ab_block.infer_attr_dict['header'] = True
        if tattr >= 3000:
            ab_block.infer_attr_dict['footer'] = True


def infer_header_footer_table_blocks(ab_page: AbbyPage) -> None:
    for ab_block in ab_page.ab_table_blocks:
        battr = ab_block.attr_dict.get('@b', -1)
        tattr = ab_block.attr_dict.get('@t', -1)

        block_num_words = get_text_block_num_words(ab_block)

        # if tattr < 200:
        if tattr < 340:
            print("\ntattr < 340, tattr = {}".format(tattr))
            block_text = block_to_text(ab_block)
            print('    block_num_words = {} [{}]'.format(block_num_words, block_text))

        if tattr < 340 and \
           block_num_words <= 120 and \
           battr < 400:  # each left, middle, right cell can contribute 40 char each
            ab_block.infer_attr_dict['header'] = True
            print("      set header true")
        if tattr >= 3000:
            ab_block.infer_attr_dict['footer'] = True
            print("      set footer true")


# there is probably a more concise way of expressing this in python, 5345
def block_get_attr_left(block: Union[AbbyTextBlock, AbbyTableBlock]) -> int:
    return block.attr_dict.get('@l', -1)


def merge_aligned_blocks(haligned_blocks: List[AbbyTextBlock]) -> AbbyTableBlock:
    # there is probably a more concise way of expressing this in python, 5345
    haligned_blocks = sorted(haligned_blocks, key=block_get_attr_left)
    cell_list = []
    is_header = False
    is_footer = False
    infer_attr_dict = {}
    for tblock in haligned_blocks:
        line_list = []  # type: List[AbbyLine]
        for par_id, ab_par in enumerate(tblock.ab_pars):
            line_list.extend(ab_par.ab_lines)
        apar = AbbyPar(line_list, {})
        cell_list.append(AbbyCell([apar], {}))

        if tblock.infer_attr_dict.get('header'):
            infer_attr_dict['header'] = True
        if tblock.infer_attr_dict.get('footer'):
            infer_attr_dict['footer'] = True

    attr_dict = {}
    attr_dict['@l'] = haligned_blocks[0].attr_dict['@l']
    attr_dict['@t'] = haligned_blocks[0].attr_dict['@t']
    attr_dict['@b'] = haligned_blocks[0].attr_dict['@b']
    attr_dict['@r'] = haligned_blocks[-1].attr_dict['@r']
    attr_dict['type'] = 'table-haligned'
    # we can combine block1+2's attr_dict
    row1 = AbbyRow(cell_list, {})

    table_block = AbbyTableBlock([row1], attr_dict)
    table_block.infer_attr_dict = infer_attr_dict

    return table_block


def is_top_bot_match(top: int,
                     bot: int,
                     prev_block_top: int,
                     prev_block_bot: int) -> bool:
    # this check for both top and bottom
    diff = 10
    return top - diff <= prev_block_top and \
           prev_block_top <= top + diff and \
           bot - diff <= prev_block_bot and \
           prev_block_bot <= bot + diff

    # this is only checking for top, seems to add too much
    # can get section heading...
    # diff = 10
    # return top - diff <= prev_block_top and \
    #        prev_block_top <= top + diff


def merge_haligned_block_as_table(ab_doc: AbbyXmlDoc) -> None:

    for pnum, abby_page in enumerate(ab_doc.ab_pages):

        # find all the blocks with similar @b and @t
        ab_text_block_list = [ab_block for ab_block in abby_page.ab_blocks
                              if isinstance(ab_block, AbbyTextBlock)]  # type: List[AbbyTextBlock]

        # print("\nmerge_adj_aligned_block_as_table")
        if not ab_text_block_list:
            continue

        haligned_blocks_list = []  # type: List[List[AbbyTextBlock]]
        # skip_blocks are the blocks that have already been found to be haligned
        skip_blocks = []  # type: List[AbbyTextBlock]
        for i, ab_text_block in enumerate(ab_text_block_list):
            if ab_text_block in skip_blocks:
                continue

            attr_dict = ab_text_block.attr_dict
            top = attr_dict['@t']
            bot = attr_dict['@b']
            cur_blocks = []  # type: List[AbbyTextBlock]
            # could sort by @b attribute first, but that might change the order in ab_text_block
            for other_text_block in ab_text_block_list[i+1:]:
                other_attr_dict = other_text_block.attr_dict
                other_top = other_attr_dict['@t']
                other_bot = other_attr_dict['@b']

                if is_top_bot_match(top, bot, other_top, other_bot):
                    """
                    print("  match prev block attr: {}".format(ab_text_block.attr_dict))

                    print("\nprev_block:")
                    print_text_block_meta(prev_block)
                    print("\ncur_block:")
                    print_text_block_meta(ab_text_block)
                    """
                    cur_blocks.append(other_text_block)

            if cur_blocks:
                skip_blocks.extend(cur_blocks)
                # add the original blocks at beginning
                this_cur_blocks = [ab_text_block]
                this_cur_blocks.extend(cur_blocks)
                haligned_blocks_list.append(this_cur_blocks)

        if not haligned_blocks_list:
            # No haligned blocks, no need to merge tables.
            # Move to next page
            continue

        out_block_list = []  # type: List[AbbyBlock]
        haligned_block_list_map = {}  # type: Dict[AbbyTextBlock, List[AbbyTextBlock]]
        for blocks in haligned_blocks_list:
            haligned_block_list_map[blocks[0]] = blocks

        # now we have a list of haligned blocks
        for ab_block in abby_page.ab_blocks:
            if ab_block in skip_blocks:
                continue

            if isinstance(ab_block, AbbyTextBlock):
                haligned_blocks = haligned_block_list_map.get(ab_block, [])

                if haligned_blocks:
                    table = merge_aligned_blocks(haligned_blocks)
                    out_block_list.append(table)
                else:
                    # print("      block attr: {}".format(ab_block.attr_dict))
                    out_block_list.append(ab_block)
            else:
                # for tables
                # print("      block attr: {}".format(ab_block.attr_dict))
                out_block_list.append(ab_block)

        abby_page.ab_blocks = out_block_list


def remake_abby_xml_doc(ab_doc: AbbyXmlDoc) -> None:
    """Infer as much as possible to make AbbyXmlDoc digestable with structure understanding
    """
    # figure out which blocks are
    #     - normal text
    #     - itemize list
    #     - 2 column blobs

    """
    for pnum, abby_page in enumerate(self.ab_pages):
        for bid, ab_text_block in enumerate(abby_page.ab_text_blocks):
            print("\n    block #{} -------- {}".format(bid, ab_text_block.infer_attr_dict))

            for par_id, ab_par in enumerate(ab_text_block.ab_pars):
                print("        par #{} {}".format(par_id, ab_par.infer_attr_dict))
                for lid, ab_line in enumerate(ab_par.ab_lines):
                    print("            line #{} [{}] {}".format(lid, ab_line.text, ab_line.infer_attr_dict))
    """

    # merge adjacent blocks
    is_merge_occurred = False
    for pnum, abby_page in enumerate(ab_doc.ab_pages):

        out_block_list = []
        # if not abby_page.ab_text_blocks:
        #     continue

        cur_par_list = []  # type: List[AbbyPar]
        cur_attr_dict = {}
        cur_infer_attr_dict = {}
        for ab_block in abby_page.ab_blocks:
            # skip table blocks
            if isinstance(ab_block, AbbyTextBlock):
                ab_text_block = ab_block

                if ab_text_block.infer_attr_dict.get('column_blobs'):
                    is_merge_occurred = True
                    if cur_par_list:  # already found column_blobs before
                        cur_par_list.extend(ab_text_block.ab_pars)
                        cur_attr_dict['@b'] = ab_text_block.attr_dict['@b']
                    else:  # first time
                        cur_par_list.extend(ab_text_block.ab_pars)
                        cur_attr_dict = dict(ab_text_block.attr_dict)  # make acopy
                        cur_infer_attr_dict = dict(ab_text_block.infer_attr_dict)  # make acopy
                else:
                    if cur_par_list:
                        tmp_text_block = AbbyTextBlock(cur_par_list, cur_attr_dict)
                        tmp_text_block.infer_attr_dict = cur_infer_attr_dict

                        split_indent_1_2(tmp_text_block)
                        # TODO, tmp_text_block.attr_dict and tmp_text_block.infer_attr_dict might
                        # not be the correct union of all the attributes, but just taking the
                        # first one is probably OK for now
                        out_block_list.append(tmp_text_block)
                        cur_par_list, cur_attr_dict, cur_infer_attr_dict = [], {}, {}

                        # add the current text block
                    out_block_list.append(ab_text_block)
            elif isinstance(ab_block, AbbyTableBlock):
                out_block_list.append(ab_block)
            else:
                raise ValueError

        # if the last block is 'column_blobs'
        if cur_par_list:
            tmp_text_block = AbbyTextBlock(cur_par_list, cur_attr_dict)
            tmp_text_block.infer_attr_dict = cur_infer_attr_dict

            split_indent_1_2(tmp_text_block)
            # TODO, tmp_text_block.attr_dict and tmp_text_block.infer_attr_dict might
            # not be the correct union of all the attributes, but just taking the
            # first one is probably OK for now
            out_block_list.append(tmp_text_block)


        if is_merge_occurred:
            abby_page.ab_blocks = out_block_list

def get_page_abby_lines(abby_page: AbbyPage) -> List[AbbyLine]:
    ab_line_list = []  # type: List[AbbyLine]
    for ab_block in abby_page.ab_text_blocks:
        for ab_par in ab_block.ab_pars:
            ab_line_list.extend(ab_par.ab_lines)

    for ab_block in abby_page.ab_table_blocks:
        for ab_row in ab_block.ab_rows:
            for ab_cell in ab_row.ab_cells:
                for ab_par in ab_cell.ab_pars:
                    ab_line_list.extend(ab_par.ab_lines)
    return ab_line_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    fname = args.file

    work_dir = 'dir-work'
    abbydoc = parse_document(fname, work_dir)

    # abbydoc.print_raw_lines()
    abbydoc.print_text_with_meta()


    # abbydoc.print_text()

    table_html_out_fn = fname.replace('.pdf.xml', '.html')
    with open(table_html_out_fn, 'wt') as fout:
        html_st = to_html_tables(abbydoc)
        print(html_st, file=fout)
    print('wrote "{}"'.format(table_html_out_fn))











