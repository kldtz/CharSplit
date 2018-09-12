#!/usr/bin/env python3

import argparse
from collections import defaultdict
import math
import os
import pprint
import shutil

# pylint: disable=unused-import
from typing import DefaultDict, Dict, List, Optional, Tuple

# from kirke.abbyyxml import AbbyLine, AbbyPar, AbbyTextBlock, AbbyTableBlock, AbbyyXmlDoc
from kirke.abbyyxml.pdfoffsets import AbbyyCell, AbbyyLine, AbbyyPar, AbbyyRow
from kirke.abbyyxml.pdfoffsets import AbbyyBlock, AbbyyTextBlock, AbbyyTableBlock
from kirke.abbyyxml.pdfoffsets import AbbyyPage, AbbyyXmlDoc
from kirke.abbyyxml.pdfoffsets import print_text_block_meta
from kirke.docstruct import linepos
from kirke.abbyyxml import abbyyutils, tableutils
from kirke.utils import mathutils

IS_DISPLAY_ATTRS = False
# IS_DISPLAY_ATTRS = True

# Normally, a page is resolution is 300 dots per intch
# Sometimes, the reoslution is changed because of images or
# pictures, such as signature.  Need to adjust accordingly

def adjust_by_resolution(coord: int, resolution: int) -> int:
    if resolution == 300:
        return coord
    multiplier = 300.0 / resolution
    # it is OK we don't do rounding here,
    # precision is not critical here
    return int(coord * multiplier)

def is_position_attr(attr):
    return attr in set(['@b', '@t', '@l', '@r',
                        '@baseline',
                        '@height',
                        '@width',
                        '@leftIndent',
                        '@rightIndent',
                        '@startIndent',
                        '@lineSpacing'])


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
def add_infer_header_footer(attr_dict: Dict) -> Dict:
    infer_attr_dict = {}  # type: Dict
    b_attr = attr_dict['@b']
    l_attr = attr_dict['@l']
    t_attr = attr_dict['@t']
    # pylint: disable=unused-variable
    r_attr = attr_dict['@r']
    align_attr = attr_dict.get('@align')
    if l_attr > 1800 and \
       b_attr < 350:
        infer_attr_dict['header'] = True
    if t_attr >= 3000:
         infer_attr_dict['footer'] = True
    # Due to document 367594.pdf
    # Must be only a small y span
    if b_attr >= 2850 and \
       b_attr - t_attr < 140:
        infer_attr_dict['footer'] = True

    if align_attr and align_attr != 'Justified':
        infer_attr_dict['align'] = align_attr
    return infer_attr_dict


def add_infer_text_block_attrs(text_block: AbbyyTextBlock) -> None:
    # par_with_indent_list = []
    indent_count_map = defaultdict(int)  # type: DefaultDict[str, int]
    num_line = 0
    total_line_len = 0
    all_ydiffs = []
    for ab_par in text_block.ab_pars:
        abbyyutils.count_indent_attr(ab_par.infer_attr_dict, indent_count_map)
        for ab_line in ab_par.ab_lines:
            total_line_len += len(ab_line.text)
            num_line += 1
            all_ydiffs.append(ab_line.infer_attr_dict['ydiff'])
    avg_line_len = total_line_len / num_line
    # print("avg_line_len = {}".format(avg_line_len))

    num_indent_1 = indent_count_map['indent_1']
    num_indent_2 = indent_count_map['indent_2']
    num_indent = num_indent_1 + num_indent_2
    perc_indent_pars = num_indent / len(text_block.ab_pars)
    # print("perc_indent_pars = {}".format(perc_indent_pars))
    text_block.infer_attr_dict['ydiff_min'] = min([diff for diff in all_ydiffs if diff > 0],
                                                  default=0)
    if num_indent_1 > 0 and num_indent_2 > 0 and \
       perc_indent_pars >= 0.75 and \
       avg_line_len < 50:
        text_block.infer_attr_dict['column_blobs'] = perc_indent_pars


# pylint: disable=unused-argument
def add_infer_table_block_attrs(table_block: AbbyyTableBlock) -> None:
    pass


def parse_abbyy_line(ajson, resolution: int) -> Optional[AbbyyLine]:
    line_attr_dict = {}

    for attr, val in sorted(ajson.items()):
        if attr.startswith('@'):
            line_attr_dict[attr] = abbyyutils.abbyy_attr_str_to_val(attr, val)
            if is_position_attr(attr):
                line_attr_dict[attr] = adjust_by_resolution(line_attr_dict[attr], resolution)

        if attr == 'formatting':
            # val can be a list, with dict with '#text' tags.  This is a mixed language.
            # This might happen when 'lang' is not specified to Abby?
            if isinstance(val, list):
                # take all the text from the list, but the space char between words between
                # dictionary are removed.
                # It is possible that there is no '#text' attribute!?
                text_list = [tmp_val['#text'] for tmp_val in val if tmp_val.get('#text')]
                # add spaces between words, assume our AligneStrMapper will resolve issues
                abbyy_line = AbbyyLine(' '.join(text_list), line_attr_dict)
                abbyy_line.infer_attr_dict = add_infer_line_attrs(line_attr_dict)
            else:
                if val.get('#text'):
                    abbyy_line = AbbyyLine(val['#text'], line_attr_dict)
                    abbyy_line.infer_attr_dict = add_infer_line_attrs(line_attr_dict)
                else:
                    # no '#text' attribute, in doc 367594.pdf
                    continue
            return abbyy_line
    # raise ValueError
    return None



def parse_abbyy_cell(ajson, resolution: int) -> AbbyyCell:
    cell_attr_dict = {}

    # print("\n\nparse_abbyy_cell:")
    # pprint.pprint(ajson)

    for attr, val in sorted(ajson.items()):
        if attr.startswith('@'):
            cell_attr_dict[attr] = abbyyutils.abbyy_attr_str_to_val(attr, val)
            if is_position_attr(attr):
                cell_attr_dict[attr] = adjust_by_resolution(cell_attr_dict[attr], resolution)

        if attr == 'text':
            # print('\n    text_block ------ {}'.format(block_attr_dict))

            if isinstance(val, list):
                par_list = []
                for tmp_val in val:
                    par_list.extend(parse_abbyy_par(tmp_val, resolution))
            elif isinstance(val, dict):
                # print('par: {}'.format(val))
                par_list = parse_abbyy_par(val, resolution)
            else:
                raise ValueError

            abbyy_cell = AbbyyCell(par_list, cell_attr_dict)
            # abbyy_cell.infer_attr_dict = add_infer_xxxattrs(cell_attr_dict)
            return abbyy_cell

    raise ValueError


def parse_abbyy_rows(ajson, resolution: int) -> List[AbbyyRow]:

    ab_row_list = []  # List[AbbyRow]

    # print("\n\najson in parse_abbyy_row:")
    # pprint.pprint(ajson)
    # [{'cell': ...}, {'cell': }]

    for cell_dict in ajson:
        ab_cell_list = []  # type: List[AbbyyCell]
        if isinstance(cell_dict, dict):
            # cell_attr, cell_val = cell_dict
            cell_val = cell_dict['cell']
            if isinstance(cell_val, list):
                for tmp_val in cell_val:
                    ab_cell_list.append(parse_abbyy_cell(tmp_val, resolution))
            elif isinstance(cell_val, dict):
                ab_cell_list.append(parse_abbyy_cell(cell_val, resolution))
            else:
                raise ValueError
        else:
            raise ValueError

        if ab_cell_list:
            ab_row = AbbyyRow(ab_cell_list, {})
            ab_row_list.append(ab_row)

    return ab_row_list



def add_ydiffs_in_lines(ab_lines: List[AbbyyLine]) -> None:
    prev_b_attr = ab_lines[0].attr_dict['@b']
    ab_lines[0].infer_attr_dict['ydiff'] = -1
    for ab_line in ab_lines[1:]:
        b_attr = ab_line.attr_dict['@b']
        ydiff = b_attr - prev_b_attr
        ab_line.infer_attr_dict['ydiff'] = ydiff
        prev_b_attr = b_attr

def add_ydiffs_in_text_block(ab_text_block: AbbyyTextBlock) -> None:
    line_list = []  # type: List[AbbyyLine]
    # collect all the abbyy_lines
    par_list = ab_text_block.ab_pars
    for par in par_list:
        for line in par.ab_lines:
            line_list.append(line)
    # calculate the ydiff for all the lines in this block
    add_ydiffs_in_lines(line_list)


# ajson is {'par': ... }
# pylint: disable=too-many-branches
def parse_abbyy_par(ajson, resolution: int) -> List[AbbyyPar]:
    par_json_list = []  # type: List

    for attr, val in sorted(ajson.items()):
        # there is no attribute for 'par'
        # if attr.startswith('@'):
        #     par_attr_dict[attr] = abbyyutils.abbyy_attr_str_to_val(attr, val)

        if attr == 'par':
            if isinstance(val, list):
                par_json_list.extend(val)
            elif isinstance(val, dict):
                par_json_list.append(val)

    ab_par_list = []  # type: List[AbbyyLine]
    # print("        par_attrs: {}".format(par_attr_dict))
    for par_json in par_json_list:

        par_attr_dict = {}  # type: Dict
        ab_line_list = []  # type: List[AbbyyLine]
        for attr, val in sorted(par_json.items()):
            if attr.startswith('@'):
                par_attr_dict[attr] = abbyyutils.abbyy_attr_str_to_val(attr, val)
                if is_position_attr(attr):
                    par_attr_dict[attr] = adjust_by_resolution(par_attr_dict[attr], resolution)

            if attr == 'line':
                # print('\n            par\t{}'.format(par_attr_dict))
                if isinstance(val, list):
                    for tmp_val in val:
                        abbyy_line = parse_abbyy_line(tmp_val, resolution)
                        if abbyy_line:
                            ab_line_list.append(abbyy_line)
                else:
                    abbyy_line = parse_abbyy_line(val, resolution)
                    if abbyy_line:
                        ab_line_list.append(abbyy_line)

        # it is possible that a par has no line
        # {'@b': '3425',
        #  '@blockType': 'Text',
        #  'region': ...
        #  'text': {'par': {'@lineSpacing': '-1'}}}
        if ab_line_list:
            abbyy_par = AbbyyPar(ab_line_list, par_attr_dict)
            abbyy_par.infer_attr_dict = add_infer_par_attrs(par_attr_dict)
            ab_par_list.append(abbyy_par)

    return ab_par_list


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def parse_abbyy_page(ajson) -> AbbyyPage:
    """Transfomr a page json into our own representation."""
    text_block_jsonlist = []
    # table_block_list = []

    # print("parse_abbyy_page")
    # print(ajson)

    # default resolution is 300 / inch
    # but can be changed by page attribute
    resolution = 300

    if isinstance(ajson, dict):
        page_attr_dict = {}
        for attr, val in sorted(ajson.items()):
            if attr.startswith('@'):
                page_attr_dict[attr] = abbyyutils.abbyy_attr_str_to_val(attr, val)

                if attr == '@resolution':
                    resolution = int(val)
                    # print('page resolution: {}'.format(resolution))

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
    ab_block_list = []  # type: List[AbbyyBlock]
    prev_block_battr = -1
    for text_block in text_block_jsonlist:

        block_attr_dict = {}
        for attr, val in sorted(text_block.items()):
            if attr.startswith('@'):
                block_attr_dict[attr] = abbyyutils.abbyy_attr_str_to_val(attr, val)
                if is_position_attr(attr):
                    block_attr_dict[attr] = adjust_by_resolution(block_attr_dict[attr], resolution)

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
                        par_list.extend(parse_abbyy_par(tmp_val, resolution))
                elif isinstance(val, dict):
                    # print('par: {}'.format(val))
                    par_list = parse_abbyy_par(val, resolution)
                else:
                    raise ValueError

                if par_list:
                    text_block = AbbyyTextBlock(par_list, block_attr_dict)
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
                    row_list = parse_abbyy_rows(val, resolution)
                elif isinstance(val, dict):  # val is a dictionary
                    # print('par: {}'.format(val))
                    row_list = parse_abbyy_rows([val], resolution)
                else:
                    raise ValueError

                if row_list:
                    table_block = AbbyyTableBlock(row_list, block_attr_dict)
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


    apage = AbbyyPage(ab_block_list, page_attr_dict)
    apage.is_multi_column = is_page_multi_column(ab_block_list)

    # apage.infer_attr_dict = infer_page_attrs(page_attr_dict)
    return apage


def is_page_multi_column(ab_block_list: List[AbbyyBlock]) -> bool:
    """Decide if a page is multi-column.

    If a page has more than 1/3 of it (in height) are in 2-column mode,
    then, it is a multi-column page.  A multi-column page doesn't undergo
    inferred table based on horizontal aligned blocks.

    If a page has 1/3 of the page in 2-column mode, the portion of
    single-column in height is 2/3.  The sum of the 2-column is also
    2/3 (or 1/3 * 2 for 2 such 2-column blocks).
    """
    num_big_block = 0
    num_2_col_block = 0
    col1_height_sum, col2_height_sum = 0, 0
    for ab_block in ab_block_list:
        block_attr = ab_block.attr_dict
        block_width = block_attr['@r'] - block_attr['@l']
        block_height = block_attr['@b'] - block_attr['@t']
        print("width = {}, h= {}, block r = {}, block l = {}".format(block_width,
                                                                     block_height,
                                                                     block_attr['@r'],
                                                                     block_attr['@l']))
        if block_height < 500:
            continue

        num_big_block += 1
        if block_width > 800 and block_width < 1400:
            num_2_col_block += 1
            col2_height_sum += block_height
        else:
            col1_height_sum += block_height

    if col2_height_sum == 0:
        col1_h_over_col2_h_ratio = 1.0
    else:
        col1_h_over_col2_h_ratio = col1_height_sum / col2_height_sum
    print("num_2_col_block = {}, num_big_block = {}, ratio = {}".format(
        num_2_col_block, num_big_block, col1_h_over_col2_h_ratio))
    if num_2_col_block >= 2 and col1_h_over_col2_h_ratio < 1.0:
        return True

    return False


def docjson_to_abbyy_page_list(ajson) -> List[AbbyyPage]:
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
    # ab_page_list = [parse_abbyy_page(page_json)
    #                 for page_json in page_json_list]
    ab_page_list = []
    for pcount, page_json in enumerate(page_json_list, 1):
        print("\n===== page seq: {} =====".format(pcount))
        abbyy_page = parse_abbyy_page(page_json)
        ab_page_list.append(abbyy_page)
        print('abby_page33 is_multi_column = {}'.format(abbyy_page.is_multi_column))

    return ab_page_list


def parse_document(file_name: str,
                   work_dir: str,
                   debug_mode: bool = False) \
                   -> AbbyyXmlDoc:

    base_fname = os.path.basename(file_name)
    xml_fname = '{}/{}'.format(work_dir, base_fname)
    if xml_fname != file_name:
        shutil.copy2(file_name, xml_fname)
    else:
        xml_fname = file_name

    # pprint.pprint(ajson, width=140)
    ajson = abbyyutils.abbyyxml_to_json(xml_fname)

    ajson_fname = '{}/{}'.format(work_dir, base_fname.replace('.pdf.xml', '.pdf.json'))
    with open(ajson_fname, 'wt') as fout:
        pprint.pprint(ajson, stream=fout)
        print('wrote {}'.format(ajson_fname))

    abbyy_page_list = docjson_to_abbyy_page_list(ajson)

    ab_xml_doc = AbbyyXmlDoc(xml_fname, abbyy_page_list)

    tmp_fname = '{}/{}'.format(work_dir, base_fname.replace('.pdf.xml', '.debug_txt'))
    with open(tmp_fname, 'wt') as fout:
        ab_xml_doc.print_debug_text(fout)
        print('wrote {}'.format(tmp_fname))

    # adjust the blocks of document according to our interpretation
    # based what we have seen in contracts
    remake_abbyy_xml_doc(ab_xml_doc)

    # jshaw, work
    # this is the more advanced version.
    # for now, focus on get the height right
    #
    # tableutils.find_haligned_blocks(ab_xml_doc)

    tableutils.merge_haligned_block_as_table(ab_xml_doc)

    abbyyutils.infer_header_footer_doc(ab_xml_doc)

    # set page number block number at the end
    # This also setup page.text_blocks, table_blocks, signature_blocks, address_blocks
    set_abbyy_page_numbers_tables(ab_xml_doc)

    return ab_xml_doc

    # pylint: disable=unreachable
    """
    li_map = defaultdict(int)
    left_indent_count_map = count_left_indent(ajson, li_map)
    print('left_indent_count_map:')
    pprint.pprint(OrderedDict(sorted(left_indent_count_map.items(),
                  key=operator.itemgetter(1), reverse=True)))

    doc_attrs = defaultdict(int)
    print_text(ajson, doc_attrs)
    """

def set_abbyy_page_numbers_tables(ab_doc: AbbyyXmlDoc) -> None:
    """This sets page numbers, and also ab_page.table_block, text_blocks, and signature blocks.
    """
    block_id = 0
    par_id = 0
    lid = 0
    table_id = 0
    # Page number starts with 1, to be consistent with UI.
    for pnum, abbyy_page in enumerate(ab_doc.ab_pages, 1):
        # print("\n\npage #{} ========== {}".format(pnum, abbyy_page.infer_attr_dict))
        abbyy_page.num = pnum

        for ab_block in abbyy_page.ab_blocks:
            if tableutils.is_signature_block(ab_block):
                abbyy_page.ab_signature_blocks.append(ab_block)
            elif tableutils.is_address_block(ab_block):
                abbyy_page.ab_address_blocks.append(ab_block)
            elif isinstance(ab_block, AbbyyTextBlock):
                abbyy_page.ab_text_blocks.append(ab_block)
            elif isinstance(ab_block, AbbyyTableBlock):
                abbyy_page.ab_table_blocks.append(ab_block)
            else:
                raise ValueError

        for ab_block in abbyy_page.ab_blocks:
            # print("\n    block #{} -------- {}".format(bid, ab_text_block.infer_attr_dict))
            ab_block.num = block_id
            block_id += 1
            if isinstance(ab_block, AbbyyTextBlock):
                ab_text_block = ab_block
                for ab_par in ab_text_block.ab_pars:
                    # print("        par #{} {}".format(par_id, ab_par.infer_attr_dict))
                    ab_par.num = par_id
                    par_id += 1
                    for ab_line in ab_par.ab_lines:
                        # print("            line #{} [{}] {}".format(lid,
                        #                    ab_line.text, ab_line.infer_attr_dict))
                        ab_line.num = lid
                        lid += 1
            elif isinstance(ab_block, AbbyyTableBlock):
                ab_table_block = ab_block
                ab_table_block.table_id = table_id
                ab_table_block.page_num = pnum
                table_id += 1
            else:
                raise ValueError


def split_indent_1_2(ab_text_block: AbbyyTextBlock) -> None:
    """Group all indent_2 para into 1 par.
    """
    ab_par_list = ab_text_block.ab_pars
    out_par_list = []  # type: List[AbbyyPar]
    cur_line_list = []  # type: List[AbbyyLine]
    cur_attr_dict, cur_infer_attr_dict = {}, {}  # type: Dict, Dict
    for ab_par in ab_par_list:
        if abbyyutils.has_indent_2_attr(ab_par.infer_attr_dict):
            if cur_line_list:
                cur_line_list.extend(ab_par.ab_lines)
                cur_attr_dict['@b'] = ab_text_block.attr_dict['@b']
            else:
                cur_line_list = list(ab_par.ab_lines)
                cur_attr_dict = dict(ab_par.attr_dict)
                cur_infer_attr_dict = dict(ab_par.infer_attr_dict)
        else:
            if cur_line_list:
                tmp_par = AbbyyPar(cur_line_list, cur_attr_dict)
                tmp_par.infer_attr_dict = cur_infer_attr_dict
                out_par_list.append(tmp_par)
                cur_line_list, cur_attr_dict, cur_infer_attr_dict = [], {}, {}
            out_par_list.append(ab_par)
    if cur_line_list:
        tmp_par = AbbyyPar(cur_line_list, cur_attr_dict)
        tmp_par.infer_attr_dict = cur_infer_attr_dict
        out_par_list.append(tmp_par)

    ab_text_block.ab_pars = out_par_list


def remake_abbyy_xml_doc(ab_doc: AbbyyXmlDoc) -> None:
    """Infer as much as possible to make AbbyyXmlDoc digestable with structure understanding
    """
    # figure out which blocks are
    #     - normal text
    #     - itemize list
    #     - 2 column blobs

    """
    for pnum, abbyy_page in enumerate(self.ab_pages):
        for bid, ab_text_block in enumerate(abbyy_page.ab_text_blocks):
            print("\n    block #{} -------- {}".format(bid, ab_text_block.infer_attr_dict))

            for par_id, ab_par in enumerate(ab_text_block.ab_pars):
                print("        par #{} {}".format(par_id, ab_par.infer_attr_dict))
                for lid, ab_line in enumerate(ab_par.ab_lines):
                    print("            line #{} [{}] {}".format(lid,
                                     ab_line.text, ab_line.infer_attr_dict))
    """

    # merge adjacent blocks
    is_merge_occurred = False
    for unused_pnum, abbyy_page in enumerate(ab_doc.ab_pages):

        out_block_list = []
        # if not abbyy_page.ab_text_blocks:
        #     continue

        cur_par_list = []  # type: List[AbbyyPar]
        cur_attr_dict = {}  # type: Dict
        cur_infer_attr_dict = {}  # type: Dict
        for ab_block in abbyy_page.ab_blocks:
            # skip table blocks
            if isinstance(ab_block, AbbyyTextBlock):
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
                        tmp_text_block = AbbyyTextBlock(cur_par_list, cur_attr_dict)
                        tmp_text_block.infer_attr_dict = cur_infer_attr_dict

                        split_indent_1_2(tmp_text_block)
                        # TODO, tmp_text_block.attr_dict and tmp_text_block.infer_attr_dict might
                        # not be the correct union of all the attributes, but just taking the
                        # first one is probably OK for now
                        out_block_list.append(tmp_text_block)
                        cur_par_list, cur_attr_dict, cur_infer_attr_dict = [], {}, {}

                        # add the current text block
                    out_block_list.append(ab_text_block)
            elif isinstance(ab_block, AbbyyTableBlock):
                out_block_list.append(ab_block)
            else:
                raise ValueError

        # if the last block is 'column_blobs'
        if cur_par_list:
            tmp_text_block = AbbyyTextBlock(cur_par_list, cur_attr_dict)
            tmp_text_block.infer_attr_dict = cur_infer_attr_dict

            split_indent_1_2(tmp_text_block)
            # TODO, tmp_text_block.attr_dict and tmp_text_block.infer_attr_dict might
            # not be the correct union of all the attributes, but just taking the
            # first one is probably OK for now
            out_block_list.append(tmp_text_block)


        if is_merge_occurred:
            abbyy_page.ab_blocks = out_block_list

# merges paragraphs within a block if the ydiff is roughly the block's minimum ydiff
def merge_block_paras(ab_pars: List[AbbyyPar],
                      block_min: int) \
                      -> List[List[AbbyyLine]]:
    merged_lines = []  # type: List[List[AbbyyLine]]
    i = 0
    while i < len(ab_pars):
        try:
            first_ydiff = ab_pars[i+1].ab_lines[0].infer_attr_dict['ydiff']
            if math.floor(first_ydiff / 10) <= round(block_min / 10) <= math.ceil(first_ydiff / 10):
                lines = ab_pars[i].ab_lines + ab_pars[i+1].ab_lines
                merged_lines.append(lines)
                i += 2
            else:
                merged_lines.append(ab_pars[i].ab_lines)
                i += 1
        except IndexError:
            merged_lines.append(ab_pars[i].ab_lines)
            i += 1
    return merged_lines


# creates a list of paragraph indices and paragraph attributes
def to_paras_with_attrs(abbyy_xml_doc: AbbyyXmlDoc,
                        file_name: str,
                        work_dir: str,
                        debug_mode: bool = False) \
                        -> Tuple[List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]],
                                            Dict]],
                                 str]:
    para_with_attrs = []  # type: List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]], Dict]]
    paraline_text = ''
    nlp_offset = 0
    # pylint: disable=too-many-nested-blocks
    for ab_page in abbyy_xml_doc.ab_pages:
        for ab_block in ab_page.ab_text_blocks:
            is_footer = ab_block.infer_attr_dict.get('footer', False)
            is_header = ab_block.infer_attr_dict.get('header', False)
            # weird pylint error on no-member, .text?? in xabline.text below
            # pylint: disable=line-too-long
            merged_paras = merge_block_paras(ab_block.ab_pars, ab_block.infer_attr_dict['ydiff_min'])  # type: List[List[AbbyyLine]]

            for ab_par_in_ablines in merged_paras:
                infer_attr_dict = {}
                # pylint: disable=line-too-long
                to_from_index_list = []  # type: List[Tuple[linepos.LnPos, linepos.LnPos]]

                para_lines = []  # type: List[str]
                for line in ab_par_in_ablines:
                    # convert to LnPos because that's expected further in the pipeline
                    if line.abbyy_pbox_offset_mapper:
                        from_lnpos = []  # type: List[linepos.LnPos]
                        line_offset = nlp_offset
                        # pylint: disable=invalid-name
                        for from_se in line.abbyy_pbox_offset_mapper.from_se_list:
                            from_lnpos.append(linepos.LnPos(from_se[0] + line_offset,
                                                            from_se[1] + line_offset,
                                                            line_num=line.num))
                            print('tttt {} {}, ({} {}): [{}]'.format(from_se[0],
                                                                     from_se[1],
                                                                     from_se[0] + line_offset,
                                                                     from_se[1] + line_offset,
                                                                     line.text[from_se[0]:from_se[1]]))
                            nlp_offset += (from_se[1] - from_se[0] + 1)
                            para_lines.append(line.text[from_se[0]:from_se[1]])
                        to_lnpos = []  # type: List[linepos.LnPos]
                        to_lnpos = [linepos.LnPos(to_se[0],
                                                  to_se[1],
                                                  line_num=line.num)
                                    for to_se in line.abbyy_pbox_offset_mapper.to_se_list]
                        zipped_lnpos = list(zip(to_lnpos, from_lnpos))
                        to_from_index_list.extend(zipped_lnpos)
                infer_attr_dict['footer'] = is_footer
                infer_attr_dict['header'] = is_header
                if to_from_index_list:
                    para_with_attrs.append((to_from_index_list, infer_attr_dict))
                    paraline_text += '\n'.join(para_lines) + '\n\n'
                    to_from_index_list = []
                    nlp_offset += 1

    return para_with_attrs, paraline_text


def get_page_abbyy_lines(abbyy_page: AbbyyPage) -> List[AbbyyLine]:
    ab_line_list = []  # type: List[AbbyyLine]
    for ab_block in abbyy_page.ab_text_blocks:
        for ab_par in ab_block.ab_pars:
            ab_line_list.extend(ab_par.ab_lines)

    for ab_block in abbyy_page.ab_table_blocks:
        for ab_row in ab_block.ab_rows:
            for ab_cell in ab_row.ab_cells:
                for ab_par in ab_cell.ab_pars:
                    ab_line_list.extend(ab_par.ab_lines)
    return ab_line_list


def main():
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
        html_st = tableutils.to_html_tables(abbydoc)
        print(html_st, file=fout)
    print('wrote "{}"'.format(table_html_out_fn))


if __name__ == '__main__':
    main()
