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

from typing import Any, DefaultDict, Dict, List

# from kirke.abbyxml import AbbyLine, AbbyPar, AbbyTextBlock, AbbyTableBlock, AbbyXmlDoc
from kirke.abbyxml.pdfoffsets import AbbyLine, AbbyPar, AbbyTextBlock, AbbyTableBlock, AbbyPage, AbbyXmlDoc

from kirke.abbyxml import abbyutils

IS_DISPLAY_ATTRS = False
# IS_DISPLAY_ATTRS = True

def add_infer_attrs(attr_dict: Dict):
    infer_attr_dict = {}
    b_attr = int(attr_dict['@b'])
    l_attr = int(attr_dict['@l'])
    # if l_attr > 1800 and \
    #    b_attr < 350:
    #    infer_attr_dict['header'] = True
    return infer_attr_dict

def add_infer_par_attrs(attr_dict: Dict):
    infer_attr_dict = {}    
    align_attr = attr_dict.get('@align')
    left_indent_attr = attr_dict.get('@leftIndent')
    start_indent_attr = attr_dict.get('@startIndent')    
    if align_attr and align_attr != 'Justified':
        infer_attr_dict['align'] = align_attr

    if left_indent_attr and start_indent_attr:
        left_val = int(left_indent_attr)
        start_val = int(start_indent_attr)
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
        left_val = int(left_indent_attr)
        if left_val > 3000 and left_val < 6000:
            infer_attr_dict['indent_1.1'] = left_val
        if left_val > 9000 and left_val < 11000:
            infer_attr_dict['indent_2.1'] = left_val
        if left_val > 18000 and left_val < 21000:
            infer_attr_dict['Center2'] = left_val
    return infer_attr_dict

# only applicable to blocks, not lines
def add_infer_header_footer(attr_dict: Dict):
    infer_attr_dict = {}
    b_attr = int(attr_dict['@b'])
    l_attr = int(attr_dict['@l'])
    t_attr = int(attr_dict['@t'])
    r_attr = int(attr_dict['@r'])    
    align_attr = attr_dict.get('@align')
    if l_attr > 1800 and \
       b_attr < 350:
        infer_attr_dict['header'] = True
    if t_attr >= 3000:
        infer_attr_dict['footer'] = True
    if align_attr and align_attr != 'Justified':
        infer_attr_dict['align'] = align_attr
    return infer_attr_dict



def parse_abby_line(ajson) -> AbbyLine:
    line_attr_dict = {}

    for attr, val in sorted(ajson.items()):
        if attr.startswith('@'):
            line_attr_dict[attr] = val

        if attr == 'formatting':
            abby_line = AbbyLine(val['#text'], line_attr_dict)
            return abby_line

    raise ValueError

# ajson is {'par': ... }
def parse_abby_par(ajson) -> List[AbbyPar]:
    par_attr_dict = {}
    par_json_list = []
    for attr, val in sorted(ajson.items()):
        # there is no attribute for 'par'
        # if attr.startswith('@'):
        #     par_attr_dict[attr] = val

        if attr == 'par':
            if isinstance(val, list):
                par_json_list.extend(val)
            elif isinstance(val, dict):
                par_json_list.append(val)

    ab_par_list = []  # type: List[AbbyLine]                
    # print("        par_attrs: {}".format(par_attr_dict))
    for par_json in par_json_list:

        par_attr_dict = {}
        aline_list = []  # type: List[AbbyLine]
        for attr, val in sorted(par_json.items()):
            if attr.startswith('@'):
                par_attr_dict[attr] = val            

            if attr == 'line':
                # print('\n            par\t{}'.format(par_attr_dict))
                if isinstance(val, list):
                    for tmp_val in val:
                        abby_line = parse_abby_line(tmp_val)
                        aline_list.append(abby_line)
                else:
                    abby_line = parse_abby_line(val)
                    aline_list.append(abby_line)                        

        abby_par = AbbyPar(aline_list, par_attr_dict)
        abby_par.infer_attr_dict = add_infer_par_attrs(par_attr_dict)
        ab_par_list.append(abby_par)

    return ab_par_list


def parse_abby_page(ajson) -> AbbyPage:
    text_block_jsonlist = []
    table_block_list = []
    if isinstance(ajson, dict):
        page_attr_dict = {}
        for attr, val in sorted(ajson.items()):
            if attr.startswith('@'):
                page_attr_dict[attr] = val

            if attr == 'block':
                if isinstance(val, list):
                    text_block_jsonlist.extend(val)
                elif isinstance(val, dict):
                    text_block_jsonlist.append(val)
    # print("page_attrs: {}".format(page_attr_dict))

    ab_text_block_list = []  # type: List[AbbyTextBlock]
    ab_table_block_list = []  # type: List[AbbyTableBlock]
    for text_block in text_block_jsonlist:

        block_attr_dict = {}
        for attr, val in sorted(text_block.items()):
            if attr.startswith('@'):
                block_attr_dict[attr] = val            
            # print("attr = [{}], val= [{}]".format(attr, val))
            if attr == '@blockType' and \
               val not in set(['Text', 'Table']):
                continue

            if attr == 'text':
                # print('\n    text_block ------ {}'.format(block_attr_dict))

                # print('par: {}'.format(val))
                par_list = parse_abby_par(val)
                text_block = AbbyTextBlock(par_list, block_attr_dict)
                text_block.infer_attr_dict = add_infer_header_footer(block_attr_dict)
                ab_text_block_list.append(text_block)

    apage = AbbyPage(ab_text_block_list, ab_table_block_list, page_attr_dict)
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

    ab_page_list = [parse_abby_page(page_json)
                    for page_json in page_json_list]
    
    return ab_page_list


def parse_document(file_name: str,
                   work_dir: str,
                   debug_mode: bool = False) \
                   -> AbbyXmlDoc:

    # pprint.pprint(ajson, width=140)
    ajson = abbyutils.abbyxml_to_json(file_name)

    abby_page_list = docjson_to_abby_page_list(ajson)

    ab_xml_doc = AbbyXmlDoc(file_name, abby_page_list)

    return ab_xml_doc

    """
    li_map = defaultdict(int)
    left_indent_count_map = count_left_indent(ajson, li_map)
    print('left_indent_count_map:')
    pprint.pprint(OrderedDict(sorted(left_indent_count_map.items(), key=operator.itemgetter(1), reverse=True)))
    
    doc_attrs = defaultdict(int)
    print_text(ajson, doc_attrs)
    """
                   

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
    abbydoc.print_text()

    # abbydoc.print_text()
    












