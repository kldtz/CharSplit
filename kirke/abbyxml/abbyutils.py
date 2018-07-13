import json
from typing import Any, DefaultDict, Dict, List, Tuple, Union

import xmltodict

from kirke.abbyxml.pdfoffsets import AbbyTextBlock, AbbyTableBlock, AbbyXmlDoc
# from kirke.abbyxml.pdfoffsets import AbbyPar, AbbyLine, AbbyCell, AbbyRow


ABBY_TEXT_ATTR_SET = set(['@align',
                          '@blockType',
                          '@lang',
                          '@languages',
                          '@producer',
                          '@version',
                          '@xmlns',
                          '@xmlns:xsi',
                          '@xsi:schemaLocation',
                          '@bottomBorder',
                          '@topBorder',
                          '@rightBorder',
                          '@leftBorder'])

def abby_attr_str_to_val(attr: str, val: str) -> Union[str, int]:
    # print('abby_attr_str_to_val({}, {})'.format(attr, val))
    if attr in ABBY_TEXT_ATTR_SET:
        return val
    return int(val)
    # try:
    #    return int(val)
    # except Exception as e:
    ##    pass
    # return val


def count_left_indent(ajson, li_map: Dict) -> Dict:
    if isinstance(ajson, dict):
        attr_dict = {}
        for attr, val in sorted(ajson.items()):
            if attr.startswith('@'):
                attr_dict[attr] = val

        if attr == 'line':
            left_indent_attr = attr_dict.get('@leftIndent')
            if left_indent_attr:
                int_val = int(left_indent_attr)
                li_map[int_val] += 1
        else:
            count_left_indent(val, li_map)
    elif isinstance(ajson, list):
        for val in ajson:
            count_left_indent(val, li_map)
    else:
        pass

    return li_map


# Dict is json
def abbyxml_to_json(file_name: str) -> Dict:
    with open(file_name) as fd:
        xdoc_dict = xmltodict.parse(fd.read())
    ajson = json.loads(json.dumps(xdoc_dict))
    return ajson


def count_indent_attr(attr_dict: Dict, count_dict: Dict[str, int]) -> bool:
    for attr, val in attr_dict.items():
        # if attr.startswith('indent'):
        if attr.startswith('indent_1'):
            count_dict['indent_1'] += 1
        if attr.startswith('indent_2'):
            count_dict['indent_2'] += 1
        if attr.startswith('no_indent'):
            count_dict['no_indent'] += 1


def has_indent_1_attr(attr_dict: Dict) -> bool:
    for attr, val in attr_dict.items():
        if attr.startswith('indent_1'):
            return True
    return False


def has_indent_2_attr(attr_dict: Dict) -> bool:
    for attr, val in attr_dict.items():
        if attr.startswith('indent_2'):
            return True
    return False


def find_text_block_minmaxy(ab_text_block: AbbyTextBlock) -> Tuple[int, int]:
    attr_dict = ab_text_block.attr_dict
    top = attr_dict['@t']
    bot = attr_dict['@b']
    return top, bot


def find_table_block_minmaxy(ab_table_block: AbbyTableBlock) -> Tuple[int, int]:
    attr_dict = ab_table_block.attr_dict
    top = attr_dict['@t']
    bot = attr_dict['@b']
    return top, bot


def find_block_minmaxy(ab_block: Union[AbbyTableBlock, AbbyTextBlock]) -> Tuple[int, int]:
    if isinstance(ab_block, AbbyTextBlock):
        return find_text_block_minmaxy(ab_block)
    return find_table_block_minmaxy(ab_block)


def print_text_block_first_line(ab_text_block: AbbyTextBlock) -> None:

    for unused_par_id, ab_par in enumerate(ab_text_block.ab_pars):
        for unused_lid, ab_line in enumerate(ab_par.ab_lines):
            print('    para [{}...]'.format(ab_line.text[:30]))
            break


def print_table_block_first_line(ab_table_block: AbbyTableBlock) -> None:

    for unused_row_id, ab_row in enumerate(ab_table_block.ab_rows):
        for unused_cell_seq, ab_cell in enumerate(ab_row.ab_cells):
            for ab_par in ab_cell.ab_pars:
                for unused_lid, ab_line in enumerate(ab_par.ab_lines):
                    print("    cell [{}]".format(ab_line.text[:30]))
                    break
                break
            break


def text_block_to_text(ab_text_block: AbbyTextBlock) -> str:

    st_list = []  # type: List[str]
    for par_id, ab_par in enumerate(ab_text_block.ab_pars):
        for lid, ab_line in enumerate(ab_par.ab_lines):
            st_list.append(ab_line.text)

    return '\n'.join(st_list)


def table_block_to_text(ab_table_block: AbbyTableBlock) -> str:
    st_list = []  # type: List[str]
    for row_id, ab_row in enumerate(ab_table_block.ab_rows):
        for cell_seq, ab_cell in enumerate(ab_row.ab_cells):
            for ab_par in ab_cell.ab_pars:
                for lid, ab_line in enumerate(ab_par.ab_lines):
                    st_list.append(ab_line.text)

    return '\n'.join(st_list)


def block_to_text(ab_block: Union[AbbyTableBlock, AbbyTextBlock]) -> str:
    if isinstance(ab_block, AbbyTextBlock):
        return text_block_to_text(ab_block)
    return table_block_to_text(ab_block)


def get_text_block_num_words(ab_block: Union[AbbyTextBlock, AbbyTableBlock]) -> int:
    block_text = block_to_text(ab_block)
    words = block_text.split()
    return len(words)


def get_only_text_blocks(ab_blocks: List[Union[AbbyTableBlock, AbbyTextBlock]]) \
    -> List[AbbyTextBlock]:
    return [ab_block for ab_block in ab_blocks
            if isinstance(ab_block, AbbyTextBlock)]

def get_only_table_blocks(ab_blocks: List[Union[AbbyTableBlock, AbbyTextBlock]]) \
    -> List[AbbyTableBlock]:
    return [ab_block for ab_block in ab_blocks
            if isinstance(ab_block, AbbyTableBlock)]


# text was using battr < 500
# table was using battr < 400, go with text
"""
def infer_table_block_is_header_footer(ab_block: AbbyTableBlock) -> None:
    battr = ab_block.attr_dict.get('@b', -1)
    tattr = ab_block.attr_dict.get('@t', -1)

    block_num_words = get_text_block_num_words(ab_block)

    # if tattr < 200:
    if tattr < 340:
        # print("\ntattr < 340, tattr = {}".format(tattr))
        block_text = block_to_text(ab_block)
        # print('    block_num_words = {} [{}]'.format(block_num_words, block_text))

    if tattr < 340 and \
       block_num_words <= 120 and \
       battr < 400:  # each left, middle, right cell can contribute 40 char each
        ab_block.infer_attr_dict['header'] = True
        # print("      set header true")
    if tattr >= 3000:
        ab_block.infer_attr_dict['footer'] = True
        # print("      set footer true")
"""


def infer_ab_block_is_header_footer(ab_block: Union[AbbyTableBlock, AbbyTextBlock]) -> None:
    battr = ab_block.attr_dict.get('@b', -1)
    tattr = ab_block.attr_dict.get('@t', -1)

    block_num_words = get_text_block_num_words(ab_block)

    # if tattr < 200:
    if tattr < 340:
        # print("tattr < 340, tattr = {}".format(tattr))
        block_text = block_to_text(ab_block)
        # print('block_num_words = {} [{}]'.format(block_num_words, block_text))

    if isinstance(ab_block, AbbyTextBlock):
        if tattr < 340 and \
           block_num_words <= 20 and \
           battr < 500:
            ab_block.infer_attr_dict['header'] = True
    elif isinstance(ab_block, AbbyTableBlock):
        if tattr < 340 and \
           block_num_words <= 120 and \
           battr < 400:  # each left, middle, right cell can contribute 40 char each
            ab_block.infer_attr_dict['header'] = True
            # print("      set header true")
    if tattr >= 3000:
        ab_block.infer_attr_dict['footer'] = True


def infer_header_footer_doc(ab_doc: AbbyXmlDoc) -> None:
    for ab_page in ab_doc.ab_pages:
        for ab_block in ab_page.ab_text_blocks:
            abbyutils.infer_ab_block_is_header_footer(ab_block)
        for ab_block in ab_page.ab_table_blocks:
            abbyutils.infer_ab_block_is_header_footer(ab_block)


