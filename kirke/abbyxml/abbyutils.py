import json
from typing import Any, DefaultDict, Dict, List, Tuple, Union

import xmltodict

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
