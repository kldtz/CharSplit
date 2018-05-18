import json
from typing import Dict

import xmltodict

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



