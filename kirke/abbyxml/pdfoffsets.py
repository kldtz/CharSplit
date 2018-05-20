from collections import namedtuple, Counter, defaultdict
from functools import total_ordering
import os
import sys
from typing import Any, DefaultDict, Dict, List, TextIO, Tuple, Union

from kirke.docstruct import jenksutils, docstructutils
from kirke.utils import engutils, strutils

from kirke.utils.alignedstr import AlignedStrMapper


class AbbyLine:

    def __init__(self,
                 text: str,
                 attr_dict: Dict) -> None:
        self.num = -1
        self.text = text
        self.attr_dict = attr_dict

        # this links to PDFBox's offset
        self.span_list = []
        self.pbox_line_ids = []

        self.infer_attr_dict = {}

        # To map from any abby doc to pbox offset.
        # Will be set by synchronizer later
        self.abby_pbox_offset_mapper = None

    def __str__(self):
        return '{} [{}]'.format(self.infer_attr_dict, self.text)


class AbbyPar:

    def __init__(self,
                 ab_lines: List[AbbyLine],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_lines = ab_lines
        self.attr_dict = attr_dict

        self.infer_attr_dict = {}


class AbbyTextBlock:

    def __init__(self,
                 ab_pars: List[AbbyPar],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_pars = ab_pars
        self.attr_dict = attr_dict

        self.infer_attr_dict = {}


class AbbyTableBlock:

    def __init__(self,
                 table_block_num: int,
                 attr_dict: Dict) -> None:
        self.num = table_block_num
        self.attr_dict = attr_dict

        self.infer_attr_dict = {}

class AbbyPage:

    def __init__(self,
                 ab_text_blocks: List[AbbyTextBlock],
                 ab_table_blocks: List[AbbyTextBlock],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_text_blocks = ab_text_blocks
        self.ab_table_blocks = ab_table_blocks
        self.attr_dict = attr_dict

        self.infer_attr_dict = {}

def _is_par_centered(attr_dict: Dict) -> bool:
    for attr, val in attr_dict.items():
        if attr.startswith('Center'):
            return True
        if attr == 'align' and \
           val == 'Center':
            return True
    return False

def _pprint_line_attrs(attr_dict: Dict) -> str:
    st_list = []  # type: List[str]
    st_list.append('x={}'.format(attr_dict['x']))
    st_list.append('y={}'.format(attr_dict['y']))
    if attr_dict['ydiff'] != -1:
        st_list.append('ydf={}'.format(attr_dict['ydiff']))
    return ', '.join(st_list)

def _get_indent_level(attr_dict: Dict) -> int:
    for attr, val in attr_dict.items():
        if attr.startswith('indent_1'):
            return 1
        elif attr.startswith('indent_2'):
            return 2
    return 0


def _print_left_right_panes(rt_line: str,
                            line_attr_dict: Dict,
                            *,
                            par_id: int = -1,
                            line_id: int = -1,
                            is_header_footer: bool = False,
                            is_par_centered: bool = False,
                            indent_level: int = 0,
                            file: TextIO) -> None:

    left_line = 'par={}, ln={}'.format(par_id, line_id)

    if is_header_footer:
        left_line += ', HF'

    if indent_level != 0:
        left_line += ', I{}'.format(indent_level)

    if is_par_centered:
        left_line += ', CC'
        tmp_rt_line = '>> <<' + '     ' * 4 + '[' + rt_line + ']'
    elif is_header_footer:
        tmp_rt_line = '##### ' + '[' + rt_line + ']'
    else:
        tmp_rt_line = '|----' * indent_level + '[' + rt_line + ']'

    # 30 column for meta info
    print("    {:26}{}".format(left_line, tmp_rt_line), file=file)


class AbbyXmlDoc:

    def __init__(self,
                 file_name: str,
                 ab_pages: List[AbbyPage]) -> None:
        self.file_id = file_name
        self.ab_pages = ab_pages

    def print_raw(self):
        for pnum, abby_page in enumerate(self.ab_pages):
            print("\n\npage #{} ========== {}".format(pnum, abby_page.attr_dict))
            for bid, ab_text_block in enumerate(abby_page.ab_text_blocks):
                print("\n    block #{} -------- {}".format(bid, ab_text_block.attr_dict))
                for par_id, ab_par in enumerate(ab_text_block.ab_pars):
                    print("        par #{} {}".format(par_id, ab_par.attr_dict))
                    for lid, ab_line in enumerate(ab_par.ab_lines):
                        print("            line #{} [{}] {}".format(lid, ab_line.text, ab_line.attr_dict))
                        # print("            line #{} [{}]".format(lid, ab_line.text, ab_line.attr_dict))

    def print_raw_lines(self):
        for pnum, abby_page in enumerate(self.ab_pages):
            print("\n\npage #{} ========== {}".format(pnum, abby_page.attr_dict))
            for bid, ab_text_block in enumerate(abby_page.ab_text_blocks):
                print("\n    block #{} -------- {}".format(bid, ab_text_block.attr_dict))
                for par_id, ab_par in enumerate(ab_text_block.ab_pars):
                    print("\n        par #{} {}".format(par_id, ab_par.attr_dict))
                    for lid, ab_line in enumerate(ab_par.ab_lines):
                        # print("            line #{} [{}] {}".format(lid, ab_line.text, ab_line.attr_dict))
                        print("            line #{} [{}]".format(lid, ab_line.text, ab_line.attr_dict))


    def print_debug_text(self, file: TextIO = sys.stdout):
        for abby_page in self.ab_pages:
            if abby_page.num != 0:
                print('\n', file=file)
            print("page #{:2d} ============ {:95}{}    {}".format(abby_page.num,
                                                             '',
                                                             abby_page.infer_attr_dict,
                                                             abby_page.attr_dict), file=file)
            for ab_text_block in abby_page.ab_text_blocks:
                print("\n  block #{:3d} -------- {:95}{}    {}".format(ab_text_block.num,
                                                                    '',
                                                                    ab_text_block.infer_attr_dict,
                                                                    ab_text_block.attr_dict), file=file)
                for ab_par in ab_text_block.ab_pars:
                    print("\n    par #{:3d} {:104}{}    {}".format(ab_par.num,
                                                               '',
                                                               ab_par.infer_attr_dict,
                                                               ab_par.attr_dict), file=file)
                    for ab_line in ab_par.ab_lines:
                        print("      line #{} {:100} {}    {}".format(ab_line.num,
                                                                      '[' + ab_line.text + ']',
                                                                      ab_line.infer_attr_dict,
                                                                      ab_line.attr_dict), file=file)


    def print_infer_text(self, file: TextIO = sys.stdout):
        for abby_page in self.ab_pages:
            if abby_page.num != 0:
                print('\n', file=file)
            print("page #{:2d} ============ {:95}{}".format(abby_page.num, '', abby_page.infer_attr_dict), file=file)
            for ab_text_block in abby_page.ab_text_blocks:
                print("\n  block #{:3d} -------- {:95}{}".format(ab_text_block.num, '', ab_text_block.infer_attr_dict), file=file)
                for ab_par in ab_text_block.ab_pars:
                    print("\n    par #{:3} {:104}{}".format(ab_par.num, '', ab_par.infer_attr_dict), file=file)
                    for ab_line in ab_par.ab_lines:
                        print("      line #{} {:100} {}".format(ab_line.num, '[' + ab_line.text + ']', ab_line.infer_attr_dict), file=file)


    def print_text_with_meta(self, file: TextIO = sys.stdout):
        for abby_page in self.ab_pages:
            if abby_page.num != 0:
                print('\n', file=file)
            print("========= page  #{:3d} ========".format(abby_page.num), file=file)

            for ab_text_block in abby_page.ab_text_blocks:
                print("\n  ------- block #{:3d} --------".format(ab_text_block.num), file=file)

                is_header_footer = ab_text_block.infer_attr_dict.get('header', False) or \
                                   ab_text_block.infer_attr_dict.get('footer', False)

                for par_id, ab_par in enumerate(ab_text_block.ab_pars):
                    # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
                    print(file=file)
                    is_par_centered = _is_par_centered(ab_par.infer_attr_dict)
                    indent_level = _get_indent_level(ab_par.infer_attr_dict)

                    for lid, ab_line in enumerate(ab_par.ab_lines):
                        _print_left_right_panes(ab_line.text,
                                                ab_line.infer_attr_dict,
                                                par_id=ab_par.num,
                                                line_id=ab_line.num,
                                                is_header_footer=is_header_footer,
                                                indent_level=indent_level,
                                                is_par_centered=is_par_centered,
                                                file=file)
