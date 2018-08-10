import re
import sys
# pylint: disable=unused-import
from typing import Dict, List, Optional, TextIO, Tuple, Union

from kirke.utils.alignedstr import AlignedStrMapper


# pylint: disable=too-few-public-methods
class AbbyyLine:

    def __init__(self,
                 text: str,
                 attr_dict: Dict) -> None:
        self.num = -1
        # These characters, for abby, it's the 2nd char
        # for pdfbox, it's the first.  They mismatch.
        # Replacing both here.
        # self.text = text
        # for special hypen char
        if re.search('№', text):
            text = re.sub('№', 'No', text)

        if re.search('[­¬]', text):
            self.text = re.sub('[­¬]', '-', text)
        else:
            self.text = text

        self.attr_dict = attr_dict

        # this links to PDFBox's offset
        self.span_list = []
        self.pbox_line_ids = []

        self.infer_attr_dict = {}

        # To map from any abby doc to pbox offset.
        # Will be set by synchronizer later
        self.abbyy_pbox_offset_mapper = None  # type: Optional[AlignedStrMapper]

    def __str__(self):
        return '{} [{}]'.format(self.infer_attr_dict, self.text)

    def to_debug_str(self):
        return '{:80}{}    {}'.format('[' + self.text + ']',
                                      self.infer_attr_dict,
                                      self.attr_dict)

# pylint: disable=too-few-public-methods
class AbbyyPar:

    def __init__(self,
                 ab_lines: List[AbbyyLine],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_lines = ab_lines
        self.attr_dict = attr_dict

        self.infer_attr_dict = {}


# pylint: disable=too-few-public-methods
class AbbyyTextBlock:

    def __init__(self,
                 ab_pars: List[AbbyyPar],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_pars = ab_pars
        self.attr_dict = attr_dict
        self.infer_attr_dict = {}

        # for indexining into page's ab_blocks
        self.page_block_seq = -1


    def __str__(self) -> str:
        st_list = []  # type: List[str]
        st_list.append("block #{:3d} -------- {:95}{}    {}".format(self.num,
                                                                    '',
                                                                    self.infer_attr_dict,
                                                                    self.attr_dict))
        for ab_par in self.ab_pars:
            st_list.append("\n  par #{:3d} {:104}{}    {}".format(ab_par.num,
                                                                  '',
                                                                  ab_par.infer_attr_dict,
                                                                  ab_par.attr_dict))
            for ab_line in ab_par.ab_lines:
                st_list.append("    line #{} {:100} {}    {}".format(ab_line.num,
                                                                     '[' + ab_line.text + ']',
                                                                     ab_line.infer_attr_dict,
                                                                     ab_line.attr_dict))
        return '\n'.join(st_list)


# pylint: disable=too-few-public-methods
class AbbyyCell:

    def __init__(self,
                 ab_pars: List[AbbyyPar],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_pars = ab_pars
        self.attr_dict = attr_dict
        self.infer_attr_dict = {}


# pylint: disable=too-few-public-methods
class AbbyyRow:

    def __init__(self,
                 ab_cells: List[AbbyyCell],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_cells = ab_cells
        self.attr_dict = attr_dict
        self.infer_attr_dict = {}


# pylint: disable=too-few-public-methods, too-many-instance-attributes
class AbbyyTableBlock:

    def __init__(self,
                 ab_rows: List[AbbyyRow],
                 attr_dict: Dict,
                 is_abbyy_original: bool = True) -> None:
        self.num = -1
        self.ab_rows = ab_rows
        self.attr_dict = attr_dict
        self.infer_attr_dict = {}
        self.table_id = -1
        self.page_num = -1
        self.is_abbyy_original = is_abbyy_original

        # for indexining into page's ab_blocks
        self.page_block_seq = -1


# pylint: disable=invalid-name
AbbyyBlock = Union[AbbyyTableBlock, AbbyyTextBlock]


"""
class UnmatchedAbbyLine:

    def __init__(self,
                 ab_line: AbbyLine,
                 page_num: int) -> None:
        self.ab_line = ab_line
        self.page_num = page_num

    def __str__(self):
        return str((self.page_num, str(self.ab_line)))
"""

class UnsyncedPBoxLine:

    def __init__(self,
                 xy_pair: Tuple[int, int],
                 se_pair: Tuple[int, int],
                 text: str) -> None:
        self.xy_pair = xy_pair
        self.se_pair = se_pair
        self.text = text

    def __str__(self) -> str:
        return 'xy={}, se={}, text=[{}]'.format(self.xy_pair,
                                                self.se_pair,
                                                self.text)

    def to_tuple(self) -> Tuple[Tuple[int, int],
                                Tuple[int, int],
                                str]:
        return self.xy_pair, self.se_pair, self.text


class UnsyncedStrWithY:

    def __init__(self,
                 y_val: int,
                 se_pair: Tuple[int, int],
                 text: str,
                 as_mapper: AlignedStrMapper) -> None:
        self.y_val = y_val
        # because sometimes text starts with space or underline, we
        # auto increment start index to avoid such junk
        """
        if len(text) > 2 and is_hyphen_underline(text[0]) and \
           not is_hyphen_underline(text[1]):
            self.se_pair = (se_pair[0]+1, se_pair[1])
            self.text = text[1:]
        else:
            self.se_pair = se_pair
            self.text = text
        """
        self.se_pair = se_pair
        self.text = text
        self.as_mapper = as_mapper

    def __str__(self) -> str:
        return 'UmStrWithY(y={}, se={}, text=[{}])'.format(self.y_val,
                                                           self.se_pair,
                                                           self.text)

    def to_tuple(self) -> Tuple[int,
                                Tuple[int, int],
                                str,
                                AlignedStrMapper]:
        return self.y_val, self.se_pair, self.text, self.as_mapper


class AbbyyPage:

    def __init__(self,
                 ab_blocks: List[Union[AbbyyTextBlock, AbbyyTableBlock]],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_blocks = ab_blocks
        self.ab_text_blocks = []  # type: List[AbbyyTextBlock]
        self.ab_table_blocks = []  # type: List[AbbyyTableBlock]

        # enable the ability to access prev and next ab_blocks
        for block_seq, ab_block in enumerate(ab_blocks):
            ab_block.page_block_seq = block_seq

        # Intentioanlly not setting this here.  Later component
        # might convert ab_text_blocks to ab_table_blocks.
        #
        # for ab_block in ab_blocks:
        #    if isinstance(ab_block, AbbyyTextBlock):
        #        self.ab_text_blocks.append(ab_block)
        #    elif isinstance(ab_block, AbbyyTableBlock):
        #        self.ab_table_blocks.append(ab_block)
        #    else:
        #        raise ValueError

        self.attr_dict = attr_dict
        self.infer_attr_dict = {}

        # for recording down unmatched info with pdfbox
        self.unsync_abbyy_lines = []  # type: List[AbbyyLine]
        self.unsync_abbyy_frags = [] # type: List[UnsyncedStrWithY]
        self.unsync_pbox_lines = []  # type: Tuple[UnsyncedPBoxLine]
        self.unsync_pbox_frags = []  # type: List[UnsyncedStrWithY]

    def has_unsynced_strs(self):
        return self.unsync_abbyy_lines or \
            self.unsync_abbyy_frags or \
            self.unsync_pbox_lines or \
            self.unsync_pbox_frags


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


def _pprint_table_attrs(attr_dict: Dict) -> str:
    st_list = []  # type: List[str]
    if attr_dict.get('type'):
        st_list.append('type={}'.format(attr_dict['type']))
    if attr_dict.get('@l'):
        st_list.append('l={}'.format(attr_dict['@l']))
    if attr_dict.get('@b'):
        st_list.append('b={}'.format(attr_dict['@b']))
    if attr_dict.get('@r'):
        st_list.append('r={}'.format(attr_dict['@r']))
    if attr_dict.get('@t'):
        st_list.append('t={}'.format(attr_dict['@t']))
    return ', '.join(st_list)


def _get_indent_level(attr_dict: Dict) -> int:
    for attr, unused_val in attr_dict.items():
        if attr.startswith('indent_1'):
            return 1
        elif attr.startswith('indent_2'):
            return 2
    return 0


def _print_left_right_panes(rt_line: str,
                            # pylint: disable=unused-argument
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


def _print_left_right_panes_with_sync(rt_line: str,
                                      line_attr_dict: Dict,
                                      to_se_list: List[Tuple[int, int]],
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

    tmp_x = line_attr_dict.get('x', -1)
    tmp_y = line_attr_dict.get('y', -1)
    left_line += ' xy=({}, {})'.format(tmp_x, tmp_y)

    if to_se_list:
        left_line += ' {}'.format(to_se_list)

    # 40 column for meta info
    print("  {:46}{}".format(left_line, tmp_rt_line), file=file)


def print_text_block_meta(ab_text_block: AbbyyTextBlock, file: TextIO = sys.stdout) -> None:
    print("\n  ----- block #{:3d} ----------".format(ab_text_block.num), file=file)

    is_header_footer = ab_text_block.infer_attr_dict.get('header', False) or \
                       ab_text_block.infer_attr_dict.get('footer', False)

    for unused_par_id, ab_par in enumerate(ab_text_block.ab_pars):
        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
        print(file=file)
        is_par_centered = _is_par_centered(ab_par.infer_attr_dict)
        indent_level = _get_indent_level(ab_par.infer_attr_dict)

        for unused_lid, ab_line in enumerate(ab_par.ab_lines):
            _print_left_right_panes(ab_line.text,
                                    ab_line.infer_attr_dict,
                                    par_id=ab_par.num,
                                    line_id=ab_line.num,
                                    is_header_footer=is_header_footer,
                                    indent_level=indent_level,
                                    is_par_centered=is_par_centered,
                                    file=file)


class AbbyyXmlDoc:

    def __init__(self,
                 file_name: str,
                 ab_pages: List[AbbyyPage]) -> None:
        self.file_id = file_name
        self.ab_pages = ab_pages

    def print_raw(self):
        for pnum, abbyy_page in enumerate(self.ab_pages):
            print("\n\npage #{} ========== {}".format(pnum, abbyy_page.attr_dict))
            for bid, ab_text_block in enumerate(abbyy_page.ab_text_blocks):
                print("\n    block #{} -------- {}".format(bid, ab_text_block.attr_dict))
                for par_id, ab_par in enumerate(ab_text_block.ab_pars):
                    print("        par #{} {}".format(par_id, ab_par.attr_dict))
                    for lid, ab_line in enumerate(ab_par.ab_lines):
                        print("            line #{} [{}] {}".format(lid,
                                                                    ab_line.text,
                                                                    ab_line.attr_dict))

    def print_raw_lines(self):
        for pnum, abbyy_page in enumerate(self.ab_pages):
            print("\n\npage #{} ========== {}".format(pnum, abbyy_page.attr_dict))
            for bid, ab_text_block in enumerate(abbyy_page.ab_text_blocks):
                print("\n    block #{} -------- {}".format(bid, ab_text_block.attr_dict))
                for par_id, ab_par in enumerate(ab_text_block.ab_pars):
                    print("\n        par #{} {}".format(par_id, ab_par.attr_dict))
                    for lid, ab_line in enumerate(ab_par.ab_lines):
                        print("            line #{} [{}]".format(lid,
                                                                 ab_line.text))


    def print_debug_text(self, file: TextIO = sys.stdout):
        for abbyy_page in self.ab_pages:
            if abbyy_page.num != 0:
                print('\n', file=file)
            print("page #{:2d} ============ {:95}{}    {}".format(abbyy_page.num,
                                                                  '',
                                                                  abbyy_page.infer_attr_dict,
                                                                  abbyy_page.attr_dict), file=file)
            for ab_text_block in abbyy_page.ab_text_blocks:
                print("\n  block #{:3d} -------- {:95}{}    {}".format(ab_text_block.num,
                                                                       '',
                                                                       # pylint: disable=line-too-long
                                                                       ab_text_block.infer_attr_dict,
                                                                       # pylint: disable=line-too-long
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
        for abbyy_page in self.ab_pages:
            if abbyy_page.num != 0:
                print('\n', file=file)
            print("page #{:2d} ============ {:95}{}".format(abbyy_page.num,
                                                            '',
                                                            abbyy_page.infer_attr_dict),
                  file=file)
            for ab_text_block in abbyy_page.ab_text_blocks:
                print("\n  block #{:3d} -------- {:95}{}".format(ab_text_block.num,
                                                                 '',
                                                                 ab_text_block.infer_attr_dict),
                      file=file)
                for ab_par in ab_text_block.ab_pars:
                    print("\n    par #{:3} {:104}{}".format(ab_par.num,
                                                            '',
                                                            ab_par.infer_attr_dict),
                          file=file)
                    for ab_line in ab_par.ab_lines:
                        print("      line #{} {:100} {}".format(ab_line.num,
                                                                '[' + ab_line.text + ']',
                                                                ab_line.infer_attr_dict),
                              file=file)


    # pylint: disable=too-many-nested-blocks, too-many-branches, too-many-locals
    def print_text_with_meta(self, file: TextIO = sys.stdout):
        for abbyy_page in self.ab_pages:
            if abbyy_page.num != 0:
                print('\n', file=file)
            print("========= page  #{:3d} ========".format(abbyy_page.num), file=file)

            # for ab_text_block in abbyy_page.ab_text_blocks:
            for ab_block in abbyy_page.ab_blocks:
                if isinstance(ab_block, AbbyyTextBlock):
                    ab_text_block = ab_block
                    print("\n  ----- block #{:3d} ----------".format(ab_text_block.num), file=file)

                    is_header_footer = ab_text_block.infer_attr_dict.get('header', False) or \
                                       ab_text_block.infer_attr_dict.get('footer', False)

                    for unused_par_id, ab_par in enumerate(ab_text_block.ab_pars):
                        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
                        print(file=file)
                        is_par_centered = _is_par_centered(ab_par.infer_attr_dict)
                        indent_level = _get_indent_level(ab_par.infer_attr_dict)

                        for unused_lid, ab_line in enumerate(ab_par.ab_lines):
                            _print_left_right_panes(ab_line.text,
                                                    ab_line.infer_attr_dict,
                                                    par_id=ab_par.num,
                                                    line_id=ab_line.num,
                                                    is_header_footer=is_header_footer,
                                                    indent_level=indent_level,
                                                    is_par_centered=is_par_centered,
                                                    file=file)
                elif isinstance(ab_block, AbbyyTableBlock):
                    ab_table_block = ab_block
                    # pylint: disable=line-too-long
                    print("\n  ----- block #{:3d}, page_num={} --Table--".format(ab_table_block.num,
                                                                                 ab_table_block.page_num), file=file)
                    print("  {}".format(_pprint_table_attrs(ab_table_block.attr_dict)),
                          file=file)

                    for row_id, ab_row in enumerate(ab_table_block.ab_rows):
                        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
                        print(file=file)
                        print("    {:26}--- row #{}".format('', row_id), file=file)
                        for cell_seq, ab_cell in enumerate(ab_row.ab_cells):
                            print("    {:26}  -- cell #{}:".format('', cell_seq), file=file)
                            for ab_par in ab_cell.ab_pars:
                                for unused_lid, ab_line in enumerate(ab_par.ab_lines):
                                    print("    {:26}        [{}]".format('',
                                                                         ab_line.text),
                                          file=file)
                else:
                    raise ValueError


        for abbyy_page in self.ab_pages:

            if abbyy_page.has_unsynced_strs():
                print("\n\n========= Unsynced strs in page {}========".format(abbyy_page.num),
                      file=file)
                print_abbyy_page_unsynced(abbyy_page, file=file)


    # pylint: disable=too-many-locals
    def print_text_with_meta_with_sync(self, file: TextIO = sys.stdout):
        for abbyy_page in self.ab_pages:
            if abbyy_page.num != 0:
                print('\n', file=file)
            print("========= page  #{:3d} ========".format(abbyy_page.num), file=file)

            # for ab_text_block in abbyy_page.ab_text_blocks:
            for ab_block in abbyy_page.ab_blocks:
                if isinstance(ab_block, AbbyyTextBlock):
                    ab_text_block = ab_block
                    print("\n  ----- block #{:3d} ----------".format(ab_text_block.num),
                          file=file)

                    is_header_footer = ab_text_block.infer_attr_dict.get('header', False) or \
                                       ab_text_block.infer_attr_dict.get('footer', False)

                    for unused_par_id, ab_par in enumerate(ab_text_block.ab_pars):
                        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
                        print(file=file)
                        is_par_centered = _is_par_centered(ab_par.infer_attr_dict)
                        indent_level = _get_indent_level(ab_par.infer_attr_dict)

                        for unused_lid, ab_line in enumerate(ab_par.ab_lines):
                            to_se_list = []
                            if ab_line.abbyy_pbox_offset_mapper and \
                               ab_line.abbyy_pbox_offset_mapper.to_se_list:
                                to_se_list = ab_line.abbyy_pbox_offset_mapper.to_se_list
                            _print_left_right_panes_with_sync(ab_line.text,
                                                              ab_line.infer_attr_dict,
                                                              to_se_list,
                                                              par_id=ab_par.num,
                                                              line_id=ab_line.num,
                                                              is_header_footer=is_header_footer,
                                                              indent_level=indent_level,
                                                              is_par_centered=is_par_centered,
                                                              file=file)
                elif isinstance(ab_block, AbbyyTableBlock):
                    ab_table_block = ab_block
                    # pylint: disable=line-too-long
                    print("\n  ----- block #{:3d}, page_num={} --Table--".format(ab_table_block.num,
                                                                                 ab_table_block.page_num), file=file)
                    print("  {}".format(_pprint_table_attrs(ab_table_block.attr_dict)),
                          file=file)

                    for row_id, ab_row in enumerate(ab_table_block.ab_rows):
                        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
                        print(file=file)
                        print("    {:26}--- row #{}".format('', row_id), file=file)
                        for cell_seq, ab_cell in enumerate(ab_row.ab_cells):
                            print("    {:26}  -- cell #{}:".format('', cell_seq), file=file)
                            for ab_par in ab_cell.ab_pars:
                                for unused_lid, ab_line in enumerate(ab_par.ab_lines):

                                    tmp_x = ab_line.infer_attr_dict.get('x', -1)
                                    tmp_y = ab_line.infer_attr_dict.get('y', -1)
                                    left_line = ' xy=({}, {})'.format(tmp_x, tmp_y)

                                    to_se_list = []
                                    if ab_line.abbyy_pbox_offset_mapper and \
                                       ab_line.abbyy_pbox_offset_mapper.to_se_list:
                                        to_se_list = ab_line.abbyy_pbox_offset_mapper.to_se_list
                                    left_line += ' {}'.format(str(to_se_list))
                                    print("    {:26}        [{}]".format(left_line, ab_line.text), file=file)
                else:
                    raise ValueError


def print_abbyy_page_unsynced(abbyy_page: AbbyyPage, file: TextIO = sys.stdout) -> int:
    return print_abbyy_page_unsynced_aux(abbyy_page.unsync_abbyy_lines,
                                         abbyy_page.unsync_abbyy_frags,
                                         abbyy_page.unsync_pbox_lines,
                                         abbyy_page.unsync_pbox_frags,
                                         file=file)


# pylint: disable=too-many-locals
def print_abbyy_page_unsynced_aux(unsync_abbyy_lines: List[AbbyyLine],
                                  unsync_abbyy_frags: List[UnsyncedStrWithY],
                                  unsync_pbox_lines: List[UnsyncedPBoxLine],
                                  unsync_pbox_frags: List[UnsyncedStrWithY],
                                  file: TextIO = sys.stdout) \
                                  -> int:
    count = 0
    for count_i, ua_line in enumerate(unsync_abbyy_lines):
        xval, yval = ua_line.infer_attr_dict['x'], ua_line.infer_attr_dict['y']
        print("  unsync abbyy_line #{}: xy={} [{}]".format(count_i, (xval, yval), ua_line.text),
              file=file)
        print(file=file)
        count += 1
    for count_i, ab_extra_se in enumerate(unsync_abbyy_frags):
        abbyy_y, ab_se, ab_text, unused_asm = ab_extra_se.to_tuple()
        print("  unsync_abbyy_frag #{}: y={} se={} [{}]".format(count_i, abbyy_y, ab_se, ab_text),
              file=file)
        print(file=file)
        count += 1
    for count_i, um_pbox_line in enumerate(unsync_pbox_lines):
        pbox_xypair, pbox_se, ptext = um_pbox_line.to_tuple()
        print("  unsync pbox_line #{}: xy={} se={} [{}]".format(count_i,
                                                                pbox_xypair,
                                                                pbox_se,
                                                                ptext),
              file=file)
        print(file=file)
        count += 1
    for count_i, pb_extra_se in enumerate(unsync_pbox_frags):
        pbox_y, pbox_se, pb_text, unused_asm = pb_extra_se.to_tuple()
        print("  unsync pbox_frag #{}: y={} se={} [{}]".format(count_i, pbox_y, pbox_se, pb_text),
              file=file)
        print(file=file)
        count += 1
    return count
