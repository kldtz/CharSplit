from collections import namedtuple, Counter, defaultdict
from functools import total_ordering
import os
import sys
from typing import Any, DefaultDict, Dict, List, Optional, TextIO, Tuple, Union

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
        self.abby_pbox_offset_mapper = None  # type: Optional[AlignedStrMapper]

    def __str__(self):
        return '{} [{}]'.format(self.infer_attr_dict, self.text)

    def to_debug_str(self):
        return '{:80}{}    {}'.format('[' + self.text + ']',
                                      self.infer_attr_dict,
                                      self.attr_dict)


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


class AbbyCell:

    def __init__(self,
                 ab_pars: List[AbbyPar],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_pars = ab_pars
        self.attr_dict = attr_dict
        self.infer_attr_dict = {}


class AbbyRow:

    def __init__(self,
                 ab_cells: List[AbbyCell],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_cells = ab_cells
        self.attr_dict = attr_dict
        self.infer_attr_dict = {}


class AbbyTableBlock:

    def __init__(self,
                 ab_rows: List[AbbyRow],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_rows = ab_rows
        self.attr_dict = attr_dict
        self.infer_attr_dict = {}
        self.table_id = -1
        self.page_num = -1


class AbbyPage:

    def __init__(self,
                 ab_blocks: List[Union[AbbyTextBlock, AbbyTableBlock]],
                 attr_dict: Dict) -> None:
        self.num = -1
        self.ab_blocks = ab_blocks
        self.ab_text_blocks = []  # type: List[AbbyTextBlock]
        self.ab_table_blocks = []  # type: List[AbbyTableBlock]

        # Intentioanlly not setting this here.  Later component
        # might convert ab_text_blocks to ab_table_blocks.
        #
        # for ab_block in ab_blocks:
        #    if isinstance(ab_block, AbbyTextBlock):
        #        self.ab_text_blocks.append(ab_block)
        #    elif isinstance(ab_block, AbbyTableBlock):
        #        self.ab_table_blocks.append(ab_block)
        #    else:
        #        raise ValueError

        self.attr_dict = attr_dict
        self.infer_attr_dict = {}


class UnmatchedAbbyLine:

    def __init__(self,
                 ab_line: AbbyLine,
                 ab_page: AbbyPage) -> None:
        self.ab_line = ab_line
        self.ab_page = ab_page

    def __str__(self):
        return str(self.ab_line)


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


def _print_left_right_panes_with_sync(rt_line: str,
                                      line_attr_dict: Dict,
                                      to_se_list : List[Tuple[int, int]],
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

    if to_se_list:
        left_line += ' {}'.format(to_se_list)

    # 40 column for meta info
    print("    {:36}{}".format(left_line, tmp_rt_line), file=file)


def print_text_block_meta(ab_text_block: AbbyTextBlock, file: TextIO = sys.stdout) -> None:
    print("\n  ----- block #{:3d} ----------".format(ab_text_block.num), file=file)

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



class AbbyXmlDoc:

    def __init__(self,
                 file_name: str,
                 ab_pages: List[AbbyPage]) -> None:
        self.file_id = file_name
        self.ab_pages = ab_pages
        # to store ab_lines that are not found in pdfbox
        self.unmatched_ab_lines = []  # type: List[UnmatchedAbbyLine]

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

            # for ab_text_block in abby_page.ab_text_blocks:
            for ab_block in abby_page.ab_blocks:
                if isinstance(ab_block, AbbyTextBlock):
                    ab_text_block = ab_block
                    print("\n  ----- block #{:3d} ----------".format(ab_text_block.num), file=file)

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
                elif isinstance(ab_block, AbbyTableBlock):
                    ab_table_block = ab_block
                    print("\n  ----- block #{:3d}, page_num={} --Table--".format(ab_table_block.num,
                                                                                 ab_table_block.page_num), file=file)
                    print("  {}".format(_pprint_table_attrs(ab_table_block.attr_dict)), file=file)

                    for row_id, ab_row in enumerate(ab_table_block.ab_rows):
                        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
                        print(file=file)
                        print("    {:26}--- row #{}".format('', row_id), file=file)
                        for cell_seq, ab_cell in enumerate(ab_row.ab_cells):
                            print("    {:26}  -- cell #{}:".format('', cell_seq), file=file)
                            for ab_par in ab_cell.ab_pars:
                                for lid, ab_line in enumerate(ab_par.ab_lines):
                                    print("    {:26}        [{}]".format('', ab_line.text), file=file)
                else:
                    raise ValueError


        if self.unmatched_ab_lines:
            print("\n\n========= Unmatched Abby Lines ========", file=file)

            for unmatched_ab_line in self.unmatched_ab_lines:
                print("  page #{:2}, {}".format(unmatched_ab_line.ab_page.num,
                                                unmatched_ab_line.ab_line.to_debug_str()), file=file)


    def print_text_with_meta_with_sync(self, file: TextIO = sys.stdout):
        for abby_page in self.ab_pages:
            if abby_page.num != 0:
                print('\n', file=file)
            print("========= page  #{:3d} ========".format(abby_page.num), file=file)

            # for ab_text_block in abby_page.ab_text_blocks:
            for ab_block in abby_page.ab_blocks:
                if isinstance(ab_block, AbbyTextBlock):
                    ab_text_block = ab_block
                    print("\n  ----- block #{:3d} ----------".format(ab_text_block.num), file=file)

                    is_header_footer = ab_text_block.infer_attr_dict.get('header', False) or \
                                       ab_text_block.infer_attr_dict.get('footer', False)

                    for par_id, ab_par in enumerate(ab_text_block.ab_pars):
                        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
                        print(file=file)
                        is_par_centered = _is_par_centered(ab_par.infer_attr_dict)
                        indent_level = _get_indent_level(ab_par.infer_attr_dict)

                        for lid, ab_line in enumerate(ab_par.ab_lines):
                            to_se_list = []
                            if ab_line.abby_pbox_offset_mapper and \
                               ab_line.abby_pbox_offset_mapper.to_se_list:
                                to_se_list = ab_line.abby_pbox_offset_mapper.to_se_list
                            _print_left_right_panes_with_sync(ab_line.text,
                                                              ab_line.infer_attr_dict,
                                                              to_se_list,
                                                              par_id=ab_par.num,
                                                              line_id=ab_line.num,
                                                              is_header_footer=is_header_footer,
                                                              indent_level=indent_level,
                                                              is_par_centered=is_par_centered,
                                                              file=file)
                elif isinstance(ab_block, AbbyTableBlock):
                    ab_table_block = ab_block
                    print("\n  ----- block #{:3d}, page_num={} --Table--".format(ab_table_block.num,
                                                                                 ab_table_block.page_num), file=file)
                    print("  {}".format(_pprint_table_attrs(ab_table_block.attr_dict)), file=file)

                    for row_id, ab_row in enumerate(ab_table_block.ab_rows):
                        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
                        print(file=file)
                        print("    {:26}--- row #{}".format('', row_id), file=file)
                        for cell_seq, ab_cell in enumerate(ab_row.ab_cells):
                            print("    {:26}  -- cell #{}:".format('', cell_seq), file=file)
                            for ab_par in ab_cell.ab_pars:
                                for lid, ab_line in enumerate(ab_par.ab_lines):
                                    to_se_list = []
                                    if ab_line.abby_pbox_offset_mapper and \
                                       ab_line.abby_pbox_offset_mapper.to_se_list:
                                        to_se_list = ab_line.abby_pbox_offset_mapper.to_se_list
                                    print("    {:26}        [{}]".format(str(to_se_list), ab_line.text), file=file)
                else:
                    raise ValueError


        if self.unmatched_ab_lines:
            print("\n\n========= Unmatched Abby Lines ========", file=file)

            for unmatched_ab_line in self.unmatched_ab_lines:
                print("  page #{:2}, {}".format(unmatched_ab_line.ab_page.num,
                                                unmatched_ab_line.ab_line.to_debug_str()), file=file)



def table_attrs_to_html(attr_dict: Dict) -> str:
    st_list = []  # type: List[str]
    if attr_dict.get('@l'):
        st_list.append('l={}'.format(attr_dict['@l']))
    if attr_dict.get('@b'):
        st_list.append('b={}'.format(attr_dict['@b']))
    if attr_dict.get('@r'):
        st_list.append('r={}'.format(attr_dict['@r']))
    if attr_dict.get('@t'):
        st_list.append('t={}'.format(attr_dict['@t']))
    return ', '.join(st_list)


def table_block_to_html(ab_table_block: AbbyTableBlock) -> str:
    st_list = []  # type: List[str]

    st_list.append('<h3>Table {} on Page {}</h3>'.format(ab_table_block.table_id + 1,
                                                         ab_table_block.page_num))

    st_list.append('<table>')
    st_list.append('<tr>')
    st_list.append('<td width="60%">')
    st_list.append('</td>')
    st_list.append('<td>')
    st_list.append('<i>Attributes</i>: {}'.format(table_attrs_to_html(ab_table_block.attr_dict)))
    st_list.append('</td>')
    st_list.append('</tr>')
    st_list.append('</table>')

    if ab_table_block.attr_dict.get('type'):
        # a haligned table
        infer_attr_dict = ab_table_block.infer_attr_dict
        if infer_attr_dict.get('header') or \
           infer_attr_dict.get('footer'):
            st_list.append('<table border="1" bgcolor="DAA520">')  # brown
            # for now, no header for footer table
            return ''
        else:
            st_list.append('<table border="1" bgcolor="ffff66">')  # yellow
    else:
        st_list.append('<table border="1" bgcolor="00ff99">')  # green

    for row_id, ab_row in enumerate(ab_table_block.ab_rows):
        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
        st_list.append('  <tr>')
        for cell_seq, ab_cell in enumerate(ab_row.ab_cells):
            st_list.append('    <td>')
            for ab_par in ab_cell.ab_pars:
                for lid, ab_line in enumerate(ab_par.ab_lines):
                    st_list.append('      {}<br/>'.format(ab_line.text))
            st_list.append('    </td>')
        st_list.append('  </tr>')
    st_list.append('</table>')
    st_list.append('<br/>')

    return '\n'.join(st_list)

def is_header_footer_table(ab_table_block: AbbyTableBlock) -> bool:
    infer_attr_dict = ab_table_block.infer_attr_dict
    return infer_attr_dict.get('header') or \
           infer_attr_dict.get('footer')


def filter_out_header_footer_tables(table_block_list: List[AbbyTableBlock]) -> List[AbbyTableBlock]:
    return [atable for atable in table_block_list if not is_header_footer_table(atable) ]

def to_html_tables(abby_doc: AbbyXmlDoc) -> str:
    st_list = []  # type: List[str]

    st_list.append('<!doctype html>')
    st_list.append('<html lang=en>')
    st_list.append('<head>')
    st_list.append('<meta charset=utf-8>')
    st_list.append('<title>{}</title>'.format(abby_doc.file_id))
    st_list.append('</head>')
    st_list.append('<body>')
    for ab_page in abby_doc.ab_pages:

        table_block_list = filter_out_header_footer_tables(ab_page.ab_table_blocks)

        if table_block_list:

            st_list.append('<h2>Page {}</h2>'.format(ab_page.num))
            # for ab_text_block in abby_page.ab_text_blocks:
            for ab_table_block in table_block_list:

                html_table_st = table_block_to_html(ab_table_block)
                st_list.append(html_table_st)
                st_list.append('')
                st_list.append('')

            st_list.append('<br/>')
            st_list.append('<hr/>')
            st_list.append('<br/>')


    st_list.append('</body>')
    st_list.append('</html>')

    return '\n'.join(st_list)


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
