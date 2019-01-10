# pylint: disable=too-many-lines
from collections import defaultdict
import configparser
import os
import re
import sys

# pylint: disable=unused-import
from typing import Any, Dict, List, Optional, Tuple

from kirke.abbyyxml.pdfoffsets import AbbyyBlock, AbbyyTextBlock, AbbyyTableBlock, AbbyyXmlDoc
# pylint: disable=unused-import
from kirke.abbyyxml.pdfoffsets import AbbyyLine, AbbyyPar, AbbyyCell, AbbyyRow, DetectSource
from kirke.abbyyxml import abbyyutils, pdfoffsets
from kirke.utils import engutils, mathutils, strutils
from kirke.docstruct import secheadutils


# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')
EB_FILES = os.environ['EB_FILES']
KIRKE_TMP_DIR = EB_FILES + config['ebrevia.com']['KIRKE_TMP']
WORK_DIR = KIRKE_TMP_DIR + '/dir-work'


IS_DEBUG_TABLE = False
IS_PRINT_HEADER_TABLE = False

IS_DEBUG_INVALID_TABLE = False

# in general, we want to preserve all
# text in order to synch with pdfbox.
IS_PRESERVE_INVALID_TABLE_AS_TEXT = True

# is is specific for KPMG's K1 form
IS_K1_FORM = True
IS_OTHER_FORM = True

IS_ENABLE_FIELD_VALUE_TABLE = True


def block_attrs_to_html(attr_dict: Dict) -> str:
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


def table_rect_to_html(ajson: Dict) -> str:
    st_list = []  # type: List[str]
    if ajson.get('start'):
        st_list.append('start={}'.format(ajson['start']))
    if ajson.get('end'):
        st_list.append('end={}'.format(ajson['end']))
    if ajson.get('y_top'):
        st_list.append('y_top={}'.format(ajson['y_top']))
    if ajson.get('y_bottom'):
        st_list.append('y_bottom={}'.format(ajson['y_bottom']))
    if ajson.get('x_left'):
        st_list.append('x_left={}'.format(ajson['x_left']))
    if ajson.get('x_right'):
        st_list.append('x_right={}'.format(ajson['x_right']))
    return ', '.join(st_list)


def table_candidate_to_html(table_cand: Dict, table_id: int) -> str:
    st_list = []  # type: List[str]

    ajson = table_cand['json']

    st_list.append('<h2>Table {} on Page {}</h2>'.format(table_id,
                                                         ajson['page']))

    st_list.append('<ul>')
    st_list.append("<li><b>Detect Source</b>: " + ajson['detect_source'] + '</li>')
    st_list.append("<li><b>Section</b>: ")
    st_list.append(ajson.get('section_head', 'None1111'))
    st_list.append('</li>')
    st_list.append("<li><b>Pre-Table Text</b>:<br/> ")
    if ajson.get('pre_table_text'):
        st_list.append(ajson['pre_table_text'])
    else:
        st_list.append('<i>None</i>')
    st_list.append('</li>')
    st_list.append("<li><b>Attributes</b>: ")
    st_list.append(table_rect_to_html(ajson))
    st_list.append('</li>\n</ul>')

    # this is mainly for printing attribute info, not a real table
    st_list.append('<table>')
    st_list.append('<tr>')
    st_list.append('<td width="60%">')
    st_list.append('</td>')
    st_list.append('<td>')
    st_list.append('</td>')
    st_list.append('</tr>')
    st_list.append('</table>')

    if ajson['is_abbyy_original']:
        st_list.append('<table border="1" bgcolor="00ff99">')  # green
    else:
        # a haligned table
        if ajson.get('is_header') or \
           ajson.get('is_footer'):
            st_list.append('<table border="1" bgcolor="DAA520">')  # brown
            if not IS_PRINT_HEADER_TABLE:
                # for now, no header for footer table
                return ''
        else:
            st_list.append('<table border="1" bgcolor="ffff66">')  # yellow


    for ab_row in ajson['row_list']:
        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
        st_list.append('  <tr>')
        for ab_cell in ab_row['cell_list']:
            cell_text = ab_cell['text']
            st_list.append('    <td>')
            st_list.append('      {}'.format(cell_text.replace('\n', '<br/>')))
            st_list.append('    </td>')
        st_list.append('  </tr>')
    st_list.append('</table>')
    st_list.append('<br/>')

    st_list.append('')
    st_list.append('')

    return '\n'.join(st_list)


def table_candidates_to_html(fname: str, table_cand_list: List[Dict]) -> str:
    st_list = []  # type: List[str]

    st_list.append('<!doctype html>')
    st_list.append('<html lang=en>')
    st_list.append('<head>')
    st_list.append('<meta charset=utf-8>')
    st_list.append('<title>{}</title>'.format(fname))
    st_list.append('</head>')
    st_list.append('<body>')

    st_list.append('')

    has_valid_table = False
    for i, table_candidate in enumerate(table_cand_list, 1):
        html_table_st = table_candidate_to_html(table_candidate, i)

        if html_table_st:
            st_list.append(html_table_st)

            st_list.append('')
            st_list.append('<br/>')
            st_list.append('<hr/>')
            st_list.append('<br/>')
            st_list.append('')
            has_valid_table = True

    if not has_valid_table:
        st_list.append('There is no tables in "{}".'.format(fname))
        st_list.append('')

    st_list.append('</body>')
    st_list.append('</html>')

    return '\n'.join(st_list)


def abbyy_table_to_html(ab_table_block: AbbyyTableBlock, table_id: int = -1) -> str:
    st_list = []  # type: List[str]

    if table_id != -1:
        st_list.append('<h2>Table {} on Page {}</h2>'.format(table_id,
                                                             ab_table_block.page_num))
    else:
        st_list.append('<h2>Table on Page {}</h2>'.format(ab_table_block.page_num))

    st_list.append('<ul>')
    st_list.append('<li><b>Detect Source</b>: ' + ab_table_block.detect_source.name + '</li>')
    st_list.append('<li><b>IsAbbyyOriginal</b>: ' + str(ab_table_block.is_abbyy_original) + '</li>')
    st_list.append('<li><b>IsInvalidKirkeTable</b>: ' +
                   str(ab_table_block.is_invalid_kirke_table) + '</li>')
    st_list.append("<li><b>Attributes</b>: ")
    st_list.append(block_attrs_to_html(ab_table_block.attr_dict))
    st_list.append('</li>\n</ul>')

    if ab_table_block.is_abbyy_original:
        st_list.append('<table border="1" bgcolor="00ff99">')  # green
        is_abbyy_table = True
    else:
        # a haligned table
        is_abbyy_table = False
        infer_attr_dict = ab_table_block.infer_attr_dict
        if infer_attr_dict.get('header') or \
           infer_attr_dict.get('footer'):
            st_list.append('<table border="1" bgcolor="DAA520">')  # brown
            if not IS_PRINT_HEADER_TABLE:
                # for now, no header for footer table
                return ''
        else:
            st_list.append('<table border="1" bgcolor="ffff66">')  # yellow


    for unused_row_id, ab_row in enumerate(ab_table_block.ab_rows):
        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
        st_list.append('  <tr>')
        for unused_cell_seq, ab_cell in enumerate(ab_row.ab_cells):
            st_list.append('    <td>')
            for ab_par in ab_cell.ab_pars:
                for unused_lid, ab_line in enumerate(ab_par.ab_lines):
                    if not is_abbyy_table:
                        # pylint: disable=line-too-long
                        # st_list.append('{}      {}<br/>'.format(block_attrs_to_html(ab_line.attr_dict),
                        #                                        ab_line.text))
                        st_list.append('      {}<br/>'.format(ab_line.text))
                    else:
                        st_list.append('      {}<br/>'.format(ab_line.text))
            st_list.append('    </td>')
        st_list.append('  </tr>')
    st_list.append('</table>')
    st_list.append('<br/>')

    return '\n'.join(st_list)


def abbyy_text_block_to_html(ab_text_block: AbbyyTextBlock,
                             block_id: int = -1) -> str:
    st_list = []  # type: List[str]

    if block_id != -1:
        st_list.append('<h2>Text Block {} on Page {}</h2>'.format(block_id,
                                                             ab_text_block.page_num))
    else:
        st_list.append('<h2>Text Block on Page {}</h2>'.format(ab_text_block.page_num))

    st_list.append('<ul>')
    st_list.append("<li><b>Attributes</b>: ")
    st_list.append(block_attrs_to_html(ab_text_block.attr_dict))
    st_list.append('</li>\n</ul>')

    st_list.append('<table border="1" bgcolor="ffccff">')  # pink

    # st_list.append(' <br/>')
    for ab_par in ab_text_block.ab_pars:
        st_list.append('  <tr>')
        st_list.append('  <td>')
        for unused_lid, ab_line in enumerate(ab_par.ab_lines):
            st_list.append('{}<br/>'.format(ab_line.text))
    st_list.append('  </td>')
    st_list.append('  </tr>')
    st_list.append('</table>')
    st_list.append('<br/>')

    return '\n'.join(st_list)


def abbyy_tables_to_html(fname: str,
                         ab_table_list: List[AbbyyTableBlock]) -> str:
    st_list = []  # type: List[str]

    st_list.append('<!doctype html>')
    st_list.append('<html lang=en>')
    st_list.append('<head>')
    st_list.append('<meta charset=utf-8>')
    st_list.append('<title>{}</title>'.format(fname))
    st_list.append('</head>')
    st_list.append('<body>')

    st_list.append('')

    has_valid_table = False
    for i, ab_table in enumerate(ab_table_list, 1):
        html_table_st = abbyy_table_to_html(ab_table, i)

        if html_table_st:
            st_list.append(html_table_st)

            st_list.append('')
            st_list.append('<br/>')
            st_list.append('<hr/>')
            st_list.append('<br/>')
            st_list.append('')
            has_valid_table = True

    if not has_valid_table:
        st_list.append('There is no tables in "{}".'.format(fname))
        st_list.append('')

    st_list.append('</body>')
    st_list.append('</html>')

    return '\n'.join(st_list)

def abbyy_blocks_to_html(fname: str,
                         ab_block_list: List[AbbyyBlock]) -> str:
    st_list = []  # type: List[str]

    st_list.append('<!doctype html>')
    st_list.append('<html lang=en>')
    st_list.append('<head>')
    st_list.append('<meta charset=utf-8>')
    st_list.append('<title>{}</title>'.format(fname))
    st_list.append('</head>')
    st_list.append('<body>')

    st_list.append('')

    has_valid_table = False
    for i, ab_block in enumerate(ab_block_list, 1):
        if isinstance(ab_block, AbbyyTableBlock):
            html_table_st = abbyy_table_to_html(ab_block, i)
        else:
            html_table_st = abbyy_text_block_to_html(ab_block, i)

        if html_table_st:
            st_list.append(html_table_st)

            st_list.append('')
            st_list.append('<br/>')
            st_list.append('<hr/>')
            st_list.append('<br/>')
            st_list.append('')
            has_valid_table = True

    if not has_valid_table:
        st_list.append('There is no tables in "{}".'.format(fname))
        st_list.append('')

    st_list.append('</body>')
    st_list.append('</html>')

    return '\n'.join(st_list)


def table_block_to_json(ab_table_block: AbbyyTableBlock) -> Dict:
    table_dict = {'type': 'table',
                  'page': ab_table_block.page_num}

    table_dict['is_abbyy_original'] = ab_table_block.is_abbyy_original
    if ab_table_block.infer_attr_dict.get('header'):
        table_dict['header'] = ab_table_block.infer_attr_dict.get('header')
    if ab_table_block.infer_attr_dict.get('footer'):
        table_dict['footer'] = ab_table_block.infer_attr_dict.get('footer')

    if hasattr(ab_table_block, 'label_row_index') and \
       ab_table_block.label_row_index != -1:
        table_dict['label_row_index'] = ab_table_block.label_row_index
    if hasattr(ab_table_block, 'label_column_index') and \
       ab_table_block.label_column_index != -1:
        table_dict['label_column_index'] = ab_table_block.label_column_index

    row_list = []  # type: List[Dict]
    for unused_row_id, ab_row in enumerate(ab_table_block.ab_rows):
        row_dict = {}  # type: Dict
        cell_list = []  # type: List[Dict]
        if ab_row.attr_dict.get('rowSpan'):
            row_dict['rowSpan'] = ab_row.attr_dict['rowSpan']
        for unused_cell_seq, ab_cell in enumerate(ab_row.ab_cells):
            cell_dict = {}  # type: Dict
            cell_st_list = []  # type: List[str]
            if ab_cell.attr_dict.get('colSpan'):
                cell_dict['colSpan'] = ab_cell.attr_dict['colSpan']
            for ab_par in ab_cell.ab_pars:
                for unused_lid, ab_line in enumerate(ab_par.ab_lines):
                    cell_st_list.append(ab_line.text)
            cell_dict['text'] = '\n'.join(cell_st_list)
            if ab_cell.is_label:
                cell_dict['is_label'] = True
            cell_list.append(cell_dict)
        row_dict['cell_list'] = cell_list
        row_list.append(row_dict)
    table_dict['row_list'] = row_list

    return table_dict


def text_block_to_html(ab_text_block: AbbyyTextBlock) -> str:
    st_list = []  # type: List[str]

    # st_list.append('<h3>Text on Page {}</h3>'.format(ab_text_block.page_num))
    # TODO: we don't have page_num info for TextBlock, maybe add
    st_list.append('<h3>Text</h3>')

    # this is mainly for printing attribute info, not a real table
    st_list.append('<table>')
    st_list.append('<tr>')
    st_list.append('<td width="60%">')
    st_list.append('</td>')
    st_list.append('<td>')
    st_list.append('<i>Attributes</i>: {}'.format(block_attrs_to_html(ab_text_block.attr_dict)))
    st_list.append('</td>')
    st_list.append('</tr>')
    st_list.append('</table>')

    # st_list.append(' <br/>')
    for ab_par in ab_text_block.ab_pars:
        for unused_lid, ab_line in enumerate(ab_par.ab_lines):
            st_list.append('{}<br/>'.format(ab_line.text))
    st_list.append('<p/>')

    return '\n'.join(st_list)


def is_header_footer_block(ab_block: AbbyyTableBlock) -> bool:
    infer_attr_dict = ab_block.infer_attr_dict
    return infer_attr_dict.get('header') or \
           infer_attr_dict.get('footer')


def filter_out_header_footer_blocks(block_list: List[AbbyyBlock]) \
    -> List[AbbyyBlock]:
    return [ablock for ablock in block_list if not is_header_footer_block(ablock)]


def save_cand_tables_to_html_file(fname: str,
                                  table_candidates: List[Dict],
                                  extension: str) -> None:
    base_fname = os.path.basename(fname)
    out_fname = '{}/{}'.format(WORK_DIR,
                               base_fname.replace('.txt', extension))
    with open(out_fname, 'wt') as fout:
        print(table_candidates_to_html(fname, table_candidates), file=fout)
    print('wrote "{}"'.format(out_fname))


def save_tables_to_html_file(fname: str,
                             abbyy_tables: List[AbbyyTableBlock],
                             extension: str,
                             work_dir: str = WORK_DIR) -> None:
    base_fname = os.path.basename(fname)
    out_fname = '{}/{}'.format(work_dir,
                               base_fname.replace('.txt', extension))
    with open(out_fname, 'wt') as fout:
        print(abbyy_tables_to_html(fname, abbyy_tables), file=fout)
        print('wrote "{}"'.format(out_fname))


def save_blocks_to_html_file(fname: str,
                             abbyy_blocks: List[AbbyyBlock],
                             extension: str,
                             work_dir: str = WORK_DIR) -> None:
    base_fname = os.path.basename(fname)
    out_fname = '{}/{}'.format(work_dir,
                               base_fname.replace('.txt', extension))
    with open(out_fname, 'wt') as fout:
        print(abbyy_blocks_to_html(fname, abbyy_blocks), file=fout)
        print('wrote "{}"'.format(out_fname))


def to_html_tables(fname: str,
                   abbyy_doc: AbbyyXmlDoc,
                   extension: str,
                   work_dir: str = WORK_DIR) -> None:

    table_list = []  # type: List[AbbyyTableBlock]

    for ab_page in abbyy_doc.ab_pages:

        if IS_PRINT_HEADER_TABLE:
            table_block_list = ab_page.ab_table_blocks
        else:
            table_block_list = filter_out_header_footer_blocks(ab_page.ab_table_blocks)

        valid_table_block_list = [table_block for table_block in table_block_list
                                  if not table_block.is_invalid_kirke_table]

        table_list.extend(valid_table_block_list)

    save_tables_to_html_file(fname,
                             table_list,
                             extension=extension,
                             work_dir=work_dir)


def get_abbyy_table_list(abbyy_doc: AbbyyXmlDoc,
                         is_include_header_footer: bool = False) \
                         -> List[AbbyyTableBlock]:
    out_table_list = []  # type: List[AbbyyTableBlock]
    for ab_page in abbyy_doc.ab_pages:
        if is_include_header_footer:
            page_table_list = ab_page.ab_table_blocks
        else:
            page_table_list = filter_out_header_footer_blocks(ab_page.ab_table_blocks)

        valid_table_block_list = [table_block for table_block in page_table_list
                                  if not table_block.is_invalid_kirke_table]

        out_table_list.extend(valid_table_block_list)
    return out_table_list


def get_abbyy_signature_list(abbyy_doc: AbbyyXmlDoc) \
                             -> List[AbbyyTableBlock]:
    out_block_list = []  # type: List[AbbyyBlock]
    for ab_page in abbyy_doc.ab_pages:
        page_signature_list = ab_page.ab_signature_blocks
        out_block_list.extend(page_signature_list)
    return out_block_list


def get_abbyy_address_list(abbyy_doc: AbbyyXmlDoc) \
    -> List[AbbyyTableBlock]:
    out_block_list = []  # type: List[AbbyyBlock]
    for ab_page in abbyy_doc.ab_pages:
        page_address_list = ab_page.ab_address_blocks
        out_block_list.extend(page_address_list)
    return out_block_list


def is_address_block(abbyy_block: AbbyyBlock) -> bool:
    block_text = abbyy_block.get_text()
    # lc_block_text = block_text.lower()
    count_name = len(re.findall(r'\b(name|contact):?', block_text, flags=re.I))
    count_phone = len(re.findall(r'\b(toll free|international|domestic|'
                                 r'phone|telephone|mobile|cell|cellular|'
                                 r'dial|cellular phone):?',
                                 block_text, flags=re.I))
    count_fax = len(re.findall(r'\bfax:?', block_text, flags=re.I))
    count_postal = len(re.findall(r'\b(postal|address|mail to):?', block_text, flags=re.I))
    count_web = len(re.findall(r'\b(web|email):?', block_text, flags=re.I))
    count_zip_code = len(re.findall(r'\bzip code', block_text, flags=re.I))
    words = block_text.split()
    num_words = len(words)

    num_address_prefix = count_name + count_phone + count_fax + count_postal + \
                         count_web + count_zip_code
    if num_address_prefix >= 3 and \
       num_words < 55:
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), address")
            print("    num_address_prefix: {}".format(num_address_prefix))
            print("    num_words: {}".format(num_words))
            print("   block_text: [{}]".format(block_text.replace('\n', r'|')))
        return True
    return False


def is_signature_block(abbyy_block: AbbyyBlock) -> bool:
    block_text = abbyy_block.get_text()
    # lc_block_text = block_text.lower()
    count_name = len(re.findall(r'\bname:?', block_text, flags=re.I))
    count_title = len(re.findall(r'\btitle:?', block_text, flags=re.I))
    count_date = len(re.findall(r'\bdate:?', block_text, flags=re.I))
    count_by = len(re.findall(r'\bb[yv]:?', block_text, flags=re.I))
    words = block_text.split()
    num_words = len(words)

    if 'DATE SIGNED' in block_text and \
       'Officer' in block_text:
        return True

    if re.search(r'title.*signer', block_text, re.I) and \
       re.search(r'name.*title ', block_text, re.I):
        return True

    num_signature_prefix = count_name + count_title + count_date + count_by
    if num_signature_prefix >= 3 and \
       num_words < 35:
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), signature")
            print("    num_signature_prefix: {}".format(num_signature_prefix))
            print("    num_words: {}".format(num_words))
            print("   block_text: [{}]".format(block_text.replace('\n', r'|')))
        return True

    if num_signature_prefix >= 1:
        num_alphanum_words = len(strutils.get_alphanum_words_gt_len1(block_text))
        num_non_alphanum_words = len(strutils.get_non_alphanum_words_gt_len1(block_text))
        x_word_total = num_alphanum_words + num_non_alphanum_words
        perc_non_alphanum_words = num_non_alphanum_words / x_word_total
        if IS_DEBUG_INVALID_TABLE:
            print("num_alphanum_words: {}".format(num_alphanum_words))
            print("num_non_alphanum_words: {}".format(num_non_alphanum_words))
            print("perc_non_alphanum_words: {}".format(perc_non_alphanum_words))

        # not valid, page 8 in 8241 doc
        """
        if x_word_total > 15 and \
           perc_non_alphanum_words > 0.33:
            if IS_DEBUG_INVALID_TABLE:
                print("--- is_invalid_table(), signature2, weird words")
                print("    num_signature_prefix: {}".format(num_signature_prefix))
                print("    num_words: {}".format(num_words))
                print("   block_text: [{}]".format(block_text.replace('\n', r'|')))
            return True
        """

    return False


def get_pbox_text_offset(ab_table: AbbyyTableBlock) \
    -> Tuple[int, int]:
    min_start = sys.maxsize
    max_end = -1
    # pylint: disable=too-many-nested-blocks
    for ab_row in ab_table.ab_rows:
        for ab_cell in ab_row.ab_cells:
            for ab_par in ab_cell.ab_pars:
                for ab_line in ab_par.ab_lines:
                    # do this only if abbyy_pbox_offset_mapper is defined
                    if ab_line.abbyy_pbox_offset_mapper:
                        to_se_list = ab_line.abbyy_pbox_offset_mapper.to_se_list
                        for start, end in to_se_list:
                            if start < min_start:
                                min_start = start
                            if end > max_end:
                                max_end = end
                    else:
                        print("get_pbox_text_offset(), abbline not found: {}".format(ab_line))

    return min_start, max_end



SYNC_SPECIAL_CHARS_PAT = re.compile(r'[ _\-\.\s]')

def merge_adjacent_spans(se_list: List[Tuple[int, int]],
                         text: str) \
                         -> List[Tuple[int, int]]:
    out_span_list = []  # type: List[Tuple[int, int]]
    se_list = sorted(se_list)
    cur_start, cur_end = se_list[0]
    for start, end in se_list[1:]:
        if start > cur_end:
            between_text = re.sub(SYNC_SPECIAL_CHARS_PAT, ' ', text[cur_end:start]).strip()
            if not between_text:
                cur_end = end
            else:
                out_span_list.append((cur_start, cur_end))
                cur_start, cur_end = start, end
        else:
            out_span_list.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    out_span_list.append((cur_start, cur_end))

    return out_span_list


IS_DEBUG_OUTPUT_SPANS = False

def get_pbox_text_span_list(ab_table: AbbyyTableBlock,
                            text: str) \
    -> List[Tuple[int, int]]:

    out_se_list = []  # type: List[Tuple[int, int]]
    # pylint: disable=too-many-nested-blocks
    for ab_row in ab_table.ab_rows:
        for ab_cell in ab_row.ab_cells:
            for ab_par in ab_cell.ab_pars:
                for ab_line in ab_par.ab_lines:
                    if ab_line.abbyy_pbox_offset_mapper:
                        to_se_list = ab_line.abbyy_pbox_offset_mapper.to_se_list
                        out_se_list.extend(to_se_list)
                    else:
                        print("get_pbox_text_span_list(), abbline not found: {}".format(ab_line))

    merged_out_se_list = merge_adjacent_spans(out_se_list, text)

    if IS_DEBUG_OUTPUT_SPANS:
        for str_i, (start, end) in enumerate(out_se_list):
            print("  str_i #{}: ({}, {}) [{}]".format(str_i,
                                                      start, end,
                                                      text[start:end]))

    return merged_out_se_list


# not used
def find_haligned_blocks(ab_doc: AbbyyXmlDoc) -> None:

    for pnum, abbyy_page in enumerate(ab_doc.ab_pages):

        # pylint: disable=line-too-long
        yminmax_block_list = []  # type: List[Tuple[int, int, int, AbbyyBlock]]
        yminmax_blockid_list = []  # type: List[Tuple[int, int, int]]
        for block_i, ab_block in enumerate(abbyy_page.ab_blocks):
            miny, maxy = abbyyutils.find_block_minmaxy(ab_block)
            yminmax_block_list.append((block_i, miny, maxy, ab_block))
            yminmax_blockid_list.append((block_i, miny, maxy))

        print("=== page # {}".format(pnum + 1))
        for yminmax_block in yminmax_block_list:
            block_i, ymin, ymax, tmp_block = yminmax_block
            if isinstance(tmp_block, AbbyyTextBlock):
                print("  text block #{}, ymin={}, ymax={}".format(block_i, ymin, ymax))
                abbyyutils.print_text_block_first_line(tmp_block)
            else:
                print("  tabel block #{}, ymin={}, ymax={}".format(block_i, ymin, ymax))
                abbyyutils.print_table_block_first_line(tmp_block)

        # now found all blocks that are haligned
        # for yminmax_blockid in yminmax_blockid_list:
        #    print("yminmax_blockid: {}".format(yminmax_blockid))
        print("yminmax_blockid_list: {}".format(yminmax_blockid_list))
        grouped_ids = mathutils.find_overlaps_in_id_se_list(yminmax_blockid_list)
        print("group_ids: {}".format(grouped_ids))


        """
        # find all the blocks with similar @b and @t
        # type: List[AbbyyTextBlock], List[AbbyyTableBlock]
        ab_text_block_list, ab_table_block_list = [], []
        for ab_block in abbyy_page.ab_blocks:
            if isinstance(ab_block, AbbyyTextBlock)]:
                ab_text_block_list.append(ab_block)
            else:
                ab_table_block_list.append(ab_block)



        haligned_blocks_list = []  # type: List[List[AbbyyTextBlock]]
        # skip_blocks are the blocks that have already been found to be haligned
        skip_blocks = []  # type: List[AbbyyTextBlock]
        for i, ab_text_block in enumerate(ab_text_block_list):
            if ab_text_block in skip_blocks:
                continue

            attr_dict = ab_text_block.attr_dict
            top = attr_dict['@t']
            bot = attr_dict['@b']
            cur_blocks = []  # type: List[AbbyyTextBlock]
            # could sort by @b attribute first, but that might change the order in ab_text_block
            for other_text_block in ab_text_block_list[i+1:]:
                other_attr_dict = other_text_block.attr_dict
                other_top = other_attr_dict['@t']
                other_bot = other_attr_dict['@b']

                if mathutils.is_interval_overlap(top, bot, other_top, other_bot,
                                                 threshold=0.4):
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

        out_block_list = []  # type: List[AbbyyBlock]
        haligned_block_list_map = {}  # type: Dict[AbbyyTextBlock, List[AbbyyTextBlock]]
        for blocks in haligned_blocks_list:
            haligned_block_list_map[blocks[0]] = blocks

        # now we have a list of haligned blocks
        for ab_block in abbyy_page.ab_blocks:
            if ab_block in skip_blocks:
                continue

            if isinstance(ab_block, AbbyyTextBlock):
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

        abbyy_page.ab_blocks = out_block_list
        """


# there is probably a more concise way of expressing this in python, 5345
def block_get_attr_left(block: AbbyyBlock) -> int:
    return block.attr_dict.get('@l', -1)


def get_row_seq_by_top(row_top_list: List[float], row_top: float) -> int:
    for row_seq, head_row_top in enumerate(row_top_list):
        if row_top <= head_row_top + 10.0:
            return row_seq
    return len(row_top_list) - 1


# pylint: disable=too-many-locals
def text_blocks_to_table_block(haligned_blocks: List[AbbyyTextBlock]) -> AbbyyTableBlock:
    """This is to convert VERY simple AbbyyTextBlock to AbbyyTableBlock.

       Mainly for blocks with a few lines.
    """

    # there is probably a more concise way of expressing this in python, 5345
    haligned_blocks = sorted(haligned_blocks, key=block_get_attr_left)

    row_top_list = []  #
    par_list = haligned_blocks[0].ab_pars
    for par in par_list:
        for ab_line in par.ab_lines:
            row_top_list.append(ab_line.attr_dict['@t'])

    # print("\ntext_blocks_to_table_block()")
    # for row_top in row_top_list:
    #     print("row_top = {}".format(row_top))
    tab_xy_cell = defaultdict(list)  # type: Dict[Tuple[int, int], List[AbbyyLine]]
    for col_seq, tblock in enumerate(haligned_blocks):
        for unused_par_id, ab_par in enumerate(tblock.ab_pars):
            for ab_line in ab_par.ab_lines:
                row_seq = get_row_seq_by_top(row_top_list, ab_line.attr_dict['@t'])

                tab_xy_cell[(row_seq, col_seq)].append(ab_line)

    row_list = []  # type; List[AbbyyRow]
    for row_seq in range(len(row_top_list)):
        cell_list = []  # type: List[AbbyyCell]
        for col_seq in range(len(haligned_blocks)):
            apar = AbbyyPar(tab_xy_cell.get((row_seq, col_seq), []),
                            {})
            cell_list.append(AbbyyCell([apar], {}))
        row_list.append(AbbyyRow(cell_list, {}))

    attr_dict = {}
    attr_dict['@l'] = haligned_blocks[0].attr_dict['@l']
    attr_dict['@t'] = haligned_blocks[0].attr_dict['@t']

    attr_dict['@b'] = haligned_blocks[0].attr_dict['@b']
    attr_dict['@r'] = haligned_blocks[0].attr_dict['@r']
    for tmp_haligned_block in haligned_blocks[1:]:
        if tmp_haligned_block.attr_dict['@b'] > attr_dict['@b']:
            attr_dict['@b'] = tmp_haligned_block.attr_dict['@b']
        if tmp_haligned_block.attr_dict['@r'] > attr_dict['@r']:
            attr_dict['@r'] = tmp_haligned_block.attr_dict['@r']
    attr_dict['type'] = 'table-haligned'
    # we can combine block1+2's attr_dict

    infer_attr_dict = {}  # type: Dict[str, Any]
    table_block = AbbyyTableBlock(row_list, attr_dict, is_abbyy_original=False)
    table_block.infer_attr_dict = infer_attr_dict
    table_block.detect_source = DetectSource.H_ALIGN

    return table_block


# pylint: disable=too-many-locals, too-many-branches
def merge_aligned_blocks(haligned_blocks: List[AbbyyTextBlock],
                         justified_row_blocks: List[AbbyyTextBlock]) \
                         -> AbbyyTableBlock:
    """Merge a list of haligned blocks.

    The left most column is used to do the row splitting.
    """
    # there is probably a more concise way of expressing this in python, 5345
    haligned_blocks = sorted(haligned_blocks, key=block_get_attr_left)

    row_top_list = []  #
    par_list = haligned_blocks[0].ab_pars
    for par in par_list:
        for ab_line in par.ab_lines:
            row_top_list.append(ab_line.attr_dict['@t'])
    # par is not reliable enough
    """
    if len(par_list) == 1:
        for ab_line in par_list[0].ab_lines:
            row_top_list.append(ab_line.attr_dict['@t'])
    elif len(par_list) == 2 and \
         len(par_list[0].ab_lines) == 1:
        # for the first line, probably header
        ab_line = par_list[0].ab_lines[0]
        row_top_list.append(ab_line.attr_dict['@t'])
        # for the 2nd
        for ab_line in par_list[1].ab_lines:
            row_top_list.append(ab_line.attr_dict['@t'])
    else:
        # par is not reliable enough
        # for par in par_list:
        #     row_top_list.append(par.ab_lines[0].attr_dict['@t'])

        for par in par_list:
            for ab_line in par.ab_lines:
                row_top_list.append(ab_line.attr_dict['@t'])
    """

    # print("\nmerge_aligned_blocks()")
    # for row_top in row_top_list:
    #     print("row_top = {}".format(row_top))
    tab_xy_cell = defaultdict(list)  # type: Dict[Tuple[int, int], List[AbbyyLine]]
    for col_seq, tblock in enumerate(haligned_blocks):
        # par is so unreliable
        """
        for unused_par_id, ab_par in enumerate(tblock.ab_pars):
           row_seq = get_row_seq_by_top(row_top_list, ab_par.ab_lines[0].attr_dict['@t'])

           tab_xy_cell[(row_seq, col_seq)].extend(ab_par.ab_lines)
        """
        for unused_par_id, ab_par in enumerate(tblock.ab_pars):
            for ab_line in ab_par.ab_lines:
                row_seq = get_row_seq_by_top(row_top_list, ab_line.attr_dict['@t'])

                tab_xy_cell[(row_seq, col_seq)].append(ab_line)

    row_list = []  # type; List[AbbyyRow]
    for row_seq in range(len(row_top_list)):
        cell_list = []  # type: List[AbbyyCell]
        for col_seq in range(len(haligned_blocks)):
            apar = AbbyyPar(tab_xy_cell.get((row_seq, col_seq), []),
                            {})
            cell_list.append(AbbyyCell([apar], {}))
        row_list.append(AbbyyRow(cell_list, {}))

    attr_dict = {}
    attr_dict['@l'] = haligned_blocks[0].attr_dict['@l']
    attr_dict['@t'] = haligned_blocks[0].attr_dict['@t']

    attr_dict['@b'] = attr_dict['@t']
    attr_dict['@r'] = attr_dict['@l']
    for tmp_haligned_block in haligned_blocks[1:]:
        if tmp_haligned_block.attr_dict['@b'] > attr_dict['@b']:
            attr_dict['@b'] = tmp_haligned_block.attr_dict['@b']
        if tmp_haligned_block.attr_dict['@r'] > attr_dict['@r']:
            attr_dict['@r'] = tmp_haligned_block.attr_dict['@r']

    # attr_dict['@b'] = haligned_blocks[0].attr_dict['@b']
    # attr_dict['@r'] = haligned_blocks[-1].attr_dict['@r']
    attr_dict['type'] = 'table-haligned'
    # we can combine block1+2's attr_dict

    infer_attr_dict = {}  # type: Dict[str, Any]
    # Add the justified rows, each block is a row
    for tblock in justified_row_blocks:
        line_list = []  # type: List[AbbyyLine]
        for unused_par_id, ab_par in enumerate(tblock.ab_pars):
            line_list.extend(ab_par.ab_lines)
        apar = AbbyyPar(line_list, {})
        cell_list = []
        cell_list.append(AbbyyCell([apar], {}))

        if tblock.infer_attr_dict.get('header'):
            infer_attr_dict['header'] = True
        if tblock.infer_attr_dict.get('footer'):
            infer_attr_dict['footer'] = True

        row_list.append(AbbyyRow(cell_list, {}))

    table_block = AbbyyTableBlock(row_list, attr_dict, is_abbyy_original=False)
    table_block.infer_attr_dict = infer_attr_dict
    table_block.detect_source = DetectSource.H_ALIGN

    if IS_DEBUG_TABLE:
        abbyyutils.infer_ab_block_is_header_footer(table_block)

        html_table = abbyy_table_to_html(table_block)
        if not (table_block.infer_attr_dict.get('header') or \
                table_block.infer_attr_dict.get('footer')):
            print('\nafter merge_aligned_blocks(), html_table:')
            print(html_table)

    return table_block


"""
def merge_aligned_blocks_old(haligned_blocks: List[AbbyyTextBlock]) \
                             -> AbbyyTableBlock:
    # there is probably a more concise way of expressing this in python, 5345
    haligned_blocks = sorted(haligned_blocks, key=block_get_attr_left)
    cell_list = []
    # is_header = False
    # is_footer = False
    infer_attr_dict = {}
    for tblock in haligned_blocks:
        line_list = []  # type: List[AbbyyLine]
        for unused_par_id, ab_par in enumerate(tblock.ab_pars):
            line_list.extend(ab_par.ab_lines)
        apar = AbbyyPar(line_list, {})
        cell_list.append(AbbyyCell([apar], {}))

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
    row1 = AbbyyRow(cell_list, {})

    table_block = AbbyyTableBlock([row1], attr_dict, is_abbyy_original=False)
    table_block.infer_attr_dict = infer_attr_dict

    return table_block
"""


def is_all_pars_align_justified(ab_pars: List[AbbyyPar]) -> bool:
    if not ab_pars:
        return False

    for ab_par in ab_pars:
        if ab_par.attr_dict.get('@align', 'None') != 'Justified':
            return False
    return True


def collect_justified_lines_after(ab_blocks: List[AbbyyBlock],
                                  page_block_seq: int,
                                  first_block_left: int,
                                  stop_block_seq: int) \
                                  -> List[AbbyyTextBlock]:
    """Collect blocks after the table that are 'justified' after the first block.
       'Justified' means ABBYY has decided those lines as indented specially.
       The intention is probably collect additional
       text block that should be a part of the table, but didn't due to large
       spaces between the rows,

    Input parameters:
        - ab_blocks: page blocks
        - page_block_seq: the seq of last block in the table
        - the left, @l, of first block
        - stop_block_seq: the start block seq of the next halighed_block.  Don't go
          to the next halighed_block for now, there might be multiple blocks that
          can be merged, but we only handle 1 block in the current logic.  So don't
          try.

    Output:
        List of text blocks that should be added to the current table
        because they are 'justified' and their left's are very similar.
        If they are not 'justified' and mergeable, then return empty list.
    """
    # if page_block_seq >= len(ab_blocks):
    #     return []
    if page_block_seq + 1 >= stop_block_seq:
        return []

    result = []  # type: List[AbbyyTextBlock]
    for ab_block in ab_blocks[page_block_seq + 1:stop_block_seq]:
        # ab_block is the next block after the current table block group
        if not isinstance(ab_block, AbbyyTextBlock):
            return result

        ab_pars = ab_block.ab_pars
        next_left = ab_block.attr_dict['@l']
        next_num_block_words = len(ab_block.get_text().split())
        if IS_DEBUG_TABLE:
            if is_all_pars_align_justified(ab_pars):
                print("first_block_left = %d, next_left = %d, num_words = %d" %
                      (first_block_left, next_left, next_num_block_words))

        # Please note that this next_left check is not really correct.
        # The order of blocks inside a table is already screwed up.
        # so there is no guarantee that what's expected to be the next
        # block visually is really the next block.  Cannot solve correctly.
        # In these cases, the collection will simply fail.  Since this is
        # related to tables, it is probably fine.
        #
        # For text blocks, this is probably still valid.
        if is_all_pars_align_justified(ab_pars) and \
           (first_block_left - 50 <= next_left and
            next_left <= first_block_left + 50) and \
            next_num_block_words < 30:
            if IS_DEBUG_TABLE:
                print("adding text block as justified:")
                pdfoffsets.print_text_block_meta(ab_block)
            result.append(ab_block)
        else:
            break

    return result


def merge_value_field_block(ab_text_block: AbbyyTextBlock) -> Optional[AbbyyTableBlock]:
    row_list = []  # type: List[AbbyyRow]
    par_list = ab_text_block.ab_pars
    num_dot_prefix = 0
    num_sechead_prefix = 0
    num_line = 0
    num_line_col_ge_2 = 0
    for unused_row_num, par in enumerate(par_list):
        for ab_line in par.ab_lines:
            cols = ab_line.text.split('    ')
            num_line += 1

            # num_words = len(ab_line.text.split())
            # if num_words >= 12:
            #     num_line_word_ge_12 += 1
            if len(cols) >= 2:
                num_line_col_ge_2 += 1

            # this is an itemize list, not a table
            if len(cols) == 2 and \
               (cols[0] == '•' or
                cols[0] == '-' or
                cols[0] == 'o'):
                num_dot_prefix += 1

            if len(cols) == 2 and \
               secheadutils.is_line_sechead_prefix_only(cols[0]):
                num_sechead_prefix += 1

            cell_list = []  # type: List[AbbyyCell]
            for col in cols:
                # reused the same ab_line.attr_dict for
                # all cells
                aline = AbbyyLine(col, ab_line.attr_dict)
                apar = AbbyyPar([aline], {})
                cell_list.append(AbbyyCell([apar], {}))
            row_list.append(AbbyyRow(cell_list, {}))

    # num_line cannot be 0
    num_sechead_prefix_perc = num_sechead_prefix / num_line
    multi_col_rate = num_line_col_ge_2 / num_line

    if multi_col_rate < 0.2:
        return None

    if num_line == 0 or \
       num_sechead_prefix_perc > 0.8:
        pass
    elif num_dot_prefix >= 2 or \
       num_sechead_prefix >= 2:
        return None

    attr_dict = ab_text_block.attr_dict
    attr_dict['type'] = 'table-field-valud'

    infer_attr_dict = {}  # type: Dict[str, Any]
    table_block = AbbyyTableBlock(row_list, attr_dict, is_abbyy_original=False)
    table_block.infer_attr_dict = infer_attr_dict
    table_block.detect_source = DetectSource.FIELD_VALUE

    if is_invalid_table(table_block):
        return None

    return table_block


def merge_field_value_as_table(ab_doc: AbbyyXmlDoc) -> None:

    if not IS_ENABLE_FIELD_VALUE_TABLE:
        return

    for pnum, abbyy_page in enumerate(ab_doc.ab_pages, 1):

        # if the page is a multi-column page, skip it
        if abbyy_page.is_multi_column:
            continue

        invalid_start_end_list = pdfoffsets.blocks_to_rect_list(abbyy_page.invalid_tables)
        # ab_text_block_list = abbyyutils.get_only_text_blocks(abbyy_page.ab_blocks)
        out_block_list = []  # type: List[AbbyyBlock]
        for ab_block in abbyy_page.ab_blocks:
            if isinstance(ab_block, AbbyyTextBlock):
                if mathutils.is_overlap_with_rect_list(pdfoffsets.block_to_rect(ab_block),
                                                       invalid_start_end_list):
                    # if overlap with invalid table, don't bother
                    # with field-value
                    out_block_list.append(ab_block)
                else:
                    block_text = abbyyutils.text_block_to_text(ab_block)

                    mat_list = list(re.finditer(r'\s{4}', block_text))
                    table = None
                    if len(mat_list) > 1:
                        table = merge_value_field_block(ab_block)
                    if table:
                        out_block_list.append(table)
                    else:
                        out_block_list.append(ab_block)
            elif isinstance(ab_block, AbbyyTableBlock):
                out_block_list.append(ab_block)

        abbyy_page.ab_blocks = out_block_list


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def merge_haligned_blocks_as_table(ab_doc: AbbyyXmlDoc) -> None:
    """Go through each page and creates haligned blocks"""

    # abbyyutils.print_doc(ab_doc)
    # pylint: disable=too-many-nested-blocks
    for pnum, abbyy_page in enumerate(ab_doc.ab_pages, 1):

        # if the page is a multi-column page, skip it
        if abbyy_page.is_multi_column:
            continue

        if IS_DEBUG_TABLE:
            print("merge_haligned_blocks_as_table, page #{}".format(abbyy_page.num))
            print("        len(abbyy_page.ab_blocks = %d" % len(abbyy_page.ab_blocks))

        # if pnum == 79:
        #     print('ehre253243234')

        # find all the blocks with similar @b and @t
        # and store them in haligned_blocks_list, only for this page
        ab_text_block_list = abbyyutils.get_only_text_blocks(abbyy_page.ab_blocks)
        # We cannot simply exit if ab_text_block_list is empty here
        # because we still check for 'invalid_table()'

        haligned_blocks_list = []  # type: List[List[AbbyyTextBlock]]
        # skip_blocks are the blocks that have already been found to be haligned
        skip_blocks = []  # type: List[AbbyyTextBlock]
        for i, ab_text_block in enumerate(ab_text_block_list):
            text_block_is_header_footer = ab_text_block.infer_attr_dict.get('footer', False) or \
                                          ab_text_block.infer_attr_dict.get('header', False)
            if ab_text_block in skip_blocks or \
               text_block_is_header_footer:
                continue

            attr_dict = ab_text_block.attr_dict
            top = attr_dict['@t']
            bot = attr_dict['@b']
            # blocks that are haligned with the current one, ab_text_block
            cur_blocks = []  # type: List[AbbyyTextBlock]
            # could sort by @b attribute first, but that might change the order in ab_text_block
            for other_text_block in ab_text_block_list[i+1:]:
                other_attr_dict = other_text_block.attr_dict
                other_top = other_attr_dict['@t']
                other_bot = other_attr_dict['@b']

                if mathutils.is_interval_overlap(top, bot, other_top, other_bot,
                                                 threshold=0.4):
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

        # We cannot simply exit if haligned_block_list is empty here
        # because we still check for 'invalid_table()'
        # Rest of the code still does the right things.

        # pylint: disable=line-too-long
        haligned_block_list_map = {}  # type: Dict[AbbyyTextBlock, Tuple[List[AbbyyTextBlock], List[AbbyyTextBlock]]]
        for hseq, blocks in enumerate(haligned_blocks_list):

            # found the column header, but no data
            # if len(blocks) <= 2:
            # merge next text block if it is "justified"
            # most likely an minor column delimitation error from abbyy
            first_block = blocks[0]
            first_block_left = first_block.attr_dict['@l']
            last_block = blocks[-1]

            stop_block_seq = len(abbyy_page.ab_blocks)
            if hseq + 1 < len(haligned_blocks_list):
                stop_block_seq = haligned_blocks_list[hseq+1][0].page_block_seq

            additional_table_row_blocks = \
                collect_justified_lines_after(abbyy_page.ab_blocks,
                                              last_block.page_block_seq,
                                              first_block_left,
                                              stop_block_seq)

            skip_blocks.extend(additional_table_row_blocks)
            haligned_block_list_map[blocks[0]] = (blocks, additional_table_row_blocks)

        # Now we have a list of haligned blocks.  First try to form all the tables, with
        # original blocks associated with table kept.  Later, we might want to undo.

        # maybe_block_origblocks_list has blocks + original_blocks
        page_block_origblocks_list = []  # type: List[Tuple[AbbyyBlock, List[AbbyyBlock]]]
        for ab_block in abbyy_page.ab_blocks:
            if ab_block in skip_blocks:
                continue

            if isinstance(ab_block, AbbyyTextBlock):
                hblocks_additional_blocks = haligned_block_list_map.get(ab_block)
                if hblocks_additional_blocks:
                    haligned_blocks, additional_row_blocks = hblocks_additional_blocks
                    table = merge_aligned_blocks(haligned_blocks, additional_row_blocks)
                    page_block_origblocks_list.append((table, haligned_blocks + additional_row_blocks))
                else:
                    # normal text blocks
                    page_block_origblocks_list.append((ab_block, [ab_block]))
            else:
                # table blocks, from abby
                page_block_origblocks_list.append((ab_block, [ab_block]))

        """
        # this is for the missing rate table issue in GoldenWest.txt
        if unused_pnum == 106:
            for bbb_seq, (ab_block, origblocks) in enumerate(page_block_origblocks_list):
                print('\n-0------ block {}: ----'.format(bbb_seq))
                if isinstance(ab_block, AbbyyTableBlock):
                    print("is_table")
                    print("[{}]".format(abbyyutils.table_block_to_text(ab_block)))
                else:
                    print("is_text")
                    print("[{}]".format(abbyyutils.text_block_to_text(ab_block)))
        """

        # The spaces between some rows might be too big so that
        # each row becomes a paragraph.
        # Now merge haligned-tables that are adjacent
        # out_block_origblocks_list = []  # type: List[Tuple[AbbyyBlock, List[AbbyyBlock]]]
        out_block_origblocks_list = merge_adjacent_haligned_tables(page_block_origblocks_list)

        # now remove invalid tables, or put back invalid tables
        out_block_list = []  # type: List[AbbyyBlock]
        invalid_tables = []  # type: List[AbbyyBlock]
        for ab_block, origblocks in out_block_origblocks_list:

            # for debug purpose
            # block_text = abbyyutils.block_to_text(ab_block)
            # print("block_text: [[{}]]".format(block_text.replace('\n', '|.')))

            if isinstance(ab_block, AbbyyTableBlock):
                ab_block.page_num = pnum

                # in general, we want to preserve all
                # text in order to synch with pdfbox.
                if IS_PRESERVE_INVALID_TABLE_AS_TEXT:
                    if is_invalid_table(ab_block):
                        if ab_block.is_abbyy_original:
                            # for now, we keep such tables
                            out_block_list.append(ab_block)
                            ab_block.is_invalid_kirke_table = True
                        else:
                            # add all the original blocks back
                            out_block_list.extend(origblocks)
                        invalid_tables.append(ab_block)
                    else:
                        out_block_list.append(ab_block)
                else:
                    # This is mainly for Table classification, without
                    # concerning for paragraph preservation
                    # simply ignore invalid tables
                    # don't bother keep them as text_blocks
                    if is_invalid_table(ab_block):
                        invalid_tables.append(ab_block)
                        ab_block.is_invalid_kirke_table = True
                        # pass
                    else:
                        out_block_list.append(ab_block)
            else:
                # normal text block
                out_block_list.append(ab_block)

        for tmp_table in invalid_tables:
            tmp_table.page_num = pnum
        abbyy_page.ab_blocks = out_block_list
        abbyy_page.invalid_tables = invalid_tables


# pylint: disable=too-many-return-statements
def is_a_mergeable_row(ab_text_block: AbbyyTextBlock, prev_attrs: Dict) -> bool:
    """A mergeable table row is basically has less than 5 words."""

    # print("is_a_mergeable_row:")
    prev_attr_b = prev_attrs['@b']
    cur_attr_t = ab_text_block.attr_dict['@t']

    # print("   prev_attrs: {}".format(prev_attrs))
    # print("        attrs: {}".format(ab_text_block.attr_dict))
    # print("         diff: {}".format(cur_attr_t - prev_attr_b))

    if cur_attr_t < prev_attr_b:
        return False

    # they must be reasoanble close to each other
    if cur_attr_t > prev_attr_b and \
       cur_attr_t - prev_attr_b > 50:
        return False

    # mergeable rows shouldn't be centered
    if ab_text_block.is_centered():
        return False

    if IS_DEBUG_TABLE:
        print("is_a_mergeable_row():")
        pdfoffsets.print_text_block_meta(ab_text_block)

    block_ab_pars = ab_text_block.ab_pars
    if len(block_ab_pars) >= 3:
        return False
    for unused_par_id, ab_par in enumerate(ab_text_block.ab_pars):
        ab_lines = ab_par.ab_lines
        if len(ab_lines) >= 3:
            return False
        total_words = 0
        for ab_line in ab_lines:
            words = ab_line.text.split()
            if len(words) > 5:
                return False
            total_words += len(words)
            if total_words > 5:
                return False
    return True


def merge_adjacent_haligned_tables(ab_block_origblocks_list: List[Tuple[AbbyyBlock,
                                                                        List[AbbyyBlock]]]) \
                                   -> List[Tuple[AbbyyBlock,
                                                 List[AbbyyBlock]]]:

    # pylint: disable=line-too-long
    out_block_origblocks_list = []  # type: List[Tuple[AbbyyBlock, List[AbbyyBlock]]]
    adjacent_table_list = []  # type: List[AbbyyTableBlock]
    adjacent_origblocks_list = []  # type: List[AbbyyBlock]
    prev_block_attrs = {}  # type: Dict
    for ab_block, origblocks in ab_block_origblocks_list:
        if isinstance(ab_block, AbbyyTableBlock) and \
           ab_block.is_abbyy_original:
            # abbyy's table is always added to the unfiltered page blocks
            out_block_origblocks_list.append((ab_block, origblocks))
        elif isinstance(ab_block, AbbyyTableBlock):
            # is inferred, or haligned table
            adjacent_table_list.append(ab_block)
            adjacent_origblocks_list.extend(origblocks)
        elif adjacent_table_list and \
             isinstance(ab_block, AbbyyTextBlock) and \
             is_a_mergeable_row(ab_block, prev_block_attrs):
            # first need to convert the text block to table
            atable_block = text_blocks_to_table_block([ab_block])
            if IS_DEBUG_TABLE:
                print("Adding a mergeable, adjacent short text block to a table:")
                print(atable_block.get_text())
            adjacent_table_list.append(atable_block)
            adjacent_origblocks_list.extend(origblocks)
        else:
            # is text block
            if adjacent_table_list:
                atable_block = merge_multiple_adjacent_tables(adjacent_table_list)
                out_block_origblocks_list.append((atable_block, adjacent_origblocks_list))
                adjacent_table_list = []
                adjacent_origblocks_list = []
            # text block, simply add them
            out_block_origblocks_list.append((ab_block, origblocks))
        # this is just get the last attr_dict from origblocks, not ab_block
        prev_block_attrs = origblocks[-1].attr_dict

    if adjacent_table_list:
        atable_block = merge_multiple_adjacent_tables(adjacent_table_list)
        out_block_origblocks_list.append((atable_block, adjacent_origblocks_list))
        adjacent_table_list = []
        adjacent_origblocks_list = []

    return out_block_origblocks_list


def merge_multiple_adjacent_tables(ab_table_list: List[AbbyyTableBlock]) -> AbbyyTableBlock:
    if len(ab_table_list) == 1:
        return ab_table_list[0]

    top_y = sys.maxsize
    bottom_y = 0
    left_x = sys.maxsize
    right_x = 0
    row_list = []  # type: List[AbbyyRow]
    for ab_table in ab_table_list:
        row_list.extend(ab_table.ab_rows)

        battr = ab_table.attr_dict['@b']
        tattr = ab_table.attr_dict['@t']
        lattr = ab_table.attr_dict['@l']
        rattr = ab_table.attr_dict['@r']
        if tattr < top_y:
            top_y = tattr
        if battr > bottom_y:
            bottom_y = battr
        if lattr < left_x:
            left_x = lattr
        if rattr > right_x:
            right_x = rattr
    attr_dict = {'@l': left_x,
                 '@r': right_x,
                 '@t': top_y,
                 '@b': bottom_y}
    table_block = AbbyyTableBlock(row_list, attr_dict, is_abbyy_original=False)
    table_block.page_num = ab_table_list[0].page_num
    table_block.detect_source = DetectSource.H_ALIGN
    return table_block

YES_NO_PAT = re.compile(r'\byes\b(.*)\bno\b', re.I)
NO_YES_PAT = re.compile(r'\bno\b(.*)\byes\b', re.I)
YES_YES_PAT = re.compile(r'\byes\b(.*)\byes\b', re.I)
NO_NO_PAT = re.compile(r'\bno\b(.*)\bno\b', re.I)

# pylint: disable=invalid-name
def count_number_yes_no_choices_orig(table_text: str) -> int:
    st_list = table_text.split('\n')
    num_yes_no = 0
    for line in st_list:
        mat = YES_NO_PAT.search(line)
        if mat and len(mat.group(1)) < 5:
            num_yes_no += 1
    return num_yes_no

def count_number_yes_no_choices(table_text: str) -> int:
    dist = 10
    num_yes_no = 0
    for mat in YES_NO_PAT.finditer(table_text):
        if mat and len(mat.group(1)) <= dist:
            num_yes_no += 1

    num_no_yes = 0
    for mat in NO_YES_PAT.finditer(table_text):
        if mat and len(mat.group(1)) <= dist:
            num_no_yes += 1

    num_yes_yes = 0
    for mat in YES_YES_PAT.finditer(table_text):
        if mat and len(mat.group(1)) <= dist:
            num_yes_yes += 1

    num_no_no = 0
    for mat in NO_NO_PAT.finditer(table_text):
        if mat and len(mat.group(1)) <= dist:
            num_no_no += 1
    return max(num_yes_no, num_no_yes, num_yes_yes, num_no_no)


# this is too aggressive, elimiated valid table
"""
YES_NO_START_PAT = re.compile(r'.{0,3}\s*\b(yes|no)\b', re.I)

def count_yes_no_startswith(table_text: str) -> int:
    st_list = table_text.split('\n')
    num_yes_no_starts = 0
    for line in st_list:
        if YES_NO_START_PAT.match(line):
            num_yes_no_starts += 1
    return num_yes_no_starts
"""


def is_invalid_table(ab_table: AbbyyTableBlock) -> bool:
    is_invalid = is_invalid_table_aux(ab_table)
    if IS_DEBUG_INVALID_TABLE:
        print("***** is_invalid_table = {}".format(is_invalid))

    if is_invalid:
        return is_invalid

    if IS_K1_FORM:
        is_invalid = is_invalid_k1_table(ab_table)
        if IS_DEBUG_INVALID_TABLE:
            print("***** is_invalid_k1_table = {}".format(is_invalid))

    if IS_OTHER_FORM:
        is_invalid = is_invalid_other_form_table(ab_table)
        if IS_DEBUG_INVALID_TABLE:
            print("***** is_invalid_other_form_table = {}".format(is_invalid))

    return is_invalid


def is_invalid_other_form_table(ab_table: AbbyyTableBlock) -> bool:
    table_text = abbyyutils.table_block_to_text(ab_table)
    num_lines = count_number_lines(ab_table)
    # words = table_text.split()
    if IS_DEBUG_INVALID_TABLE:
        print('\n***** is_invalid_other_form_table[[{}]]'.format(table_text.replace('\n', '|')))

    if 'SOLICITATION' in table_text and \
       'CONTRACT' in table_text and \
       'ORDER FOR COMMERCIAL ITEMS' in table_text:
        return True

    if '13a' in table_text and \
       '13b' in table_text:
        return True

    if '9A.' in table_text and \
       '9B.' in table_text:
        return True

    if '30a.' in table_text and \
       '31c.' in table_text:
        return True

    if '28.' in table_text and \
       '29.' in table_text:
        return True

    if '17a' in table_text and \
       ('17b' in table_text or
        '18a' in table_text):
        return True

    if 'See Schedule' in table_text and \
       'TOTAL AWARD AMOUNT ' in table_text:
        return True

    if 'Change Order' in table_text and \
       ('Dispatch via' in table_text or
        'Bill To' in table_text or
        'Contract ID' in table_text):
        return True

    if 'Attn:' in table_text and \
       'Bill To' in table_text:
        return True

    if 'ISSUED BY' in table_text and \
       'ADMINISTERED BY' in table_text:
        return True

    if 'Control Number:' in table_text and \
       'See Summary' in table_text:
        return True

    if 'OFFICIAL' in table_text and \
       'SIGNATURE' in table_text:
        return True

    if 'ADDRESS' in table_text and \
       'CONTRACTOR' in table_text:
        return True

    if 'DISCOUNT FOR PROMPT PAYMENT' in table_text and \
       'FOB ORIGIN' in table_text:
        return True

    if 'Destination' in table_text and \
       ('See Herein' in table_text or
        'Duty' in table_text):
        return True

    # this is only in header
    if 'Contract No.' in table_text and \
       'Client Ref. No.' in table_text and \
       num_lines <= 5:
        return True

    if 'CONTRACT' in table_text and \
       ('DATE' in table_text or
        'SEE SCHEDULE' in table_text):
        return True

    # signature footer on the form
    if re.search(r'authorized', table_text, re.I) and \
       'STANDARD FORM' in table_text:
        return True

    return False

def is_invalid_k1_table(ab_table: AbbyyTableBlock) -> bool:
    table_text = abbyyutils.table_block_to_text(ab_table)
    words = table_text.split()
    if IS_DEBUG_INVALID_TABLE:
        print('\n***** is_invalid_k1_table[[{}]]'.format(table_text.replace('\n', '|')))

    if 'See Statement' in table_text and \
       'Self-employment earnings' in table_text and \
       'Qualified nonrecourse financing' in table_text and \
       'share of liabilities' in table_text:
        return True

    if 'capital account analysis' in table_text and \
       'Beginning capital account' in table_text and \
       'Current year increase' in table_text and \
       'Ending capital account' in table_text:
        return True

    if 'Profit' in table_text and \
       'Loss' in table_text and \
       'Capital' in table_text and \
       len(words) < 20:
        return True

    if 'Other Form 1116' in table_text and \
       'Other portfolio income' in table_text and \
       'Cancellation of debt' in table_text and \
       'Involuntary conversions' in table_text:
        return True

    if 'Cash contributions' in table_text and \
       'Noncash contributions' in table_text and \
       'Investment interest expense' in table_text and \
       'Section 59(e)(2) expenditures' in table_text:
        return True

    if 'Section 453(l)(3) information' in table_text and \
       'Low-income housing credit' in table_text and \
       'Disabled access credit' in table_text and \
       'Recapture of section 179 deduction' in table_text:
        return True

    if 'Total foreign taxes paid' in table_text and \
       'Form 8873' in table_text and \
       'Form 1116, Part II' in table_text and \
       'Extraterritorial income exclusion' in table_text:
        return True

    if 'Post-1986 depreciation adjustment' in table_text and \
       'Oil, gas, & geothermal' in table_text and \
       'gross income' in table_text and \
       'deduction' in table_text and \
       'Form 6251' in table_text and \
       'Other AMT items' in table_text:
        return True

    if 'Form 4952, line 4a' in table_text and \
       'Form 4136' in table_text and \
       'Form 8611, line 8' in table_text and \
       'Form 8697' in table_text:
        return True

    if 'identifying number' in table_text and \
       '9a Net long-term capital gain' in table_text and \
       '9b Collectibles' in table_text:
        return True

    if 'General category' in table_text and \
       'Form 1116' in table_text and \
       'Mining exploration costs' in table_text and \
       'Sec. 1256 contracts & straddles' in table_text:
        return True

    if 'Schedule K-1' in table_text and \
       '6b' in table_text and \
       '9a' in table_text and \
       '9b' in table_text:
        return True

    if 'Schedule K-1' in table_text and \
       'Form 1065' in table_text and \
       'Department of the Treasury' in table_text:
        return True

    if 'Guaranteed payments' in table_text and \
       'Interest income' in table_text and \
       'Ordinary dividends' in table_text and \
       '6a' in table_text:
        return True

    if 'See Statement' in table_text and \
       'Information About the Partner' in table_text:
        return True

    if 'See Statement' in table_text and \
       ('Foreign partner' in table_text or
        'Domestic partner' in table_text):
        return True

    if 'See Statement' in table_text and \
       'Profit' in table_text and \
       'Nonrecourse' in table_text:
        return True

    if 'Current year increase' in table_text and \
       'Withdrawals & distributions' in table_text and \
       'Tax basis' in table_text:
        return True

    if 'Passive loss' in table_text and \
       'Passive income' in table_text and \
       'Nonpassive loss' in table_text:
        return True

    if 'back of form' in table_text and \
       'separate instructions' in table_text and \
       'See Statement' in table_text:
        return True

    if '8 Net short-term capital gain' in table_text or \
       '9a Net long-term capital gain' in table_text:
        return True

    return False


# pylint: disable=too-many-return-statements
def is_invalid_table_aux(ab_table: AbbyyTableBlock) -> bool:
    """A table is invalid for following reason:

       - has only 1 line, by looking at y's
       - has multiple '...', a toc
       - has multiple 'exhibit', etc, a toc
    """

    table_text = abbyyutils.table_block_to_text(ab_table)
    if IS_DEBUG_INVALID_TABLE:
        print('\n***** is_invalid_table[[{}]]'.format(table_text.replace('\n', '|')))

    table_top_y, table_bottom_y = abbyyutils.table_block_to_y_top_bottom(ab_table)
    table_y_diff = table_bottom_y - table_top_y
    lc_table_text = table_text.lower()
    words = table_text.split()

    # if len(words) <= 12:
    # some header might be inverted, so really few words
    if len(words) <= 8:
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), too few words, {}".format(len(words)))
            print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True

    if ab_table.is_header() or \
       ab_table.is_footer():
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), is_footer=%r, is_header=%r" %
                  (ab_table.is_header(), ab_table.is_footer()))
            print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True

    if count_number_yes_no_choices(table_text) >= 2:
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), number of yes_no >= 2")
            print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True

    # this is too aggressive, 8242, page 14
    """
    if count_yes_no_startswith(table_text) >= 2:
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), number of yes_no_startswith >= 2")
            print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True
    """

    toc_mat = re.search(r'\btable(.*)contents?\b', table_text, flags=re.I)
    pagenum_mat = re.search(r'\bPage(.*)N(o|um|umber)?\b', table_text, flags=re.I)
    if (toc_mat and len(toc_mat.group(1)) < 8) or \
       (pagenum_mat and len(pagenum_mat.group(1)) < 8):
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), table of content or page number phrase found")
            print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True

    # this code is not triggered for true abbyytables?
    """
    if ab_table.is_abbyy_original and \
       ab_table.get_num_cols() == 1:
        print("--- is_invalid_table(), col == 1")
        print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True
    """

    if table_text.count('...') >= 2 or \
       lc_table_text.count('exhibit') >= 3 or \
       lc_table_text.count('appendix') >= 3 or \
       lc_table_text.count('article') >= 3:
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), dash, toc")
            print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True

    # Saw a two-line table,table_y_diff is 85
    # but that table has no heading.  Keeping
    # a minimal table as 100.
    if table_y_diff <= 100:
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), too small y-diff")
            print("  table_y_top_bottom: {}, {}, diff={}".format(table_top_y,
                                                                 table_bottom_y,
                                                                 table_y_diff))
            print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True

    shorten_text = table_text[:150]
    count_paren = shorten_text.count('(') + \
                  shorten_text.count(')') + \
                  shorten_text.count('[') + \
                  shorten_text.count(']')
    math_op_mat_list = re.findall(r'\)\s*[xX\+\-\*\/]\s*\(', shorten_text)

    if re.search(r'\d+\s*[xX\+\-\*\/]\s*\(', shorten_text):
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), too mathy branch 1")
            # pylint: disable=line-too-long
            print("  count_paren= {}, len(math_op_mat_list)= {} ".format(count_paren,
                                                                         len(math_op_mat_list)))
            print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True

    len_math_op_mat = len(math_op_mat_list)
    count_math_op = shorten_text.count(' x ') + \
                    shorten_text.count(' X ') + \
                    shorten_text.count(' + ') + \
                    shorten_text.count(' - ') + \
                    shorten_text.count(' * ') + \
                    shorten_text.count(' / ') + \
                    len_math_op_mat

    # print("checking for is_invalid_table({})".format(table_text[:150]))
    # print("count_paren= {}, count_math_op= {}".format(count_paren,
    #                                                   count_math_op))

    # this is a formula, not a table
    if count_paren >= 6 and count_math_op >= 2:
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), too mathy branch2")
            print("  count_paren= {}, count_math_op= {}".format(count_paren,
                                                                count_math_op))
            print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True

    # failed on document 366655.pdf
    """
    num_period_cap = engutils.num_letter_period_cap(table_text)
    if num_period_cap >= 3:
        if IS_DEBUG_INVALID_TABLE:
            print("--- is_invalid_table(), too sentency")
            print("num_period_cap: {}".format(num_period_cap))
            print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True
    """

    return False



def print_page_tables(ab_doc: AbbyyXmlDoc, page_num: int) -> None:
    """Go page and print table blocks"""

    for pnum, abbyy_page in enumerate(ab_doc.ab_pages, 1):

        if pnum == page_num:

            for block_seq, ab_block in enumerate(abbyy_page.ab_blocks):
                if isinstance(ab_block, AbbyyTableBlock):
                    print("\n=====table #{} in page {}".format(block_seq, page_num))
                    html_table = abbyy_table_to_html(ab_block)
                    print(html_table)


            for block_seq, ab_block in enumerate(abbyy_page.ab_table_blocks):
                if isinstance(ab_block, AbbyyTableBlock):
                    print("\n=====table_block #{} in page {}".format(block_seq, page_num))
                    html_table = abbyy_table_to_html(ab_block)
                    print(html_table)

            for block_seq, ab_block in enumerate(abbyy_page.invalid_tables):
                if isinstance(ab_block, AbbyyTableBlock):
                    print("\n=====invalid table_block #{} in page {}".format(block_seq, page_num))
                    html_table = abbyy_table_to_html(ab_block)
                    print(html_table)


def count_number_lines(ab_table: AbbyyTableBlock) -> int:
    return len(ab_table.ab_rows)



