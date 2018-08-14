from collections import defaultdict
import re
import sys

# pylint: disable=unused-import
from typing import Any, Dict, List, Tuple

from kirke.abbyyxml.pdfoffsets import AbbyyBlock, AbbyyTextBlock, AbbyyTableBlock, AbbyyXmlDoc
# pylint: disable=unused-import
from kirke.abbyyxml.pdfoffsets import AbbyyLine, AbbyyPar, AbbyyCell, AbbyyRow
from kirke.abbyyxml import abbyyutils, pdfoffsets
from kirke.utils import mathutils, engutils

IS_DEBUG_TABLE = True
IS_PRINT_HEADER_TABLE = False
# IS_PRINT_HEADER_TABLE = True

IS_PRESERVE_INVALID_TABLE_AS_TEXT = False

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


def table_block_to_html(ab_table_block: AbbyyTableBlock) -> str:
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

    is_abbyy_table = True

    if ab_table_block.attr_dict.get('type'):
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
    else:
        st_list.append('<table border="1" bgcolor="00ff99">')  # green

    for unused_row_id, ab_row in enumerate(ab_table_block.ab_rows):
        # print("\n    par #{} {}".format(par_id, ab_par.infer_attr_dict))
        st_list.append('  <tr>')
        for unused_cell_seq, ab_cell in enumerate(ab_row.ab_cells):
            st_list.append('    <td>')
            for ab_par in ab_cell.ab_pars:
                for unused_lid, ab_line in enumerate(ab_par.ab_lines):
                    if not is_abbyy_table:
                        # pylint: disable=line-too-long
                        # st_list.append('{}      {}<br/>'.format(table_attrs_to_html(ab_line.attr_dict),
                        #                                        ab_line.text))
                        st_list.append('      {}<br/>'.format(ab_line.text))
                    else:
                        st_list.append('      {}<br/>'.format(ab_line.text))
            st_list.append('    </td>')
        st_list.append('  </tr>')
    st_list.append('</table>')
    st_list.append('<br/>')

    return '\n'.join(st_list)


def is_header_footer_table(ab_table_block: AbbyyTableBlock) -> bool:
    infer_attr_dict = ab_table_block.infer_attr_dict
    return infer_attr_dict.get('header') or \
           infer_attr_dict.get('footer')


def filter_out_header_footer_tables(table_block_list: List[AbbyyTableBlock]) \
    -> List[AbbyyTableBlock]:
    return [atable for atable in table_block_list if not is_header_footer_table(atable)]


def to_html_tables(abbyy_doc: AbbyyXmlDoc) -> str:
    st_list = []  # type: List[str]

    st_list.append('<!doctype html>')
    st_list.append('<html lang=en>')
    st_list.append('<head>')
    st_list.append('<meta charset=utf-8>')
    st_list.append('<title>{}</title>'.format(abbyy_doc.file_id))
    st_list.append('</head>')
    st_list.append('<body>')
    for ab_page in abbyy_doc.ab_pages:

        if IS_PRINT_HEADER_TABLE:
            table_block_list = ab_page.ab_table_blocks
        else:
            table_block_list = filter_out_header_footer_tables(ab_page.ab_table_blocks)

        if table_block_list:

            st_list.append('<h2>Page {}</h2>'.format(ab_page.num))
            # for ab_text_block in abbyy_page.ab_text_blocks:
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

def get_abbyy_table_list(abbyy_doc: AbbyyXmlDoc,
                         is_include_header_footer: bool = False) \
    -> List[AbbyyTableBlock]:
    out_table_list = []  # type: List[AbbyyTableBlock]
    for ab_page in abbyy_doc.ab_pages:
        if is_include_header_footer:
            page_table_list = ab_page.ab_table_blocks
        else:
            page_table_list = filter_out_header_footer_tables(ab_page.ab_table_blocks)
        out_table_list.extend(page_table_list)
    return out_table_list


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

                if is_top_bot_match(top, bot, other_top, other_bot):
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

def percent_top_bot_match(top: int,
                          bot: int,
                          prev_block_top: int,
                          prev_block_bot: int) -> float:
    diff = min(bot, prev_block_bot) - max(top, prev_block_top)
    if prev_block_bot - prev_block_top > bot - top:
        return diff / (prev_block_bot - prev_block_top)
    return diff / (bot - top)


def is_top_bot_match(top: int,
                     bot: int,
                     prev_block_top: int,
                     prev_block_bot: int) -> bool:
    is_overlap = top < prev_block_bot and \
                 prev_block_top < bot
    if is_overlap:
        perc_overlap = percent_top_bot_match(top,
                                             bot,
                                             prev_block_top,
                                             prev_block_bot)
        if perc_overlap >= 0.4:
            return True
    return False



# there is probably a more concise way of expressing this in python, 5345
def block_get_attr_left(block: AbbyyBlock) -> int:
    return block.attr_dict.get('@l', -1)


def get_row_seq_by_top(row_top_list: List[float], row_top: float) -> int:
    for row_seq, head_row_top in enumerate(row_top_list):
        if row_top <= head_row_top + 10.0:
            return row_seq
    return len(row_top_list) - 1


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

    print("\nmerge_aligned_blocks()")
    for row_top in row_top_list:
        print("row_top = {}".format(row_top))

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
    attr_dict['@b'] = haligned_blocks[0].attr_dict['@b']
    attr_dict['@r'] = haligned_blocks[-1].attr_dict['@r']
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

    if IS_DEBUG_TABLE:
        abbyyutils.infer_ab_block_is_header_footer(table_block)

        html_table = table_block_to_html(table_block)
        if not (table_block.infer_attr_dict.get('header') or \
                table_block.infer_attr_dict.get('footer')):
            print('\nhtml_table:')
            print(html_table)

    return table_block


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


def is_all_pars_align_justified(ab_pars: List[AbbyyPar]) -> bool:
    if not ab_pars:
        return False

    for ab_par in ab_pars:
        if ab_par.attr_dict.get('@align', 'None') != 'Justified':
            return False
    return True


def collect_justified_lines_after(ab_blocks: List[AbbyyBlock],
                                  page_block_seq: int,
                                  first_block_left: int) \
                                  -> List[AbbyyTextBlock]:
    if page_block_seq >= len(ab_blocks):
        return []

    result = []  # type: List[AbbyyTextBlock]
    # cur_ab_block = ab_blocks[page_block_seq]
    for ab_block in ab_blocks[page_block_seq + 1:]:
        if not isinstance(ab_block, AbbyyTextBlock):
            return result

        ab_pars = ab_block.ab_pars
        next_left = ab_block.attr_dict['@l']
        if is_all_pars_align_justified(ab_pars):
            print("first_block_left = {}, next_left = {}".format(first_block_left, next_left))

        # please note that this next_left check is not really correct.
        # the order of blocks inside a table is already screwed up.
        # so there is no guarantee that what's expected to be the next
        # block visually is really the next block.  Cannot solve correctly.
        # In these cases, the collection will simply fail.  Since this is
        # related to tables, it is probably fine.
        #
        # But for text blocks, this is probably still valid.
        if is_all_pars_align_justified(ab_pars) and \
           (first_block_left - 50 <= next_left and
            next_left <= first_block_left + 50):
            print("adding text block as justified:")
            pdfoffsets.print_text_block_meta(ab_block)
            result.append(ab_block)
        else:
            break

    return result


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def merge_haligned_block_as_table(ab_doc: AbbyyXmlDoc) -> None:

    # pylint: disable=too-many-nested-blocks
    for unused_pnum, abbyy_page in enumerate(ab_doc.ab_pages, 1):

        print("merge_haligned_blocks_as_table, page #{}".format(unused_pnum))
        print("        len(abbyy_page.ab_blocks = {}".format(len(abbyy_page.ab_blocks)))
        # find all the blocks with similar @b and @t
        ab_text_block_list = abbyyutils.get_only_text_blocks(abbyy_page.ab_blocks)

        # print("\nmerge_adj_aligned_block_as_table")
        if not ab_text_block_list:
            continue

        haligned_blocks_list = []  # type: List[List[AbbyyTextBlock]]
        # skip_blocks are the blocks that have already been found to be haligned
        skip_blocks = []  # type: List[AbbyyTextBlock]
        for i, ab_text_block in enumerate(ab_text_block_list):
            text_block_is_header_footer = ab_text_block.infer_attr_dict.get('footer', False) or \
                                          ab_text_block.infer_attr_dict.get('header', False)
            if ab_text_block in skip_blocks or text_block_is_header_footer:
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

        haligned_block_list_map = {}  # type: Dict[AbbyyTextBlock, List[AbbyyTextBlock]]
        justified_row_block_list_map = {}  # type: Dict[AbbyyTextBlock, List[AbbyyTextBlock]]
        for blocks in haligned_blocks_list:

            # found the column header, but no data
            # if len(blocks) <= 2:
            # jshaw, work
            # merge next text block if it is "justified"
            # most likely an minor column delimitation error from abbyy
            first_block = blocks[0]
            first_block_left = first_block.attr_dict['@l']
            last_block = blocks[-1]
            additional_table_row_blocks = \
                collect_justified_lines_after(abbyy_page.ab_blocks,
                                              last_block.page_block_seq,
                                              first_block_left)

            haligned_block_list_map[blocks[0]] = blocks
            justified_row_block_list_map[blocks[0]] = additional_table_row_blocks

        # now we have a list of haligned blocks
        # first try to form all the tables, with original blocks associated with table
        # kept.  Later, we might want to undo

        # maybe_block_origblocks_list has blocks + original_blocks
        maybe_block_origblocks_list = []  # type: List[Tuple[AbbyyBlock, List[AbbyyBlock]]]
        for ab_block in abbyy_page.ab_blocks:
            if ab_block in skip_blocks:
                continue

            if isinstance(ab_block, AbbyyTextBlock):
                haligned_blocks = haligned_block_list_map.get(ab_block, [])
                justified_row_blocks = justified_row_block_list_map.get(ab_block, [])

                if haligned_blocks:
                    table = merge_aligned_blocks(haligned_blocks, justified_row_blocks)
                    maybe_block_origblocks_list.append((table,
                                                        haligned_blocks +
                                                        justified_row_blocks))
                else:
                    # normal text blocks
                    # print("      block attr: {}".format(ab_block.attr_dict))
                    maybe_block_origblocks_list.append((ab_block, [ab_block]))
            else:
                # table blocks, from abby
                maybe_block_origblocks_list.append((ab_block, [ab_block]))

        """
        # this is for the missing rate table issue in GoldenWest.txt
        if unused_pnum == 106:
            for bbb_seq, (ab_block, origblocks) in enumerate(maybe_block_origblocks_list):
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
        out_block_origblocks_list = merge_adjacent_haligned_tables(maybe_block_origblocks_list)

        # now remove invalid tables, or put back invalid tables
        out_block_list = []  # type: List[AbbyyBlock]
        for ab_block, origblocks in out_block_origblocks_list:
            if isinstance(ab_block, AbbyyTableBlock):
                if IS_PRESERVE_INVALID_TABLE_AS_TEXT:
                    if is_invalid_table(ab_block):
                        if ab_block.is_abbyy_original:
                            # for now, we keep such tables
                            out_block_list.append(ab_block)
                        else:
                            # add all the original blocks back
                            out_block_list.extend(origblocks)
                    else:
                        out_block_list.append(ab_block)
                else:
                    # This is mainly for Table classification, without
                    # concerning for paragraph preservation
                    # simply ignore invalid tables
                    # don't bother keep them as text_blocks
                    if is_invalid_table(ab_block):
                        pass
                    else:
                        out_block_list.append(ab_block)
            else:
                # normal text block
                out_block_list.append(ab_block)

        abbyy_page.ab_blocks = out_block_list


def merge_adjacent_haligned_tables(ab_block_origblocks_list:
                                   List[Tuple[AbbyyBlock,
                                              List[AbbyyBlock]]]) \
                                   -> List[Tuple[AbbyyBlock,
                                                 List[AbbyyBlock]]]:
    out_block_origblocks_list = []  # type: List[Tuple[AbbyyBlock, List[AbbyyBlock]]]
    adjacent_table_list = []  # type: List[AbbyyTableBlock]
    adjacent_origblocks_list = []  # type: List[AbbyyBlock]
    for ab_block, origblocks in ab_block_origblocks_list:
        if isinstance(ab_block, AbbyyTableBlock) and \
           ab_block.is_abbyy_original:
            # abbyy's table is always trusted
            out_block_origblocks_list.append((ab_block, origblocks))
        elif isinstance(ab_block, AbbyyTableBlock):
            # is inferred, or haligned table
            adjacent_table_list.append(ab_block)
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
        if rattr < right_x:
            right_x = rattr
    attr_dict = {'@l': left_x,
                 '@r': right_x,
                 '@t': top_y,
                 '@b': bottom_y}
    table_block = AbbyyTableBlock(row_list, attr_dict, is_abbyy_original=False)
    table_block.page_num = ab_table_list[0].page_num
    return table_block


def is_invalid_table(ab_table: AbbyyTableBlock) -> bool:
    """A table is invalid for following reason:

       - has only 1 line, by looking at y's
       - has multiple '...', a toc
       - has multiple 'exhibit', etc, a toc
    """

    table_text = abbyyutils.table_block_to_text(ab_table)
    table_top_y, table_bottom_y = abbyyutils.table_block_to_y_top_bottom(ab_table)
    table_y_diff = table_bottom_y - table_top_y
    lc_table_text = table_text.lower()

    # this code is not triggered for true abbyytables?
    """
    if ab_table.is_abbyy_original and \
       ab_table.get_num_cols() == 1:
        print("--- is_invalid_table(), col == 1")
        print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True
    """

    if table_text.count('...') >= 2 or \
       lc_table_text.count('exhibit') >= 2 or \
       lc_table_text.count('appendix') >= 2 or \
       lc_table_text.count('article') >= 2:
        print("--- is_invalid_table(), dash, toc")
        print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True
    if table_y_diff <= 100:
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
        return True

    len_math_op_mat = len(math_op_mat_list)
    count_math_op = shorten_text.count(' x ') + \
                    shorten_text.count(' X ') + \
                    shorten_text.count(' + ') + \
                    shorten_text.count(' - ') + \
                    shorten_text.count(' * ') + \
                    shorten_text.count(' / ') + \
                    len_math_op_mat

    print("checking for is_invalid_table({})".format(table_text[:150]))
    print("count_paren= {}, count_math_op= {}".format(count_paren,
                                                      count_math_op))
    # this is a formula, not a table
    if count_paren >= 6 and count_math_op >= 2:
        print("--- is_invalid_table(), too mathy")
        print("  count_paren= {}, count_math_op= {}".format(count_paren,
                                                            count_math_op))
        print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True

    num_period_cap = engutils.num_letter_period_cap(table_text)
    if num_period_cap >= 3:
        print("--- is_invalid_table(), too sentency")
        print("num_period_cap: {}".format(num_period_cap))
        print("  table_text: [{}]".format(table_text.replace('\n', r'|')))
        return True


    return False
