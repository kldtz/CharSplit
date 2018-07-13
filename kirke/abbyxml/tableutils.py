from collections import defaultdict

# pylint: disable=unused-import
from typing import Dict, List, Tuple

from kirke.abbyxml.pdfoffsets import AbbyBlock, AbbyTextBlock, AbbyTableBlock, AbbyXmlDoc
# pylint: disable=unused-import
from kirke.abbyxml.pdfoffsets import AbbyLine, AbbyPar, AbbyCell, AbbyRow
from kirke.abbyxml import abbyutils, pdfoffsets
from kirke.utils import mathutils

IS_DEBUG_TABLE = True
IS_PRINT_HEADER_TABLE = False
# IS_PRINT_HEADER_TABLE = True

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

    is_abby_table = True

    if ab_table_block.attr_dict.get('type'):
        # a haligned table
        is_abby_table = False
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
                    if not is_abby_table:
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


def is_header_footer_table(ab_table_block: AbbyTableBlock) -> bool:
    infer_attr_dict = ab_table_block.infer_attr_dict
    return infer_attr_dict.get('header') or \
           infer_attr_dict.get('footer')


def filter_out_header_footer_tables(table_block_list: List[AbbyTableBlock]) \
    -> List[AbbyTableBlock]:
    return [atable for atable in table_block_list if not is_header_footer_table(atable)]


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

        if IS_PRINT_HEADER_TABLE:
            table_block_list = ab_page.ab_table_blocks
        else:
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


# not used
def find_haligned_blocks(ab_doc: AbbyXmlDoc) -> None:

    for pnum, abby_page in enumerate(ab_doc.ab_pages):

        # pylint: disable=line-too-long
        yminmax_block_list = []  # type: List[Tuple[int, int, int, AbbyBlock]]
        yminmax_blockid_list = []  # type: List[Tuple[int, int, int]]
        for block_i, ab_block in enumerate(abby_page.ab_blocks):
            miny, maxy = abbyutils.find_block_minmaxy(ab_block)
            yminmax_block_list.append((block_i, miny, maxy, ab_block))
            yminmax_blockid_list.append((block_i, miny, maxy))

        print("=== page # {}".format(pnum + 1))
        for yminmax_block in yminmax_block_list:
            block_i, ymin, ymax, tmp_block = yminmax_block
            if isinstance(tmp_block, AbbyTextBlock):
                print("  text block #{}, ymin={}, ymax={}".format(block_i, ymin, ymax))
                abbyutils.print_text_block_first_line(tmp_block)
            else:
                print("  tabel block #{}, ymin={}, ymax={}".format(block_i, ymin, ymax))
                abbyutils.print_table_block_first_line(tmp_block)

        # now found all blocks that are haligned
        # for yminmax_blockid in yminmax_blockid_list:
        #    print("yminmax_blockid: {}".format(yminmax_blockid))
        print("yminmax_blockid_list: {}".format(yminmax_blockid_list))
        grouped_ids = mathutils.find_overlaps_in_id_se_list(yminmax_blockid_list)
        print("group_ids: {}".format(grouped_ids))


        """
        # find all the blocks with similar @b and @t
        # type: List[AbbyTextBlock], List[AbbyTableBlock]
        ab_text_block_list, ab_table_block_list = [], []
        for ab_block in abby_page.ab_blocks:
            if isinstance(ab_block, AbbyTextBlock)]:
                ab_text_block_list.append(ab_block)
            else:
                ab_table_block_list.append(ab_block)



        haligned_blocks_list = []  # type: List[List[AbbyTextBlock]]
        # skip_blocks are the blocks that have already been found to be haligned
        skip_blocks = []  # type: List[AbbyTextBlock]
        for i, ab_text_block in enumerate(ab_text_block_list):
            if ab_text_block in skip_blocks:
                continue

            attr_dict = ab_text_block.attr_dict
            top = attr_dict['@t']
            bot = attr_dict['@b']
            cur_blocks = []  # type: List[AbbyTextBlock]
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

        out_block_list = []  # type: List[AbbyBlock]
        haligned_block_list_map = {}  # type: Dict[AbbyTextBlock, List[AbbyTextBlock]]
        for blocks in haligned_blocks_list:
            haligned_block_list_map[blocks[0]] = blocks

        # now we have a list of haligned blocks
        for ab_block in abby_page.ab_blocks:
            if ab_block in skip_blocks:
                continue

            if isinstance(ab_block, AbbyTextBlock):
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

        abby_page.ab_blocks = out_block_list
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
def block_get_attr_left(block: AbbyBlock) -> int:
    return block.attr_dict.get('@l', -1)


def get_row_seq_by_top(row_top_list: List[float], row_top: float) -> int:
    for row_seq, head_row_top in enumerate(row_top_list):
        if row_top <= head_row_top + 10.0:
            return row_seq
    return len(row_top_list) - 1


# pylint: disable=too-many-locals
def merge_aligned_blocks(haligned_blocks: List[AbbyTextBlock],
                         justified_row_blocks: List[AbbyTextBlock]) \
                         -> AbbyTableBlock:
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

    tab_xy_cell = defaultdict(list)  # type: Dict[Tuple[int, int], List[AbbyLine]]
    for col_seq, tblock in enumerate(haligned_blocks):
        line_list = []  # type: List[AbbyLine]

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


    row_list = []  # type; List[AbbyRow]
    for row_seq in range(len(row_top_list)):
        cell_list = []  # type: List[AbbyCell]
        for col_seq in range(len(haligned_blocks)):

            line_list = tab_xy_cell.get((row_seq, col_seq), [])

            apar = AbbyPar(line_list, {})
            cell_list.append(AbbyCell([apar], {}))
        row_list.append(AbbyRow(cell_list, {}))

    attr_dict = {}
    attr_dict['@l'] = haligned_blocks[0].attr_dict['@l']
    attr_dict['@t'] = haligned_blocks[0].attr_dict['@t']
    attr_dict['@b'] = haligned_blocks[0].attr_dict['@b']
    attr_dict['@r'] = haligned_blocks[-1].attr_dict['@r']
    attr_dict['type'] = 'table-haligned'
    # we can combine block1+2's attr_dict

    # Add the justified rows, each block is a row
    for tblock in justified_row_blocks:
        line_list = []  # type: List[AbbyLine]
        for unused_par_id, ab_par in enumerate(tblock.ab_pars):
            line_list.extend(ab_par.ab_lines)
        apar = AbbyPar(line_list, {})
        cell_list = []  # type: List(AbbyCell)
        cell_list.append(AbbyCell([apar], {}))

        if tblock.infer_attr_dict.get('header'):
            infer_attr_dict['header'] = True
        if tblock.infer_attr_dict.get('footer'):
            infer_attr_dict['footer'] = True

        row_list.append(AbbyRow(cell_list, {}))

    table_block = AbbyTableBlock(row_list, attr_dict)

    if IS_DEBUG_TABLE:
        abbyutils.infer_ab_block_is_header_footer(table_block)

        html_table = table_block_to_html(table_block)
        if not (table_block.infer_attr_dict.get('header') or \
                table_block.infer_attr_dict.get('footer')):
            print('\nhtml_table:')
            print(html_table)

    return table_block


def merge_aligned_blocks_old(haligned_blocks: List[AbbyTextBlock]) \
                             -> AbbyTableBlock:
    # there is probably a more concise way of expressing this in python, 5345
    haligned_blocks = sorted(haligned_blocks, key=block_get_attr_left)
    cell_list = []
    # is_header = False
    # is_footer = False
    infer_attr_dict = {}
    for tblock in haligned_blocks:
        line_list = []  # type: List[AbbyLine]
        for unused_par_id, ab_par in enumerate(tblock.ab_pars):
            line_list.extend(ab_par.ab_lines)
        apar = AbbyPar(line_list, {})
        cell_list.append(AbbyCell([apar], {}))

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
    row1 = AbbyRow(cell_list, {})

    table_block = AbbyTableBlock([row1], attr_dict)
    table_block.infer_attr_dict = infer_attr_dict

    return table_block


def is_all_pars_align_justified(ab_pars: List[AbbyPar]) -> bool:
    if not ab_pars:
        return False
    is_all_justified = True
    for ab_par in ab_pars:
        if ab_par.attr_dict.get('@align', 'None') != 'Justified':
            return False
    return True


def collect_justified_lines_after(ab_blocks: List[AbbyBlock],
                                  page_block_seq: int,
                                  first_block_left: int) \
                                  -> List[AbbyTextBlock]:
    if page_block_seq >= len(ab_blocks):
        return []

    result = []  # type: List[AbbyTextBlock]
    cur_ab_block = ab_blocks[page_block_seq]
    for ab_block in ab_blocks[page_block_seq + 1:]:
        if not isinstance(ab_block, AbbyTextBlock):
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


# pylint: disable=too-many-locals, too-many-branches
def merge_haligned_block_as_table(ab_doc: AbbyXmlDoc) -> None:

    for unused_pnum, abby_page in enumerate(ab_doc.ab_pages, 1):

        print("merge_haligned_blocks_as_table, page #{}".format(unused_pnum))
        print("        len(abby_page.ab_blocks = {}".format(len(abby_page.ab_blocks)))
        # find all the blocks with similar @b and @t
        ab_text_block_list = abbyutils.get_only_text_blocks(abby_page.ab_blocks)

        # print("\nmerge_adj_aligned_block_as_table")
        if not ab_text_block_list:
            continue

        haligned_blocks_list = []  # type: List[List[AbbyTextBlock]]
        # skip_blocks are the blocks that have already been found to be haligned
        skip_blocks = []  # type: List[AbbyTextBlock]
        for i, ab_text_block in enumerate(ab_text_block_list):
            if ab_text_block in skip_blocks:
                continue

            attr_dict = ab_text_block.attr_dict
            top = attr_dict['@t']
            bot = attr_dict['@b']
            cur_blocks = []  # type: List[AbbyTextBlock]
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

        out_block_list = []  # type: List[AbbyBlock]
        haligned_block_list_map = {}  # type: Dict[AbbyTextBlock, List[AbbyTextBlock]]
        justified_row_block_list_map = {}  # type: Dict[AbbyTextBlock, List[AbbyTextBlock]]
        for blocks in haligned_blocks_list:

            # found the column header, but no data
            # if len(blocks) <= 2:
            # jshaw, work
            # merge next text block if it is "justified"
            # most likely an minor column delimitation error from abby
            first_block = blocks[0]
            first_block_left = first_block.attr_dict['@l']
            last_block = blocks[-1]
            additional_table_row_blocks = \
                collect_justified_lines_after(abby_page.ab_blocks,
                                              last_block.page_block_seq,
                                              first_block_left)

            haligned_block_list_map[blocks[0]] = blocks
            justified_row_block_list_map[blocks[0]] = additional_table_row_blocks

        # now we have a list of haligned blocks
        for ab_block in abby_page.ab_blocks:
            if ab_block in skip_blocks:
                continue

            if isinstance(ab_block, AbbyTextBlock):
                haligned_blocks = haligned_block_list_map.get(ab_block, [])
                justified_row_blocks = justified_row_block_list_map.get(ab_block, [])

                if haligned_blocks:
                    table = merge_aligned_blocks(haligned_blocks, justified_row_blocks)
                    out_block_list.append(table)
                else:
                    # print("      block attr: {}".format(ab_block.attr_dict))
                    out_block_list.append(ab_block)
            else:
                # for tables
                # print("      block attr: {}".format(ab_block.attr_dict))
                out_block_list.append(ab_block)

        abby_page.ab_blocks = out_block_list
