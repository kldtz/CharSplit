from collections import defaultdict
# pylint: disable=unused-import
from typing import Dict, List, Tuple

from kirke.docstruct.pdfoffsets import PageInfo3, PDFTextDoc, StrInfo
from kirke.utils import mathutils

IS_DEBUG = False

IS_TOP_LEVEL_DEBUG = False

# Page Size in Point at 72 dpi
# letter size: width 8.5 in, height 11 in
HZ_10TH_DIV = 61.2   # 612.0 / 10, weidth
MAX_Y = 792.0
ONE_THIRD_MAX_Y = 792.0 * 2 / 3

MIN_FULL_YDIFF = 3


def get_lx_min_max_x(strinfo_list: List[StrInfo]) -> Tuple[int, int]:
    min_x, max_x = 1000, 0
    for strinfo in strinfo_list:
        if strinfo.xEnd > max_x:
            max_x = strinfo.xEnd
        if strinfo.xStart < min_x:
            min_x = strinfo.xStart
    return min_x, max_x


def merge_adjacent_hz_cols(freq_col_list: List[Tuple[int, int]]) \
    -> List[Tuple[int, int]]:
    """Merge the 3rd col with 1st or 2nd frequent column if their
       difference is only 1.
       If first and 2nd diff is less than or equals to 2, abort."""
    if len(freq_col_list) < 3:
        return freq_col_list
    col_1_freq, col_1 = freq_col_list[0]
    col_2_freq, col_2 = freq_col_list[1]
    col_3_freq, col_3 = freq_col_list[2]

    if abs(col_1 - col_2) <= 2:
        return freq_col_list

    if abs(col_3 - col_1) == 1:
        first_col_freq = (col_1_freq + col_3_freq, col_1)
        return [first_col_freq, freq_col_list[1]] + freq_col_list[3:]
    elif abs(col_3 - col_2) == 1:
        second_col_freq = (col_2_freq + col_3_freq, col_2)
        return [freq_col_list[0], second_col_freq]  + freq_col_list[3:]
    return freq_col_list


# pylint: disable=too-many-statements, too-many-arguments, too-many-locals
def calc_one_page_format(page_num: int,
                         lxid_strinfos_list: List[Tuple[int, List[StrInfo]]],
                         nl_text: str,
                         page_ydiff_mode_map: Dict[int, float],
                         failed_page_ydiff_mode_pages: List[int],
                         prev_page_num_col: int) \
                         -> int:
    """Return the number of column in the page."""

    unused_last_linenum_in_page = lxid_strinfos_list[-1][0]
    num_lines = len(lxid_strinfos_list)

    if IS_DEBUG:
        print('\n=calc_one_page_format  @page {}, num_lines = {},'
              'prev_page_num_col = {}'.format(page_num,
                                              num_lines,
                                              prev_page_num_col))
        print()

    hz_len_count_map = defaultdict(int)  # type: Dict[int, int]
    hz_col_count_map = defaultdict(int)  # type: Dict[int, int]

    # store this for later use in this function
    line_hz_startnth_nthlen_list = []  # type: List[Tuple[int, int]]
    for unused_line_num, lxid_strinfos in lxid_strinfos_list:
        lx_min_x, lx_max_x = get_lx_min_max_x(lxid_strinfos)

        hz_start_nth = round(lx_min_x / HZ_10TH_DIV)
        hz_end_nth = round(lx_max_x / HZ_10TH_DIV)
        hz_nth_len = round((lx_max_x - lx_min_x) / HZ_10TH_DIV)

        line_hz_startnth_nthlen_list.append((hz_start_nth,
                                             hz_nth_len))

        hz_len_count_map[hz_nth_len] += 1
        hz_col_count_map[hz_start_nth] += 1

    # pylint: disable=line-too-long
    sorted_freq_col_list = sorted(((freq, nth_col)
                                   for nth_col, freq in hz_col_count_map.items()),
                                  reverse=True)
    sorted_freq_col_perc_list = sorted(((freq, nth_col, round(freq / num_lines * 100.0))
                                        for nth_col, freq in hz_col_count_map.items()),
                                       reverse=True)
    if IS_DEBUG:
        # pylint: disable=line-too-long
        for freq, nth_col, perc in sorted(((freq, nth_col, round(freq / num_lines * 100.0))
                                           for nth_col, freq in hz_col_count_map.items()),
                                          reverse=True):
            print("    >>>> hz_col= {}, freq={}, perc={}%".format(nth_col, freq, perc))
        print()

    # pylint: disable=line-too-long
    sorted_freq_hzlen_list = sorted(((freq, nth_col) for nth_col, freq in hz_len_count_map.items()),
                                    reverse=True)
    if IS_DEBUG:
        # pylint: disable=line-too-long
        for freq, nth_col in sorted(((freq, nth_col) for nth_col, freq in hz_len_count_map.items()),
                                    reverse=True):
            perc = round(freq / num_lines * 100.0)
            print("    >>>> hz_len= {}, freq={}, perc={}%".format(nth_col, freq, perc))
        print()

    merge_adj_col_count_list = merge_adjacent_hz_cols(sorted_freq_col_list)

    if IS_DEBUG:
        for freq, nth_col in merge_adj_col_count_list:
            perc = round(freq / num_lines * 100.0)
            print("    >>>> merged hz_col= {}, freq={}, perc={}%".format(nth_col, freq, perc))
        print()

    top_col_freq, top_col = 0, -1
    top_col_perc = 0
    if merge_adj_col_count_list:
        top_col_freq, top_col = merge_adj_col_count_list[0]
        top_col_perc = round(top_col_freq / num_lines * 100.0)

    top_2nd_col_freq, top_2nd_col = 0, -1
    top_2nd_col_perc = 0
    if len(merge_adj_col_count_list) > 1:
        top_2nd_col_freq, top_2nd_col = merge_adj_col_count_list[1]
        top_2nd_col_perc = round(top_2nd_col_freq / num_lines * 100.0)

    sorted_col_len_list = sorted_freq_hzlen_list
    top_col_len = 0
    if sorted_col_len_list:
        _, top_col_len = sorted_col_len_list[0]
    top_2nd_col_len = -1
    if len(sorted_col_len_list) > 1:
        _, top_2nd_col_len = sorted_col_len_list[1]

    if IS_DEBUG:
        print('    top_col = {}, top_col_perc = {}'.format(top_col, top_col_perc))
        print('    top_2nd_col = {}, top_2nd_col_perc = {}'.format(top_2nd_col, top_2nd_col_perc))
        print()
        print('    top_col_len_ = {}'.format(top_col_len))
        print('    top_2nd_col_len = {}'.format(top_2nd_col_len))

    num_col_len_le_3 = hz_len_count_map.get(0, 0) + \
                       hz_len_count_map.get(1, 0) + \
                       hz_len_count_map.get(2, 0) + \
                       hz_len_count_map.get(3, 0)
    num_col_len_le_3_perc = num_col_len_le_3 / num_lines * 100.0

    num_col_len_ge_5 = hz_len_count_map.get(5, 0) + \
                       hz_len_count_map.get(6, 0) + \
                       hz_len_count_map.get(7, 0) + \
                       hz_len_count_map.get(8, 0) + \
                       hz_len_count_map.get(9, 0) + \
                       hz_len_count_map.get(10, 0)
    num_col_len_ge_5_perc = num_col_len_ge_5 / num_lines * 100.0

    if IS_DEBUG:
        print('\n    num_col_len_le_3 = {}, perc = {}%'.format(num_col_len_le_3, num_col_len_le_3_perc))
        print('\n    num_col_len_ge_5 = {}, perc = {}%'.format(num_col_len_ge_5, num_col_len_ge_5_perc))

    num_col_in_page = -1
    unused_is_template_page = False
    # first check if template
    if len(hz_col_count_map) >= 4 and \
       num_lines > 50 and \
       num_col_len_le_3_perc >= 50 and\
       sorted_freq_col_perc_list[1][2] > 12 and \
       sorted_freq_col_perc_list[2][2] > 12 and \
       sorted_freq_col_perc_list[3][2] > 10:
        if IS_DEBUG:
            print("\n  ===>>> page {}, page_format: template, br 1".format(page_num))
        num_col_in_page = 0
        unused_is_template_page = True
    elif num_lines >= 200:
        if len(hz_col_count_map) >= 3 and \
           num_col_len_le_3_perc > 75 and \
           sorted_freq_col_perc_list[0][0] > 60 and \
           sorted_freq_col_perc_list[1][0] > 60 and \
           sorted_freq_col_perc_list[2][0] > 60:
            if IS_DEBUG:
                print("\n  ===>>> page {}, page_format: 3 columns, br 1".format(page_num))
            num_col_in_page = 3
        # below is a case where 2 and 3 column are often joined
        elif len(hz_col_count_map) >= 2 and \
             num_col_len_le_3_perc > 58 and \
             sorted_freq_col_perc_list[0][0] > 90 and \
             sorted_freq_col_perc_list[1][0] > 90:
            if IS_DEBUG:
                print("\n  ===>>> page {}, page_format: 3 columns, br 2".format(page_num))
            num_col_in_page = 3
        else:
            if IS_DEBUG:
                print("\n  ===>>> page {}, page_format: 3 columns, br 3".format(page_num))
            num_col_in_page = 3
    elif top_col_len == 3 or top_col_len == 4:
        """
        # determine the column size
        if top_col == 1 and top_col_perc > 85:
            print("\n  ===>>> page {}, page_format: 3 columns, br 2".format(page_num))
            num_col_in_page = 3
            # pformat = PageFormatStatus(num_column=3)
        # this is for 2 column detection
        """
        if top_col_perc > 37 and \
             top_2nd_col_perc >= 37 and \
             ((top_col in set([1, 2]) and top_2nd_col in set([4, 5, 6])) or
              (top_col in set([4, 5, 6]) and top_2nd_col in set([1, 2]))):

            # for 2 column, I have seen 2, 5, 6 as top cols
            if IS_DEBUG:
                print("\n  ===>>> page {}, page_format: 2 columns, br 1".format(page_num))
            # pformat = PageFormatStatus(num_column=2)
            num_col_in_page = 2
        elif top_col_perc >= 35 and \
             top_2nd_col_perc >= 35 and \
             ((top_col in set([1, 2]) and top_2nd_col in set([4, 5, 6])) or
              (top_col in set([4, 5, 6]) and top_2nd_col in set([1, 2]))) and \
             top_col_perc + top_2nd_col_perc > 80:   # doc 8921, page 8, 52 + 38
            if IS_DEBUG:
                print("\n  ===>>> page {}, page_format: 2 columns, br 1.2".format(page_num))
            num_col_in_page = 2
            # pformat = PageFormatStatus(num_column=2)
            # elif top_col == 2 and top_col_perc > 85:
            #     print("===>>> page {}, page_format: 2 columns, br2.234".format(apage.page_num))
            #     pformat = PageFormatStatus(num_column=2)
        elif num_col_len_le_3_perc > 80 and \
             top_col in set([1, 2]):
            # just one column
            if prev_page_num_col == 3:
                if IS_DEBUG:
                    print("\n  ===>>> page {}, page_format: 3 columns, br 3.3".format(page_num))
                num_col_in_page = 3
            else:
                if IS_DEBUG:
                    print("\n  ===>>> page {}, page_format: 2 columns, br 1.3".format(page_num))
                num_col_in_page = 2

    if num_col_in_page == -1:
        if IS_DEBUG:
            print("\n  ===>>> page {}, page_format: 1 column, br default".format(page_num))
        num_col_in_page = 1

    valid_line_nth_len = 5
    # we don't really care about number of column in a page, just
    # if top_col_len < 5:
    # For some pages, there might not be enough lines with full text.
    # Even for those with full text, y-diff is only computed between
    # adjacent lines.  So we set long line check value to 7.
    if num_col_len_ge_5 < MIN_FULL_YDIFF or \
       num_col_in_page in set([2, 3]):
        # check if the page is 1-column or 0=template
        # if num_col_in_page not in set([0, 1]):
        if top_col_len == 1:
            # top_col_len == 1 is never the correct solution
            valid_line_nth_len = 2
        else:
            valid_line_nth_len = top_col_len
    if IS_DEBUG:
        print('    top_col_len = {}, valid_line_nth_len = {}'.format(top_col_len, valid_line_nth_len))
        print()

    full_hz_col_count_map = defaultdict(int)  # type: Dict[int, int]

    full_ydiff_count_map = defaultdict(int)  # type: Dict[float, int]
    ydiff_count_map = defaultdict(int)  # type: Dict[float, int]

    prev_y = 1000
    prev_hz_nth_len = 0
    nth_len_ge_5 = 0
    for (unused_line_num, lxid_strinfos), (hz_start_nth, hz_nth_len) \
        in zip(lxid_strinfos_list, line_hz_startnth_nthlen_list):
        lx_min_x, lx_max_x = get_lx_min_max_x(lxid_strinfos)
        lx_ystart = round(lxid_strinfos[0].yStart, 2)

        # only for long lines
        if hz_nth_len >= valid_line_nth_len:
            full_hz_col_count_map[hz_start_nth] += 1
            nth_len_ge_5 += 1

        tmp_start = lxid_strinfos[0].start
        tmp_end = lxid_strinfos[-1].end
        line_len = len(nl_text[tmp_start:tmp_end].strip())

        # pylint: disable=invalid-name
        IS_DETAIL_DEBUG = False
        if IS_DETAIL_DEBUG:
            print('   x3 line: [{}]'.format(nl_text[tmp_start:tmp_end]))
            print('     hz_start_nth: {}, end_nth: {}, hz_nth_len: {}'.format(hz_start_nth,
                                                                              hz_end_nth,
                                                                              hz_nth_len))
        if line_len > 0 and \
           not nl_text[tmp_start:tmp_end].isspace():
            y_diff = mathutils.half_round(lxid_strinfos[0].yStart - prev_y)
            if y_diff <= 0:
                y_diff = -1
            if IS_DEBUG:
                print('     y_diff = {}, current_y = {}, prev_y = {}'.format(y_diff,
                                                                             lx_ystart,
                                                                             prev_y))
        else:
            y_diff = -1
            if IS_DEBUG:
                print('     y_diff = {}, current_y = {}, prev_y = {}'.format(y_diff,
                                                                             lx_ystart,
                                                                             prev_y))

        if IS_DEBUG:
            print('  prev_hz_nth_len = {}, valid_line_nth_len = {}, y_diff = {}'.format(
                prev_hz_nth_len, valid_line_nth_len, y_diff))
        if prev_hz_nth_len >= valid_line_nth_len and \
           y_diff != -1:
            full_ydiff_count_map[y_diff] += 1

        if y_diff != -1:
            ydiff_count_map[y_diff] += 1

        if line_len > 0 and \
           not nl_text[tmp_start:tmp_end].isspace():
            # only increment prev_y is not empty
            prev_y = lx_ystart
            prev_hz_nth_len = hz_nth_len


    perc = round(nth_len_ge_5 / num_lines * 100.0)
    if IS_DEBUG:
        print('    >> nth_len_ge_5 = {}, perc={}%'.format(nth_len_ge_5, perc))
        print()

    # pylint: disable=line-too-long
    sorted_count_full_ydiff_list = sorted(((count, ydiff) for ydiff, count in full_ydiff_count_map.items()),
                                          reverse=True)
    sorted_count_ydiff_list = sorted(((count, ydiff) for ydiff, count in ydiff_count_map.items()),
                                     reverse=True)
    if IS_DEBUG:
        total_full_ydiff_count = 0
        for freq, ydiff in sorted(((count, ydiff) for ydiff, count in full_ydiff_count_map.items()),
                                  reverse=True):
            total_full_ydiff_count += freq
        for freq, ydiff in sorted(((count, ydiff) for ydiff, count in full_ydiff_count_map.items()),
                                  reverse=True):
            perc = round(freq /  total_full_ydiff_count * 100.0)
            print("    >>>> full_ydiff= {}, freq={}, perc={}%".format(ydiff, freq, perc))
        print()


        total_ydiff_count = 0
        for freq, ydiff in sorted(((count, ydiff) for ydiff, count in ydiff_count_map.items()),
                                  reverse=True):
            total_ydiff_count += freq
        for freq, ydiff in sorted(((count, ydiff) for ydiff, count in ydiff_count_map.items()),
                                  reverse=True):
            perc = round(freq /  total_ydiff_count * 100.0)
            print("    >>>> ydiff= {}, freq={}, perc={}%".format(ydiff, freq, perc))
        print()

    if sorted_count_full_ydiff_list:
        # this 5 is not related to valid_line_nth_len
        if sorted_count_full_ydiff_list[0][0] < MIN_FULL_YDIFF:
            # if sorted_count_full_ydiff_list[0][0] < MIN_FULL_YDIFF and \
            #    not (len(sorted_count_full_ydiff_list) >= 2 and \
            #    abs(sorted_count_full_ydiff_list[0][1] - sorted_count_full_ydiff_list[1][1]) < 1):
            # too infrequent
            # 2nd check if to NOT fail those with enough info, doc 9326, page 55
            page_ydiff_mode_map[page_num] = sorted_count_full_ydiff_list[0][1]
            failed_page_ydiff_mode_pages.append(page_num)
            if IS_DEBUG:
                print('failed page {}, branch 1'.format(page_num))
        # elif sorted_count_full_ydiff_list[0][1] >= 22:
        elif sorted_count_full_ydiff_list[0][1] >= 28:  # have seen 27,5, doc 9325, page 64
            # too big
            page_ydiff_mode_map[page_num] = sorted_count_full_ydiff_list[0][1]
            failed_page_ydiff_mode_pages.append(page_num)
            if IS_DEBUG:
                print('failed page {}, branch 2'.format(page_num))
        else:
            # ok
            """
            if len(sorted_count_full_ydiff_list) >= 2 and \
               abs(sorted_count_full_ydiff_list[0][0] - sorted_count_full_ydiff_list[1][0]) <= 2 and \
               abs(sorted_count_full_ydiff_list[0][1] - sorted_count_full_ydiff_list[1][1]) < 1:
                page_ydiff_mode_map[page_num] = max(sorted_count_full_ydiff_list[0][1],
                                                    sorted_count_full_ydiff_list[1][1])
            else:
            """
            page_ydiff_mode_map[page_num] = sorted_count_full_ydiff_list[0][1]
    else:
        if sorted_count_ydiff_list:
            # for title page, or last page in a doc?
            page_ydiff_mode_map[page_num] = sorted_count_ydiff_list[0][1]
        else:
            page_ydiff_mode_map[page_num] = -1
        failed_page_ydiff_mode_pages.append(page_num)
        if IS_DEBUG:
            print('failed page {}, branch 3'.format(page_num))

    if IS_DEBUG:
        print('  Done.  calc_one_page_format()')
    return num_col_in_page


def pick_page_adjacent_ydiff_mode(ydiff_mode_list: List[float],
                                  global_ydiff: float,
                                  is_after_page: bool = False) -> float:
    """This specifically handle the case where the 3 ydiff in
       previous 3 page before or after doesn't agree.

       Previous approach of taking the most frequent failed when
       all 3 page ydiff are distinct.  Now this is deterministic.
    """
    # print('pick_page_adjacent_ydiff_mode: {}'.format(ydiff_mode_list))
    if not ydiff_mode_list:
        return global_ydiff

    # print("ydiff_mode_list: {}".format(ydiff_mode_list))
    count_dict = defaultdict(int)
    for ydiff_mode in ydiff_mode_list:
        count_dict[ydiff_mode] += 1
    count_ydiff_list = sorted(((freq, ydiff) for ydiff, freq in count_dict.items()),
                              reverse=True)
    # if multiple pages say the X, we take X
    if count_ydiff_list[0][0] > 1:
        return count_ydiff_list[0][1]
    if is_after_page:
        # this is only for first 3 pages
        maybe_result = count_ydiff_list[0][1]
    else:
        # take the last page
        maybe_result = count_ydiff_list[-1][1]
    # if all else failed, take the global_ydiff
    if maybe_result == -1:
        maybe_result = global_ydiff
    return maybe_result


# pylint: disable=too-many-branches, too-many-statements, too-many-locals
def calc_page_formats(page_linenum_list_map: Dict[int, List[int]],
                      lxid_strinfos_map: Dict[int, List[StrInfo]],
                      nl_text: str,
                      all_ydiffs: List[float]) -> Tuple[int, Dict[int, float]]:
    """Return the following:

    doc_ydiff
    page_ydiff_map: has the ydiff for each page
    page_colnum_map: has the -1, 1-, 2-, or 3-column page
                     Not sure if this is necessary
    """
    page_num_list = sorted(page_linenum_list_map.keys())

    if IS_TOP_LEVEL_DEBUG:
        print('\n========= calc_page_formats()')
    page_ydiff_mode_map = {}  # type: Dict[int, float]
    failed_page_ydiff_mode_pages = []  # type: List[int]

    page_num_col_map = {}  # type: Dict[int, int]
    prev_page_num_col = -1
    for page_num in page_num_list:
        page_linenum_set = set(page_linenum_list_map[page_num])
        page_linenum_list = sorted(page_linenum_set)
        # Previously, there might be duplicated entries in page_linenum_list_map,
        # This is cleaned up version.
        page_linenum_list_map[page_num] = page_linenum_list

        lxid_strinfos_list = []  # type: List[Tuple[int, List[StrInfo]]]
        for line_num in page_linenum_list:
            lxid_strinfos = lxid_strinfos_map[line_num]
            lxid_strinfos_list.append((line_num, lxid_strinfos))

        page_num_col = calc_one_page_format(page_num,
                                            lxid_strinfos_list,
                                            nl_text,
                                            page_ydiff_mode_map,
                                            failed_page_ydiff_mode_pages,
                                            prev_page_num_col)
        page_num_col_map[page_num] = page_num_col
        prev_page_num_col = page_num_col

    if IS_DEBUG:
        print('page_ydiff_mode_map:')
    page_num_set = set(page_num_list)
    page_ydiff_mode_list = []  # type: List[float]
    for page_num in range(max(page_num_list) + 1):
        if page_num not in page_num_set:
            page_ydiff_mode_list.append(0)
        else:
            page_ydiff_mode_list.append(page_ydiff_mode_map[page_num])
            if IS_DEBUG:
                if page_num in failed_page_ydiff_mode_pages:
                    print('     page {}: {}, failed'.format(page_num,
                                                            page_ydiff_mode_map[page_num]))
                else:
                    print('     page {}: {}'.format(page_num, page_ydiff_mode_map[page_num]))

    if IS_TOP_LEVEL_DEBUG:
        print('page_ydiff_mode_list:')
        for tmp_page_num, tmp_ydiff in enumerate(page_ydiff_mode_list):
            if tmp_page_num == 0:
                continue
            print('   page {}\t{}'.format(tmp_page_num, tmp_ydiff))

        print('\npage_num_col_list:')
        for tmp_page_num in page_num_list:
            tmp_page_col_num = page_num_col_map[tmp_page_num]
            print('   page {} num_col = {}'.format(tmp_page_num, tmp_page_col_num))
        print()

    doc_ydiff_mode = mathutils.get_mode_in_list(all_ydiffs)

    if IS_TOP_LEVEL_DEBUG:
        print('failed_page_ydiff_mode_pages: {}'.format(failed_page_ydiff_mode_pages))
    # adjust page_ydiff_mode using neighbors
    for page_num in failed_page_ydiff_mode_pages:
        if page_num < 3:
            if IS_DEBUG:
                print('taking post {} to {}'.format(page_num-3, page_num))
            page_ydiff_mode_list[page_num] = \
                pick_page_adjacent_ydiff_mode(page_ydiff_mode_list[page_num+1:page_num+4],
                                              global_ydiff=doc_ydiff_mode,
                                              is_after_page=True)
                # mathutils.get_mode_in_list(page_ydiff_mode_list[page_num+1:page_num+4])
        else:
            if IS_DEBUG:
                print('taking prev {} to {}'.format(page_num-3, page_num))
            page_ydiff_mode_list[page_num] = \
                pick_page_adjacent_ydiff_mode(page_ydiff_mode_list[page_num-3:page_num],
                                              global_ydiff=doc_ydiff_mode)
                # mathutils.get_mode_in_list(page_ydiff_mode_list[page_num-3:page_num])
        if IS_DEBUG:
            print('    page_ydiff_mode_list[{}] = {}'.format(page_num,
                                                             page_ydiff_mode_list[page_num]))

    for page_num in failed_page_ydiff_mode_pages:
        page_ydiff_mode_map[page_num] = page_ydiff_mode_list[page_num]

    if IS_TOP_LEVEL_DEBUG:
        print('\nadjusted page_ydiff_mode_map:')
        for page_num in page_num_list:
            if page_num in failed_page_ydiff_mode_pages:
                print('     page {}: {}, failed'.format(page_num, page_ydiff_mode_map[page_num]))
            else:
                print('     page {}: {}'.format(page_num, page_ydiff_mode_map[page_num]))


    return doc_ydiff_mode, page_ydiff_mode_map
