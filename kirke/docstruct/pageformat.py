from collections import defaultdict
# pylint: disable=unused-import
from typing import Dict, List, Tuple

from kirke.docstruct.pdfoffsets import PageInfo3, PDFTextDoc, StrInfo
from kirke.utils import mathutils

# TODO, is this still used?
from kirke.docstruct.pdfoffsets import PageFormatStatus

IS_DEBUG = False

# Page Size in Point at 72 dpi
# letter size: width 8.5 in, height 11 in
HZ_10TH_DIV = 61.2   # 612.0 / 10, weidth
MAX_Y = 792.0
ONE_THIRD_MAX_Y = 792.0 * 2 / 3

"""
pformat_classifier = pageformat.PageFormatClassifier()

def is_double_spaced_doc(pdftxt_doc) -> bool:
    "Classifying a double_spaced_no_period_document."

    num_page_double_spaced_no_period = 0
    for page in pdftxt_doc.page_list:
        # assume add_doc_structure_to_page(page, pdftxt_doc) has already been applied
        page.page_format = pformat_classifier.classify(page, pdftxt_doc.doc_text)
        if page.page_format.is_double_spaced_no_period():
            num_page_double_spaced_no_period += 1
        # break blocks if they are in the middle of header, english sents
        # adjust_blocks_in_page(page, pdftxt_doc)


    if len(pdftxt_doc.page_list) >= 4 and \
       num_page_double_spaced_no_period / len(pdftxt_doc.page_list) >= 0.75:
        # allowing first page to be title page and not too much text
        # pdftxt_doc.is_one_paragraph_per_page = True
        return True
    return False
"""


# pylint: disable=too-few-public-methods
class PageFormatClassifier:

    def __init__(self) -> None:
        pass

    # pylint: disable=no-self-use
    def classify(self,
                 apage: PageInfo3,
                 text: str) \
                 -> PageFormatStatus:
        pformat = calc_page_stats(apage, text)

        return pformat


# pylint: disable=too-many-locals
def calc_page_stats(apage: PageInfo3, doc_text: str) -> PageFormatStatus:

    # page_start, unused_page_end = apage.start, apage.end
    page_text = doc_text[apage.start:apage.end].strip()
    # print('\npage #{}, len(page_text) = {}'.format(apage.page_num, len(page_text)))
    num_new_lines = page_text.count('\n')
    # print('  num_new_lines = {}'.format(num_new_lines))
    num_periods = page_text.count('.\n') + page_text.count('. ')
    if page_text.endswith('.'):
        num_periods += 1

    # print('  num_periods = {}'.format(num_periods))

    hz_col_count_map = defaultdict(int)  # type: Dict[int, int]
    hz_len_count_map = defaultdict(int)  # type: Dict[int, int]
    num_line = 0
    block_list = apage.get_blocked_lines()
    unused_prev_yend = 0
    # pylint: disable=invalid-name
    unused_len_col_ge_6_before_1third_page = 0
    print('calc_page_stats, page_num = {}, num_block = {}'.format(apage.page_num,
                                                                  len(block_list)))
    for block_seq, lines_with_attrs in enumerate(block_list):
        if IS_DEBUG:
            print('  j1  page_para_seq= {}'.format(block_seq))
        para_lines = []  # type: List[str]
        for line_attrs in lines_with_attrs:
            line_info = line_attrs.lineinfo
            if IS_DEBUG:
                # pylint: disable=line-too-long
                print('  page={}, ln={} bnum= {} se=({}, {}) xstart={}, ltext=[{}]'.format(apage.page_num,
                                                                                           line_attrs.page_line_num,
                                                                                           line_attrs.block_num,
                                                                                           line_info.start,
                                                                                           line_info.end,
                                                                                           line_info.xStart,
                                                                                           line_attrs.line_text))
            # print('   verify_text [{}]'.format(text[line_info.start:line_info.end]))
            para_lines.append(doc_text[line_info.start:line_info.end])


            hz_start_nth = round(line_info.xStart / HZ_10TH_DIV)
            hz_end_nth = round(line_info.xEnd / HZ_10TH_DIV)
            hz_nth_len = round((line_info.xEnd - line_info.xStart) / HZ_10TH_DIV)

            if IS_DEBUG:
                print("    hz_start_nth= {}, hz_end_nth={}, hz_nth_len= {}".format(hz_start_nth,
                                                                                   hz_end_nth,
                                                                                   hz_nth_len))

            hz_col_count_map[hz_start_nth] += 1
            hz_len_count_map[hz_nth_len] += 1
            num_line += 1


    unused_is_one_para_per_page = False
    if len(page_text) > 1500 and \
       num_new_lines == 0:
        # pylint: disable=unused-variable
        is_one_para_per_page = True

        pformat = PageFormatStatus(is_one_para_per_page=True,
                                   has_periods=False)
        return pformat

    """
    if is_one_para_per_page and \
       num_periods < 5:
        pformat = PageFormatStatus(is_one_para_per_page=True,
                                   has_periods=False)
        return pformat
    """
    # return a normal page
    return PageFormatStatus()


"""
class PageFormatStatus:

    def __init__(self,
                 page_num: int,
                 num_lines: int) -> None:
        self.page_num = page_num
        self.num_lines = num_lines
        self.num_columns = 0  # default, 1, 2, 3, 0=unknown

    def __str__(self):
        out_st = '{}'.format(self.page_num)
        return out_st
"""


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


# pylint: disable=too-many-statements
def calc_one_page_format(page_num: int,
                         lxid_strinfos_list: List[Tuple[int, List[StrInfo]]],
                         nl_text: str,
                         page_ydiff_mode_map: Dict[int, float],
                         failed_page_ydiff_mode_pages: List[int],
                         prev_page_num_col: int) \
                         -> int:
    """Return the number of column in the page."""

    IS_DEBUG = True
    unused_last_linenum_in_page = lxid_strinfos_list[-1][0]
    num_lines = len(lxid_strinfos_list)

    print('\n=calc_one_page_format  @page {}, num_lines = {}, prev_page_num_col = {}'.format(page_num,
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
        # pylint: disable=line-too-longpp
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

    print('\n    num_col_len_le_3 = {}, perc = {}%'.format(num_col_len_le_3, num_col_len_le_3_perc))

    num_col_in_page = -1
    is_template_page = False
    # first check if template
    if len(hz_col_count_map) >= 4 and \
       num_lines > 50 and \
       num_col_len_le_3_perc >= 50 and\
       sorted_freq_col_perc_list[1][2] > 12 and \
       sorted_freq_col_perc_list[2][2] > 12 and \
       sorted_freq_col_perc_list[3][2] > 10:
        print("\n  ===>>> page {}, page_format: template, br 1".format(page_num))
        num_col_in_page = 0
        is_template_page = True
    elif num_lines >= 200:
        if len(hz_col_count_map) >= 3 and \
           num_col_len_le_3_perc > 75 and \
           sorted_freq_col_perc_list[0][0] > 60 and \
           sorted_freq_col_perc_list[1][0] > 60 and \
           sorted_freq_col_perc_list[2][0] > 60:
            print("\n  ===>>> page {}, page_format: 3 columns, br 1".format(page_num))
            num_col_in_page = 3
        # below is a case where 2 and 3 column are often joined
        elif len(hz_col_count_map) >= 2 and \
             num_col_len_le_3_perc > 58 and \
             sorted_freq_col_perc_list[0][0] > 90 and \
             sorted_freq_col_perc_list[1][0] > 90:
            print("\n  ===>>> page {}, page_format: 3 columns, br 2".format(page_num))
            num_col_in_page = 3
        else:
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

            print("\n  ===>>> page {}, page_format: 2 columns, br 1".format(page_num))
            # pformat = PageFormatStatus(num_column=2)
            num_col_in_page = 2
        elif top_col_perc >= 35 and \
             top_2nd_col_perc >= 35 and \
             ((top_col in set([1, 2]) and top_2nd_col in set([4, 5, 6])) or
              (top_col in set([4, 5, 6]) and top_2nd_col in set([1, 2]))) and \
             top_col_perc + top_2nd_col_perc > 80:   # doc 8921, page 8, 52 + 38
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
                print("\n  ===>>> page {}, page_format: 3 columns, br 3.3".format(page_num))
                num_col_in_page = 3
            else:
                print("\n  ===>>> page {}, page_format: 2 columns, br 1.3".format(page_num))
                num_col_in_page = 2

    if num_col_in_page == -1:
        print("\n  ===>>> page {}, page_format: 1 column, br default".format(page_num))
        num_col_in_page = 1


    valid_line_nth_len = 5
    if top_col_len < 5:
        valid_line_nth_len = top_col_len

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
            # only increment prev_y is not empty
            prev_y = lx_ystart
            prev_hz_nth_len = hz_nth_len
        else:
            y_diff = -1
            if IS_DEBUG:
                print('     y_diff = {}, current_y = {}, prev_y = {}'.format(y_diff,
                                                                             lx_ystart,
                                                                             prev_y))

        if prev_hz_nth_len >= valid_line_nth_len and \
           y_diff != -1:
            full_ydiff_count_map[y_diff] += 1

        if y_diff != -1:
            ydiff_count_map[y_diff] += 1


    perc = round(nth_len_ge_5 / num_lines * 100.0)
    print('    >> nth_len_ge_5 = {}, perc={}%'.format(nth_len_ge_5, perc))
    print()

    # pylint: disable=line-too-long
    sorted_count_full_ydiff_list = sorted(((count, ydiff) for ydiff, count in full_ydiff_count_map.items()),
                                          reverse=True)
    sorted_count_ydiff_list = sorted(((count, ydiff) for ydiff, count in ydiff_count_map.items()),
                                     reverse=True)
    IS_DEBUG = True
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
        if sorted_count_full_ydiff_list[0][0] < 5:
            # too infrequent
            page_ydiff_mode_map[page_num] = sorted_count_full_ydiff_list[0][1]
            failed_page_ydiff_mode_pages.append(page_num)
        elif sorted_count_full_ydiff_list[0][1] >= 22:
            # too big
            page_ydiff_mode_map[page_num] = sorted_count_full_ydiff_list[0][1]
            failed_page_ydiff_mode_pages.append(page_num)
        else:
            # ok
            page_ydiff_mode_map[page_num] = sorted_count_full_ydiff_list[0][1]
    else:
        # for title page, or last page in a doc?
        page_ydiff_mode_map[page_num] = sorted_count_ydiff_list[0][1]
        failed_page_ydiff_mode_pages.append(page_num)

    print('  Done.  calc_one_page_format()')
    return num_col_in_page



# pylint: disable=too-many-branches, too-many-statements
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


    print('page_ydiff_mode_map:')
    page_num_set = set(page_num_list)
    page_ydiff_mode_list = []  # type: List[float]
    for page_num in range(max(page_num_list) + 1):
        if page_num not in page_num_set:
            page_ydiff_mode_list.append(0)
        else:
            page_ydiff_mode_list.append(page_ydiff_mode_map[page_num])
            if page_num in failed_page_ydiff_mode_pages:
                print('     page {}: {}, failed'.format(page_num, page_ydiff_mode_map[page_num]))
            else:
                print('     page {}: {}'.format(page_num, page_ydiff_mode_map[page_num]))

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


    print('failed_page_ydiff_mode_pages: {}'.format(failed_page_ydiff_mode_pages))
    # adjust page_ydiff_mode using neighbors
    for page_num in failed_page_ydiff_mode_pages:
        if page_num < 3:
            print('taking post {} to {}'.format(page_num-3, page_num))
            page_ydiff_mode_list[page_num] = \
                mathutils.get_mode_in_list(page_ydiff_mode_list[page_num+1:page_num+4])
        else:
            print('taking prev {} to {}'.format(page_num-3, page_num))
            page_ydiff_mode_list[page_num] = \
                mathutils.get_mode_in_list(page_ydiff_mode_list[page_num-3:page_num])

    for page_num in failed_page_ydiff_mode_pages:
        page_ydiff_mode_map[page_num] = page_ydiff_mode_list[page_num]

    print('\nadjusted page_ydiff_mode_map:')
    for page_num in page_num_list:
        if page_num in failed_page_ydiff_mode_pages:
            print('     page {}: {}, failed'.format(page_num, page_ydiff_mode_map[page_num]))
        else:
            print('     page {}: {}'.format(page_num, page_ydiff_mode_map[page_num]))


    doc_ydiff_mode = mathutils.get_mode_in_list(all_ydiffs)

    return doc_ydiff_mode, page_ydiff_mode_map
