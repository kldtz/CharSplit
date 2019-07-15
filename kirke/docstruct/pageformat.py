"""The goal of pageformat.py is to decide if the page level y-diff
should be used to segment paragraphs in a page.  To achieve this goal,
the system currently uses a somewhat complex algorithm based on
PDFBox's line information.  Because we expecte the block-level
information (in PDFBox or ABBYY) to be more effective than line-based
information, in the future, we might try to implement a different
algorithm based on block-level information instead.

This code is not used in the ebrevia's heuristics for recognizing
horizontally aligned text blocks as tables.

In PDFBOx's world, the object's location in a page is described using
points, with 72 points per inch.  In ABBYY's world, in contrast, uses
300 dots per inch to represent its object's coordinates.  To figure
out how many lines are in 1-column 2-column, 3-column mode, we convert
those PDFBOx lines into objects with more standard lengths and
locations, such as where that line belongs in a virtual page grid:

  - A page is divided into 10 vertical columns, with each column has
    61.2 points (8.5 inches * 72 points / 10 columns = 61.2 points per
    column).

  - A page is divided into 4 horizontal rows, with each row has 198
    points (11 inches * 72 points / 4 rows = 198 points per row).

We convert lines into an object with one of the 10 possible length
(how many horizontal columns), which nth column the line started, and
in which nth rows in the page that the line belongs to.  By counting
those line-based information, we apply a set of rules to determine if
a page is 1-column (default), 2-, 3-column or a form-template.  If a
page is determined to be a form-template, then page level y-diff
cannot be used to make paragraph segmentation decisions.  If a page is
NOT a form-template, the page-level ydiff would be applicable for
paragraph segmentation decisions.  The key here is that we mainly care
about if a page is a form-template or not, not really concerned with a
page is 1-, 2-, or 3-column, because in those pages, page y-diff is
still valid.

"""

from collections import defaultdict
import logging
# pylint: disable=unused-import
from typing import Dict, List, Tuple

from kirke.docstruct.pdfoffsets import PageInfo3, PDFTextDoc, StrInfo
from kirke.docstruct import pdfoffsets
from kirke.utils import mathutils

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IS_DEBUG = False
IS_PAGE_NUM_COL_DEBUG = False
IS_TOP_LEVEL_DEBUG = False

HZ_10TH_DIV = 61.2   # 612.0 / 10, width
ONE_THIRD_MAX_PAGE_Y = 792.0 * 2 / 3
VT_4TH_DIV = 792.0 / 4   # 792.0 / 4, height

MIN_FULL_YDIFF = 3

def xval_to_hz_nth(aval: float) -> int:
    return round((aval / HZ_10TH_DIV) - 0.5)

def yval_to_vt_nth(aval: float) -> int:
    return round((aval / VT_4TH_DIV) - 0.5)

#####
# ustility function
#####
def _merge_adjacent_hz_cols(freq_col_list: List[Tuple[int, int]]) \
    -> List[Tuple[int, int]]:
    """Merge the 3rd col with 1st or 2nd frequent column if their
    difference is only 1.

    If first and 2nd column diff is less than or equals to 2, it is
    ambiguous which one to merge to, so abort.

    Purpose?  Merge adjacent hz_cols if because the different hz_col are
    probably a result of indentations.  Having this addressed some
    borderline cases.

    Args:
        freq_col_list: a list of most frequent (freq, col)
    Returns:
        a list of (freq, col), which might be shorter if the first 2
        elements are adjacent.
    """
    # there is too few lines in this block, kept of it is
    if len(freq_col_list) < 3:
        return freq_col_list
    col_1_freq, col_1 = freq_col_list[0]
    col_2_freq, col_2 = freq_col_list[1]
    col_3_freq, col_3 = freq_col_list[2]

    # if first column and 2nd column are very closed together, don't
    # merge because it is ambiguous.
    if abs(col_1 - col_2) <= 2:
        return freq_col_list

    if abs(col_3 - col_1) == 1:
        first_col_freq = (col_1_freq + col_3_freq, col_1)
        return [first_col_freq, freq_col_list[1]] + freq_col_list[3:]
    elif abs(col_3 - col_2) == 1:
        second_col_freq = (col_2_freq + col_3_freq, col_2)
        return [freq_col_list[0], second_col_freq]  + freq_col_list[3:]
    return freq_col_list


def is_form_page_vt_row(vt_row_count_map: Dict[int, int],
                        vt_row_hzclen_count_map: Dict[int, Dict[int, int]],
                        num_lines: int) -> bool:
    """Rerturn true is the page is a form-page based on vt_row info.

    A page is now divided into 4 rows.  If 3/4 of the rows has > 39% short lines,
    then it is a form-template.

    Question, how to distinguish this from a table page?  In that case, don't want
    apply page-level ydiff to such a page anyway, so this is not an issue.

    If a page has some text and then table, this function should not be called.

    """
    num_row_gt_10_perc = 0
    for freq, nth_row, perc in sorted(((freq, nth_row, round(freq / num_lines * 100.0))
                                       for nth_row, freq in vt_row_count_map.items()),
                                      reverse=True):
        if freq > 6 and perc >= 10:
            num_row_gt_10_perc += 1
        num_lines += freq
    # must have at least 3/4 of the page populated with something
    if num_row_gt_10_perc < 3:
        return False

    num_row_with_short_lines = 0
    for nth_row in sorted(vt_row_hzclen_count_map.keys()):
        hzclen_count_map = vt_row_hzclen_count_map[nth_row]
        num_row_lines, num_short_lines = 0, 0
        for freq, hzclen, perc in sorted(((freq, hzclen, round(freq / num_lines * 100.0))
                                          for hzclen, freq in hzclen_count_map.items()),
                                         reverse=True):
            if hzclen <= 4:
                num_short_lines += freq
            num_row_lines += freq
        short_perc = num_short_lines / num_row_lines * 100

        if short_perc > 39.0:
            num_row_with_short_lines += 1

    if num_row_with_short_lines >= 3:
        return True
    return False


# pylint: disable=too-many-statements, too-many-arguments, too-many-locals
def calc_one_page_format(page_num: int,
                         lxid_strinfos_list: List[Tuple[int, List[StrInfo]]],
                         nl_text: str,
                         prev_page_num_col: int) \
                         -> Tuple[int, float, bool]:
    """Return the number of column in the page.

    Args:
        page_num: the page number
        lxid_strinfos_list: the list of lines in that page, in strinfos_list
        nl_text: doc_text, with line breaks
        prev_page_num_col: the number of column previous page has
    Returns:
        - the number of columns in a page.
          0 = a form page
          1, 2, 3 = one-, two- or 3-column page
        - the page-level y-diff, in float
        - return True if finding the page level y-diff failed

    The way we detect number of columns in a page is by using a hz_col mechanism.
    We first divide a page into 10 hz_cols (MAX_PAGE_X / 10) and compute the
    length of those lines in those hz_col units.  This is to simplify the problem
    using more rough estimates.

    vt_row is another mechanism to detect form-page.  A page is divided into 4
    horizontal rows.  If 3+ of them have a lot of short lines (> 39% of the lines
    in that row), then it is a form-page.  This will detect false-positive
    form-pages.  But it doesn't have a significant negative impact.  In those
    scenario, the PDFBox's paragraph will be used instead of page-level y-diff.

    If a form page, then y-diff will not be set, so we will use pdfbox's
    paragraphs instead.

    Below is the psudo algorithm:

      - find the most frequent hz_col, and the 2nd most frequent hz_col in
        this page.  hz_col = hz_start_nth

      - find the most common hz_col_len, and the 2nd most common
        hz_col_len

      - compute num_col_len_le_3_perc
      - compute num_col_len_ge_5_perc

      if num_col_len_ge_5 is less than 30% and \
         is_form_page_vt_row():
         return is_form_template = True, page_ydiff_mode = -1,
         is_ydiff_failed = True

      if num_line > 50 and num_col_len_le_3_perc >= 50;
         is_form_template = True

      if num_lines >= 200:
        # When there are that many lines, must be 3 column.
        # Ignore all other logic in this branch, at the end,
        # it is a 3-column page.
        num_col_in_page = 3

      if top_hz_col_len == 3 or 4:
        if both top column or 2nd top column has > 37% and
           they starts in col_nth[1, 2] and col_nth[4, 5, 6]:
           return num_col_in_page = 2
        if both top column or 2nd top column has > 35% and
           they starts in col_nth[1, 2] and col_nth[4, 5, 6] and
           top_col_perc + top_2nd_col_perc > 80:
           return num_col_in_page = 2
        if num_cl_len_le_3_perc > 80% and top_col in col_nth[1, 2]:
          if prev_page_num_col == 3:
            num_col_in_page = 3
          else:
            num_col_in_page = 2

      if num_col_in_page is not set yet, num_col_in_page = 1

      #
      # now trying to figure out is page y-dff is valid
      #
      - first figure out valid_line_nth_len, which is
        defaulted to 5, but can change.
      - if the number of full line is < 3 (MIN_FULL_YDIFF):
        return the top page ydiff, but is_ydiff_failed = True

      #
      # if there is enough full-length lines, page-ydiff is
      # valid.
      #
      is_ydiff_failed = False
      if there are full-lines in this page:
        if the freq of this y-diff < 3 (MIN_FULL_YDIFF):
          page-ydiff-mode = top y-diff (not full y-diff)
          is_ydiff_failed = True
        elif the ydiff is too big (>= 28 points):
          page-ydiff-mode = top full y-diff
          is_ydiff_failed = True
        else:
          page-ydiff-mode = top full y-diff
      else:
        page-ydiff = top y-diff (not full y-diff)
        is_ydiff_failed = True

      return page_num_col, page-ydiff-mode, is_ydiff_failed

    """

    # for detecting the last line is reached
    unused_last_linenum_in_page = lxid_strinfos_list[-1][0]
    num_lines = len(lxid_strinfos_list)

    if IS_DEBUG:
        print('\n=calc_one_page_format  @page {}, num_lines = {}, '
              'prev_page_num_col = {}'.format(page_num,
                                              num_lines,
                                              prev_page_num_col))
        print()

    # count how many lines starts in a particular vt_row (vertical row)
    vt_row_count_map = defaultdict(int)  # type: Dict[int, int]
    # count how many lines in vt_row are of hz_col_len
    vt_row_hzclen_count_map = \
        defaultdict(lambda: defaultdict(int))  # type: Dict[int, Dict[int, int]]

    # count how many lines are of hz_col length
    hz_len_count_map = defaultdict(int)  # type: Dict[int, int]
    # count how many lines start from a particular hz_col, (horizontal column)
    hz_col_count_map = defaultdict(int)  # type: Dict[int, int]
    # a list of (line-col-start, line-col-len)
    line_hz_startnth_nthlen_list = []  # type: List[Tuple[int, int]]
    # store above information for later use in this function
    for unused_line_num, lxid_strinfos in lxid_strinfos_list:
        lx_min_x, lx_max_x = pdfoffsets.get_lx_min_max_x(lxid_strinfos)
        lx_min_y, unused_lx_max_y = pdfoffsets.get_lx_min_max_y(lxid_strinfos)

        hz_start_nth = xval_to_hz_nth(lx_min_x)
        hz_end_nth = xval_to_hz_nth(lx_max_x)
        hz_nth_len = round((lx_max_x - lx_min_x) / HZ_10TH_DIV)

        line_hz_startnth_nthlen_list.append((hz_start_nth,
                                             hz_nth_len))

        hz_len_count_map[hz_nth_len] += 1
        hz_col_count_map[hz_start_nth] += 1

        vt_start_nth = yval_to_vt_nth(lx_min_y)
        vt_row_count_map[vt_start_nth] += 1
        vt_row_hzclen_count_map[vt_start_nth][hz_nth_len] += 1

    # create a list to find the most frequent hz_col
    sorted_freq_col_list = \
        sorted(((freq, nth_col)
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

    # create a list to find the most frequent line length in hz_col unit
    sorted_freq_hzlen_list = sorted(((freq, nth_col) for nth_col, freq
                                     in hz_len_count_map.items()),
                                    reverse=True)
    if IS_DEBUG:
        for freq, nth_col in sorted(((freq, nth_col) for nth_col, freq
                                     in hz_len_count_map.items()),
                                    reverse=True):
            perc = round(freq / num_lines * 100.0)
            print("    >>>> hz_len= {}, freq={}, perc={}%".format(nth_col, freq, perc))
        print()

    # merge adjacent hz_cols, to handle some borderline cases
    merge_adj_col_count_list = _merge_adjacent_hz_cols(sorted_freq_col_list)
    if IS_DEBUG:
        for freq, nth_col in merge_adj_col_count_list:
            perc = round(freq / num_lines * 100.0)
            print("    >>>> merged hz_col= {}, freq={}, perc={}%".format(nth_col,
                                                                         freq, perc))
        print()

    # find the most frequent hz_col that a line starts in this page
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

    # find the most frequent hz_len for the lines in this page
    sorted_col_len_list = sorted_freq_hzlen_list
    top_col_len = 0
    if sorted_col_len_list:
        _, top_col_len = sorted_col_len_list[0]
    top_2nd_col_len = -1
    if len(sorted_col_len_list) > 1:
        _, top_2nd_col_len = sorted_col_len_list[1]

    if IS_DEBUG:
        print('    top_col = {}, top_col_perc = {}'.format(top_col, top_col_perc))
        print('    top_2nd_col = {}, top_2nd_col_perc = {}'.format(top_2nd_col,
                                                                   top_2nd_col_perc))
        print()
        print('    top_col_len_ = {}'.format(top_col_len))
        print('    top_2nd_col_len = {}'.format(top_2nd_col_len))

    # num of lines with hz_len <= 3
    num_col_len_le_3 = hz_len_count_map.get(0, 0) + \
                       hz_len_count_map.get(1, 0) + \
                       hz_len_count_map.get(2, 0) + \
                       hz_len_count_map.get(3, 0)
    num_col_len_le_3_perc = num_col_len_le_3 / num_lines * 100.0

    # num of lines with hz_len >= 5;  these are lines for normal
    # sentences
    # hz_len_count_map.get(5, 0) + \
    num_col_len_ge_5 = hz_len_count_map.get(6, 0) + \
                       hz_len_count_map.get(7, 0) + \
                       hz_len_count_map.get(8, 0) + \
                       hz_len_count_map.get(9, 0) + \
                       hz_len_count_map.get(10, 0)
    num_col_len_ge_5_perc = num_col_len_ge_5 / num_lines * 100.0

    if IS_DEBUG:
        print('\n    num_col_len_le_3 = {}, perc = {}%'.format(num_col_len_le_3,
                                                               num_col_len_le_3_perc))
        print('\n    num_col_len_ge_5 = {}, perc = {}%'.format(num_col_len_ge_5,
                                                               num_col_len_ge_5_perc))

    # two column text
    if num_lines >= 60 and \
       num_col_len_ge_5_perc < 30.0 and \
       abs(top_col - top_2nd_col) > 2 and \
       top_col_perc + top_2nd_col_perc > 80.0:
        out_num_col = 2
        out_page_ydiff_mode = -1.0
        out_is_failed = True  # we will use pdfbox's ydiff instead
        # print('2 columnt text')
        return out_num_col, out_page_ydiff_mode, out_is_failed

    if num_col_len_ge_5_perc < 30.0:
        # print('mostly short lines< col_len_ge_5 < 30%')
        if is_form_page_vt_row(vt_row_count_map,
                               vt_row_hzclen_count_map,
                               num_lines):
            # print('is_a form..............')
            # out_num_col = 0 means it is a form-page
            out_num_col = 0
            out_page_ydiff_mode = -1.0
            out_is_failed = True
            return out_num_col, out_page_ydiff_mode, out_is_failed

    # With the above information, predict how many column a page has.
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

    #
    # now trying to figure out is page y-dff is valid
    #
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
        print('    top_col_len = {}, valid_line_nth_len = {}'.format(top_col_len,
                                                                     valid_line_nth_len))
        print()

    #
    # if there is enough full-length lines, page-ydiff is
    # valid.
    #
    full_hz_col_count_map = defaultdict(int)  # type: Dict[int, int]
    full_ydiff_count_map = defaultdict(int)  # type: Dict[float, int]
    ydiff_count_map = defaultdict(int)  # type: Dict[float, int]
    prev_y = 1000
    prev_hz_nth_len = 0
    nth_len_ge_5 = 0
    for (unused_line_num, lxid_strinfos), (hz_start_nth, hz_nth_len) \
        in zip(lxid_strinfos_list, line_hz_startnth_nthlen_list):
        lx_min_x, lx_max_x = pdfoffsets.get_lx_min_max_x(lxid_strinfos)
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

    out_num_col = 0
    out_page_ydiff_mode = -1.0
    out_is_failed = False

    if sorted_count_full_ydiff_list:
        # this 5 is not related to valid_line_nth_len
        if sorted_count_full_ydiff_list[0][0] < MIN_FULL_YDIFF:
            # if sorted_count_full_ydiff_list[0][0] < MIN_FULL_YDIFF and \
            #    not (len(sorted_count_full_ydiff_list) >= 2 and \
            #    abs(sorted_count_full_ydiff_list[0][1] - sorted_count_full_ydiff_list[1][1]) < 1):
            # too infrequent
            # 2nd check if to NOT fail those with enough info, doc 9326, page 55
            out_page_ydiff_mode = sorted_count_full_ydiff_list[0][1]
            out_is_failed = True
            if IS_DEBUG:
                print('failed page {}, branch 1'.format(page_num))
        # elif sorted_count_full_ydiff_list[0][1] >= 22:
        elif sorted_count_full_ydiff_list[0][1] >= 28:  # have seen 27.5, doc 9325, page 64
            # too big
            out_page_ydiff_mode = sorted_count_full_ydiff_list[0][1]
            # failed_page_ydiff_mode_pages.append(page_num)
            out_is_failed = True
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
            out_page_ydiff_mode = sorted_count_full_ydiff_list[0][1]
    else:
        if sorted_count_ydiff_list:
            # for title page, or last page in a doc?
            out_page_ydiff_mode = sorted_count_ydiff_list[0][1]
        else:
            out_page_ydiff_mode = -1.0
        out_is_failed = True
        if IS_DEBUG:
            print('failed page {}, branch 3'.format(page_num))

    if IS_DEBUG:
        print('  Done.  calc_one_page_format()')
    return num_col_in_page, out_page_ydiff_mode, out_is_failed


def pick_page_adjacent_ydiff_mode(ydiff_mode_list: List[float],
                                  global_ydiff: float,
                                  is_after_page: bool = False) -> float:
    """This specifically handle the case where the 3 ydiff in
       previous 3 page before or after doesn't agree.

       ydiff_mode_list is always of length 3 or less.

       Previous approach of taking the most frequent failed when
       all 3 page ydiff are distinct.  Now this is deterministic.
    """
    # print('pick_page_adjacent_ydiff_mode: {}'.format(ydiff_mode_list))
    if not ydiff_mode_list:
        return global_ydiff

    # print("ydiff_mode_list: {}".format(ydiff_mode_list))
    count_dict = defaultdict(int)  # type: Dict[float, int]
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
def calc_page_ydiff_modes_num_cols(page_linenum_list_map: Dict[int, List[int]],
                                   lxid_strinfos_map: Dict[int, List[StrInfo]],
                                   nl_text: str,
                                   all_ydiffs: List[float]) \
                                   -> Tuple[Dict[int, float],
                                            Dict[int, int]]:
    """Compute the page-level y_diff for all pages in a document.

    ARGS:
        page_linenum_list_map: a map of page_num and all the line numbers
                               in that page
        lxid_strinfos_map: a map of line number and its str's
        nl_text: the document text, with line breaks
        all_ydifs: all the y_diffs in the documents
    RETURNS:
        Tuple of following:
          - a map with the y_diff_mode for each page
          - a map with num_col for each page

        NOTE:
          if y_diff_mode for a page is -1, then it is a form page in which
          y_diff_mode doesn't apply.

    Psuedo code:
      - go through every page and call calc_one_page_format()
        - compute page_num_col_map
        - compute page_ydiff_mode_map
        - collect failed_page_ydiff_mode_pages, from is_ydiff_failed in calc_one_page_format()

      - initialize page_ydiff_mode_list, using page_ydiff_mode_map
      - initialize doc_ydiff_mode, by taking mode of all ydiff in the doc

      for all failed_page_yidff_mode_list
        - compute a replacement page-ydiff from the page-ydiff of 3 adjacent pages
        - normally, those page should be before the failed page
        - if the failed page is page 1 or page 2, then take the 3 pages
          after to adjust the page-ydiff of the failed page.

      go through the page list
        - if a page is form-template (page_num_col_map[page_num] == 0)
             page_ydiff_mode_map[page_num] = -1
          else if the page-ydiff is not computable for other reason
            # use the document y-diff instead
            page_ydiff_mode_map[page_num] = doc_ydiff_mode

      return page_ydiff_mode_map, page_num_col_map
    """

    if IS_TOP_LEVEL_DEBUG:
        print('\n========= calc_page_ydiff_modes()')

    page_num_list = sorted(page_linenum_list_map.keys())
    page_ydiff_mode_map = {}  # type: Dict[int, float]
    failed_page_ydiff_mode_pages = []  # type: List[int]

    page_num_col_map = {}  # type: Dict[int, int]
    prev_page_num_col = -1
    for page_num in page_num_list:
        page_linenum_list = sorted(set(page_linenum_list_map[page_num]))
        # Previously, there might be duplicated entries in page_linenum_list_map,
        # This is cleaned up version.
        page_linenum_list_map[page_num] = page_linenum_list

        lxid_strinfos_list = [(line_num, lxid_strinfos_map[line_num])
                              # pylint: disable=line-too-long
                              for line_num in page_linenum_list]  # type: List[Tuple[int, List[StrInfo]]]
        # page_ydif_mode_map is modified inline
        page_num_col, page_ydiff_mode, is_failed = \
            calc_one_page_format(page_num,
                                 lxid_strinfos_list,
                                 nl_text,
                                 prev_page_num_col)

        page_ydiff_mode_map[page_num] = page_ydiff_mode
        if is_failed:
            failed_page_ydiff_mode_pages.append(page_num)
        page_num_col_map[page_num] = page_num_col
        prev_page_num_col = page_num_col

    if IS_DEBUG:
        print('page_ydiff_mode_map:')
    page_num_set = set(page_num_list)
    page_ydiff_mode_list = []  # type: List[float]
    if page_num_list:
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

    if IS_PAGE_NUM_COL_DEBUG:
        for page_num in page_num_list:
            print('page_num_col_map[{}] = {}'.format(page_num,
                                                     page_num_col_map[page_num]))
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

    if all_ydiffs:
        doc_ydiff_mode = mathutils.get_mode_in_list(all_ydiffs)
    else:
        doc_ydiff_mode = -1

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

    # set all problematic pages with the newly adjusted ydff based on
    # adjacent pages.
    for page_num in failed_page_ydiff_mode_pages:
        page_ydiff_mode_map[page_num] = page_ydiff_mode_list[page_num]

    # final adjustment based on
    #   - if the page is a form-page, then y_diff_mode for that page is -1
    #   - if page_y_diff_mode is invalid, use document level y-diff
    #   - otherwise, keep whatever we computed
    for page_num in page_num_list:
        if page_num_col_map[page_num] == 0:
            page_ydiff_mode_map[page_num] = -1.0
        # reset if there are any issue, 0 or -1
        elif page_ydiff_mode_map[page_num] <= 0:
            page_ydiff_mode_map[page_num] = doc_ydiff_mode
            # print('page_ydiff_mode_map[{}] after = {}'.format(page_num,
            #                                                   doc_ydiff_mode))

    if IS_TOP_LEVEL_DEBUG:
        print('\nadjusted page_ydiff_mode_mao:')
        for page_num in page_num_list:
            if page_num in failed_page_ydiff_mode_pages:
                print('     page {}: {}, failed'.format(page_num, page_ydiff_mode_map[page_num]))
            else:
                print('     page {}: {}'.format(page_num, page_ydiff_mode_map[page_num]))

        print('page_column_list:')
        for page_num in page_num_list:
            num_col = page_num_col_map[page_num]
            print('page {}: num_col = {}'.format(page_num, num_col))

    return page_ydiff_mode_map, page_num_col_map
