from collections import defaultdict, namedtuple
import re
import statistics
# pylint: disable=unused-import
from typing import Dict, List, Tuple

from kirke.docstruct.pdfoffsets import PageInfo3, PageFormatStatus

IS_DEBUG = True

# Page Size in Point at 72 dpi
# letter size: width 8.5 in, height 11 in
HZ_10TH_DIV = 61.2   # 612.0 / 10, weidth
MAX_Y = 792.0
ONE_THIRD_MAX_Y = 792.0 * 2 / 3


class PageFormatClassifier:

    def __init__(self) -> None:
        pass

    def classify(self,
                 apage: PageInfo3,
                 text: str) \
                 -> PageFormatStatus:
        pformat = calc_page_stats(apage, text)

        return pformat


def calc_page_stats(apage: PageInfo3, doc_text: str) -> PageFormatStatus:

    page_start, page_end = apage.start, apage.end
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
    prev_yend = 0
    len_col_ge_6_before_1third_page = 0
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


    is_one_para_per_page = False
    if len(page_text) > 1500 and \
       num_new_lines == 0:
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
