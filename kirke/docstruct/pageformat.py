from collections import defaultdict, namedtuple
import re
import statistics
# pylint: disable=unused-import
from typing import Dict, List, Tuple

from kirke.docstruct.pdfoffsets import PageInfo3, PageFormatStatus


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

    page_text = doc_text[apage.start:apage.end].strip()
    # print('\npage #{}, len(page_text) = {}'.format(apage.page_num, len(page_text)))
    num_new_lines = page_text.count('\n')
    # print('  num_new_lines = {}'.format(num_new_lines))
    num_periods = page_text.count('.\n') + page_text.count('.')
    # print('  num_periods = {}'.format(num_periods))

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
