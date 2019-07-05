
import logging
import re
from typing import List, Dict

from kirke.utils.unicodeutils import normalize_dbcs_sbcs
from kirke.nlputil.text2int_jp import NUMERIC_WORDS, NUM_DIGIT_REGEX_ST
from kirke.nlputil import text2int_jp

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

IS_DEBUG = False

DATE_NUMERIC_WORDS = NUMERIC_WORDS + ['元']
DATE_NUM = r'(({})|({})+)'.format(NUM_DIGIT_REGEX_ST,
                                  '|'.join(DATE_NUMERIC_WORDS))

# http://www.jlit.net/reference/history/era-names.html
ERA_NAMES = ['大正', '昭和', '平成', '令和']
ERA_YEARS_MAP = {'大正': (1912, 1926), # 1912.7  - 1926.12
                 '昭和': (1926, 1989), # 1926,12 - 1989.1
                 '平成': (1989, 2019), # 1989.1  - 2019.4,
                 '令和': (2019, 2200)} # 2019.5  -   -
ERA_NAMES_REGEX_ST = '|'.join(ERA_NAMES)
JP_DATE_REGEX_ST = r'({})?{}年{}月{}日'.format(ERA_NAMES_REGEX_ST, DATE_NUM, DATE_NUM, DATE_NUM)
JP_DATE_REGEX = re.compile(JP_DATE_REGEX_ST)

ARABIC_DATE_REGEX_ST = r'([0-9]{2,4})[\-/／]([0-9]{1,2})[\-/／]([0-9]{1,2})'
ARABIC_DATE_REGEX = re.compile(ARABIC_DATE_REGEX_ST)

def date_num_to_int(line: str) -> int:
    if line == '元':
        return 1
    adict = text2int_jp.extract_number(line)
    if adict:
        return adict['norm']['value']
    return -1

def era_to_gregorian_year(era_name: str, year: int) -> int:
    start_year, unused_end_year = ERA_YEARS_MAP[era_name]
    return start_year + year - 1


# pylint: disable=too-many-locals
def extract_dates(line: str, is_norm_dbcs_sbcs=False) -> List[Dict]:
    if is_norm_dbcs_sbcs:
        sbcs_line = normalize_dbcs_sbcs(line)
    else:
        sbcs_line = line
    result = []  # type: List[Dict]
    mat_list = list(JP_DATE_REGEX.finditer(sbcs_line))
    for mat in mat_list:
        numeric_span = (mat.start(), mat.end(), mat.group())
        if IS_DEBUG:
            print('numeric_span: {}'.format(numeric_span))

            for gidx, unused_group in enumerate(mat.groups(), 1):
                print('group {} ({}, {}) [{}]'.format(gidx,
                                                      mat.start(gidx),
                                                      mat.end(gidx),
                                                      mat.group(gidx)))
        era = mat.group(1)
        year = mat.group(2)
        month = mat.group(5)
        day = mat.group(8)

        print('era = {}'.format(era))
        print('year = {}'.format(date_num_to_int(year)))
        print('month = {}'.format(date_num_to_int(month)))
        print('day = {}'.format(date_num_to_int(day)))

        if era:
            gregorian_year = era_to_gregorian_year(era, date_num_to_int(year))
        else:
            gregorian_year = date_num_to_int(year)

        month_val = date_num_to_int(month)
        day_val = date_num_to_int(day)


        print('gregorian year: {}'.format(gregorian_year))

        adict = {'norm': {'date': '{:04d}-{:02d}-{:02d}'.format(gregorian_year,
                                                                month_val,
                                                                day_val)},
                 'start': mat.start(),
                 'end': mat.end(),
                 'concept': 'date',
                 'text': line[mat.start():mat.end()]}
        result.append(adict)

    mat_list = list(ARABIC_DATE_REGEX.finditer(sbcs_line))
    for mat in mat_list:
        numeric_span = (mat.start(), mat.end(), mat.group())
        if IS_DEBUG:
            print('xx numeric_span: {}'.format(numeric_span))

            for gidx, unused_group in enumerate(mat.groups(), 1):
                print('xx group {} ({}, {}) [{}]'.format(gidx,
                                                         mat.start(gidx),
                                                         mat.end(gidx),
                                                         mat.group(gidx)))
        year_val = int(mat.group(1))
        month_val = int(mat.group(2))
        day_val = int(mat.group(3))

        print('year_val: {}'.format(year_val))
        print('month_val: {}'.format(month_val))
        print('day_val: {}'.format(day_val))

        adict = {'norm': {'date': r'{:04d}-{:02d}-{:02d}'.format(year_val,
                                                                 month_val,
                                                                 day_val)},
                 'start': mat.start(),
                 'end': mat.end(),
                 'concept': 'date',
                 'text': line[mat.start():mat.end()]}
        result.append(adict)

    return result
