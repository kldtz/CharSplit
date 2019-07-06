import logging
import re
from typing import Dict, List, Match, Tuple

from kirke.utils import text2int
from kirke.utils.text2int import remove_num_words_join_hyphen


# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IS_DEBUG = False

# pylint: disable=line-too-long
# (?:^| |\()
# want to avoid "rst", which is "rs" + "t" where "t" is just trillion
CURRENCY_PAT_ST = r'((((\bUSD|\bEUR|\bGBP|\bCNY|\bJPY|\bINR|\bRupees?|\bRs\b\.?)|[\$€£円¥₹]) *({})|({}) *((USD|euros?|EUR|GBP|CNY|JPY|INR|Rs|[dD]ollars?|u\.\s*s\.\s*dollars?|ドル|米ドル|㌦|アメリカドル|弗|[eE]uros?|ユーロ|[pP]ounds?|ポンド|[yY]uans?|人民元|元|[yY]ens?|[rR]upees?|インド・ルピー|インドルピー|ルピー)|[\$€£円¥₹])))'.format(text2int.numeric_regex_st, text2int.numeric_regex_st_with_b)
CURRENCY_PAT = re.compile(CURRENCY_PAT_ST, re.I)

CURRENCY_SYMBOL_EXACT_PAT = re.compile(r'([\$€£円¥₹]|\bR[Ss]\.)')

FLOAT_WITH_WORD_REGEX = re.compile(r'\b(\d+\.\d+)\s+(\S+)')


# print('\nCURRENCY_PAT_ST')
# print(CURRENCY_PAT_ST)

# pylint: disable=line-too-long
# NUMBER_PAT = re.compile(r'(^|\s)\(?(-?({}))\)?[,\.:;]?(\s|$)'.format(text2int.numeric_regex_st), re.I)
# NUMBER_PAT = re.compile(r'((^|\s)\(?(-?([0-9]+([,\.][0-9]{3})*[,\.]?[0-9]*|[,\.][0-9]+))\)?[,\.:;]?(\s|$))' +
#                         r'|({})'.format(text2int.numeric_words_regex_st))

# standard floating point number
# https://www.regular-expressions.info/floatingpoint.html
# NUM_PAT_ST = '[-+]?[0-9]*\.?[0-9]+'
# (?:^| |\()
# NUM_PAT_ST = r'(([-+]?\b[0-9,\.]*[0-9]+)|' + \
#              r'\b({}))\b'.format(text2int.numeric_words_regex_st)

NUM_PAT_ST = r'(((?<=\s)|(?<=^)|(?<=\()|(?<=\[)|(?<=\<))({}))\b'.format(text2int.numeric_regex_st)

NUMBER_PAT = re.compile(NUM_PAT_ST, re.I)

# print('\nNUM_PAT_ST')
# print(NUM_PAT_ST)

# pylint: disable=line-too-long
PERCENT_PAT_ST = r'({})\s*(%|percent)'.format(text2int.numeric_regex_st_with_b)
PERCENT_PAT = re.compile(PERCENT_PAT_ST, re.I)

# print('\nPERCENT_PAT_ST:')
# print(PERCENT_PAT_ST)

# pylint: disable=too-many-return-statements
def normalize_currency_unit(line: str) -> str:
    lc_line = line.lower()
    if lc_line in set(['$', 'usd', 'dollar', 'dollars',
                       'ドル', '米ドル', '㌦', 'アメリカドル', '弗', 'セント']) or \
       re.search(r'u\.\s*s\.\s*dollars?', lc_line):
        return 'USD'
    elif lc_line in set(['€', 'eur', 'euro', 'euros', 'ユーロ']):
        return 'EUR'
    elif lc_line in set(['£', 'gbp', 'pound', 'pounds', 'ポンド']):
        return 'GBP'
    elif lc_line in set(['元', 'cny', 'yuan', 'yuans', '人民元']):
        # '¥' should be here also, but conflict with JPY.
        # since we support Mitsubishi now, JPY wins
        return 'CNY'
    elif lc_line in set(['¥', '円', 'jpy', 'yen', 'yens']):
        # we don't handle double-byte character here because should have
        # been normalized by text2int_jp.dbcs_to_sbcs()
        return 'JPY'
    elif lc_line in set(['₹', 'inr', 'rupee', 'rupees', 'rs', 'rs.',
                         'インド・ルピー', 'インドルピー', 'ルピー']):
        return 'INR'

    return 'UNKNOWN_CURRENCY'


def currency_to_norm_dict(cx_mat: Match, line: str) -> Dict:
    if IS_DEBUG:
        print('  currency cx_mat group: {} {} [{}]'.format(cx_mat.start(), cx_mat.end(), cx_mat.group()))
        for gi, unused_group in enumerate(cx_mat.groups(), 1):
            print("    cx_mat.group #{}: [{}]".format(gi, cx_mat.group(gi)))
    norm_unit = 'USD'
    norm_value = -1
    if cx_mat.group(3):
        norm_unit = normalize_currency_unit(cx_mat.group(3))
        norm_value = text2int.extract_number(cx_mat.group(5))['norm']['value']
    elif cx_mat.group(11):
        norm_unit = normalize_currency_unit(cx_mat.group(19))
        norm_value = text2int.extract_number(cx_mat.group(11))['norm']['value']
    norm_dict = {'norm': {'unit': norm_unit,
                          'value': norm_value},
                 'text': line[cx_mat.start():cx_mat.end()],
                 'start': cx_mat.start(),
                 'concept': 'currency',
                 'end': cx_mat.end()}
    return norm_dict


def extract_currencies(line: str) -> List[Dict]:
    norm_line = remove_num_words_join_hyphen(line)
    result = []
    mat_list = CURRENCY_PAT.finditer(norm_line)
    for mat in mat_list:
        norm_dict = currency_to_norm_dict(mat, line)
        result.append(norm_dict)
    return result


def percent_to_norm_dict(cx_mat: Match, line: str) -> Dict:
    # print('  percent cx_mat group: {} {} [{}]'.format(cx_mat.start(), cx_mat.end(), cx_mat.group()))
    # for gi, group in enumerate(cx_mat.groups(), 1):
    #     print("    perc cx_mat.group #{}: [{}]".format(gi, cx_mat.group(gi)))
    norm_value = -1
    if cx_mat.group(1):
        norm_value = text2int.extract_number(cx_mat.group(1))['norm']['value']
    norm_dict = {'norm': {'unit': '%',
                          'value': norm_value},
                 'text': line[cx_mat.start():cx_mat.end()],
                 'start': cx_mat.start(),
                 'concept': 'percent',
                 'end': cx_mat.end()}
    return norm_dict


def extract_percents(line: str) -> List[Dict]:
    norm_line = remove_num_words_join_hyphen(line)
    result = []
    mat_list = PERCENT_PAT.finditer(norm_line)
    for mat in mat_list:
        norm_dict = percent_to_norm_dict(mat, line)
        result.append(norm_dict)
    return result


"""
def number_to_norm_dict(cx_mat: Match, line: str, offset: int = -1) -> Dict:
    # print('  number cx_mat group: {} {} [{}]'.format(cx_mat.start(), cx_mat.end(), cx_mat.group()))
    # for gi, group in enumerate(cx_mat.groups(), 1):
    #     print("    numb cx_mat.group #{}: [{}]".format(gi, cx_mat.group(gi)))
    norm_value = -1
    if cx_mat.group():
        norm_value = text2int.extract_number(cx_mat.group())['norm']['value']

    adjusted_offset = 0
    if offset != -1:
        adjusted_offset = offset
    norm_dict = {'norm': {'value': norm_value},
                 'text': line[adjusted_offset + cx_mat.start():adjusted_offset + cx_mat.end()],
                 'start': adjusted_offset + cx_mat.start(),
                 'concept': 'number',
                 'end': adjusted_offset + cx_mat.end()}
    return norm_dict
"""


def number_to_norm_dict(num_st: str,
                        start: int,
                        end: int,
                        line: str) -> Dict:
    norm_value = -1
    if num_st:
        norm_dict = text2int.extract_number(num_st).get('norm')
        if norm_dict:
            norm_value = norm_dict['value']

    norm_dict = {'norm': {'value': norm_value},
                 'text': line[start:end],
                 'start': start,
                 'end': end}
    return norm_dict



D_D_SPLIT_REGEX = re.compile(r'(?<=\d)\s+(?=\d)')

def num_num_split(line: str, offset: int = 0) -> List[Tuple[int, int, str]]:
    line = line.strip()
    chunks = list(D_D_SPLIT_REGEX.finditer(line))
    if chunks:
        result = []  # type: List[Tuple[int, int, str]]
        prev = 0
        for chunk in chunks:
            result.append((offset + prev, offset + chunk.start(), line[prev:chunk.start()]))
            prev = chunk.end()
        result.append((offset + prev, offset + len(line), line[prev:]))
        return result
    return [(offset, offset + len(line), line)]

# following numbers are not valid
# 'm'  'b', 't'
INVALID_NUM_REGEX = re.compile(r'(\s*\b[mbt]\-\S+\s*|^and\s*|(?<=\d)\s+and\s+(?=\d)|,(?=\d{4})|(?<=\d)\s*\-\s*(?=\d))')

MATCH_ALL_REGEX = re.compile(r'^.*$')

D_DASH_D_REGEX = re.compile(r'\d+\-\d+')

D_DASH_D_WORD_REGEX = re.compile(r'^\d+(\-\d+)+$')

# is line always just one word, or can be multiple words?
# it seems to be possible to be multiple words
def is_invalid_number_word(word: str) -> bool:
    # a date, 02-03
    if D_DASH_D_REGEX.search(word):
        return True

    norm_st = text2int.normalize_comma_period(word)
    # sechead, 2.3.3
    if norm_st.count('.') > 1:
        return True
    return False

NUM_MBT_REGEX = re.compile(r'^\d+[mbt]$', re.I)

NUM_SP_MBT_REGEX = re.compile(r'^([\d\.]+)\s+[mbt]$', re.I)

def is_invalid_number_phrase(line: str) -> bool:
    if line.lower() in set(['b', 'm', 't']):
        return True

    if NUM_MBT_REGEX.search(line):
        return True

    # a date, 02-03, or phone number
    if D_DASH_D_WORD_REGEX.search(line):
        return True

    return False


"""
def invalid_num_split(mat: Match) -> List[Tuple[Match, int]]:
    line = mat.group()
    if is_invalid_number_phrase(line):
        return []

    # should not start with 'and' and all 'mbt-*' are not valid numbers
    # so they are removed
    split_chunks = list(INVALID_NUM_REGEX.finditer(line))
    if split_chunks:
        adjusted_offset = mat.start()
        result = []  # type: List[Tuple[Match, int]]
        prev = 0
        for split_chunk in split_chunks:
            chunk_st = line[prev:split_chunk.start()]
            tmp_mat = NUMBER_PAT.search(chunk_st)
            if tmp_mat and chunk_st and not is_invalid_number_word(chunk_st):
                result.append((tmp_mat, adjusted_offset + prev))
            prev = split_chunk.end()
        tmp_mat = NUMBER_PAT.search(line[prev:])
        if tmp_mat and line[prev:] and not is_invalid_number_word(line[prev:]):
            result.append((tmp_mat, adjusted_offset + prev))
        return result

    return [(mat, -1)]
"""

def invalid_num_split(mat: Match) -> List[Tuple[int, int, str]]:
    """Find an adjusted match for valid number, and offsets from mat.group() in arg.

    Args:
      mat: the Match of the number, might have invalid prefix such as 'and'. Want
           to adjust the offset accordingly.

    Return:
      a list of
        mat: the mat for the new number
        mat_start: the index of the adjustment from begin of the mat, index 0

      This list can be empty if the number is not valid
    """
    mat_start = mat.start()
    mat_line = mat.group()
    # cannot be \d [mbt] or just [mbt]
    if is_invalid_number_phrase(mat_line):
        return []

    # only take the number portion of '23.3 M'
    num_mbt_mat = NUM_SP_MBT_REGEX.search(mat_line)
    if num_mbt_mat:
        return [(mat_start + num_mbt_mat.start(1),
                 mat_start + num_mbt_mat.end(1),
                 num_mbt_mat.group(1))]

    # should not start with 'and' and all 'mbt-*' are not valid numbers
    # so they are removed

    # take all invalid mat and remove them from the
    # potential list of numbers
    split_chunks = list(INVALID_NUM_REGEX.finditer(mat_line))
    if split_chunks:
        result = []  # type: List[Tuple[int, int, str]]
        prev = 0
        for split_chunk in split_chunks:
            chunk_st = mat_line[prev:split_chunk.start()]
            tmp_mat = NUMBER_PAT.search(chunk_st)
            if tmp_mat and chunk_st and not is_invalid_number_word(chunk_st):
                result.append((mat_start + prev + tmp_mat.start(),
                               mat_start + prev + tmp_mat.end(),
                               tmp_mat.group()))
            prev = split_chunk.end()
        tmp_mat = NUMBER_PAT.search(mat_line[prev:])
        if tmp_mat and mat_line[prev:] and not is_invalid_number_word(mat_line[prev:]):
            result.append((mat_start + prev + tmp_mat.start(),
                           mat_start + prev + tmp_mat.end(),
                           tmp_mat.group()))

        return result

    # skip '1-12'
    # TODO, might still accept '-3 M'
    if mat_line.count('-') > 1 or \
        (mat_line.count('-') == 1 and not mat_line.startswith('-')):
        return []

    # split '10.03 Fifty'
    first_float_mat = FLOAT_WITH_WORD_REGEX.search(mat_line)
    if first_float_mat:
        word_after = first_float_mat.group(2).lower()
        if word_after not in ['million', 'billion', 'trillion']:
            return [(mat_start, mat_start + first_float_mat.end(1), mat_line[:first_float_mat.end(1)]),
                    (mat_start + first_float_mat.start(2), mat.end(), mat_line[first_float_mat.start(2):])]

    return [(mat.start(), mat.end(), mat.group())]


def space_same_len(matchobj: Match) -> str:
    return ' ' * len(matchobj.group())


"""
def extract_numbers_0705(line: str, is_ignore_currency_symbol: bool = False) -> List[Dict]:
    norm_line = remove_num_words_join_hyphen(line)
    if is_ignore_currency_symbol:
        norm_line = re.sub(CURRENCY_SYMBOL_EXACT_PAT, space_same_len, norm_line)

    result = []
    mat_list = NUMBER_PAT.finditer(norm_line)
    # some mat in list might have mutliple intergers, such as '2 3'
    # 'better_mat_list' will store the real numeric mat
    mat_offset_list = []  # type: List[Tuple[Match, int]]
    for mat in mat_list:
        # Due ot our preference to parse English expressions mixed
        # with numbers, '2 4' might be captured in the match part.
        # We separate those numbers here.

        num_num_span_list = num_num_split(mat.group(), mat.start())
        if len(num_num_span_list) > 1:
            for offset_start, unused_offset_end, span_st in num_num_span_list:
                # string is 'm-3'
                # re.search(r'^[mtb]\-', span_st):
                if re.search(r'^[mtb]\-', span_st):
                    continue

                mat2 = NUMBER_PAT.search(span_st)
                if mat2:
                    mat_offset_list.append((mat2, offset_start))
        else:
            # get rid of 'm-2'
            # remove_invalid_num_spans = invalid_num_split(mat)
            mat_offset_list.extend(invalid_num_split(mat))

    for mat, mat_start in mat_offset_list:
        # print('mat = {}, mat_start = {}'.format(mat, mat_start))
        # norm_st = text2int.normalize_comma_period(mat.group())
        # 2.3.4, or section head
        # 2018-01-01 or date
        if is_invalid_number_phrase(mat.group()) or \
           is_invalid_number_word(mat.group()):
            continue

        norm_dict = number_to_norm_dict(mat, line, mat_start)
        result.append(norm_dict)
    return result
"""


# pylint: disable=too-many-locals
def extract_numbers(line: str, is_ignore_currency_symbol: bool = False) -> List[Dict]:
    """Extract numbers.

    Args:
        line: the string to extract the numbers from.
        is_ignore_currency: if True, will capture the '3' in $3.

    """
    norm_line = remove_num_words_join_hyphen(line)
    if is_ignore_currency_symbol:
        norm_line = re.sub(CURRENCY_SYMBOL_EXACT_PAT, space_same_len, norm_line)
        norm_line = re.sub(CURRENCY_SYMBOL_EXACT_PAT, space_same_len, norm_line)

    result = []
    mat_list = NUMBER_PAT.finditer(norm_line)
    # some mat in list might have mutliple intergers, such as '2 3'
    # 'better_mat_list' will store the real numeric mat
    num_se_list = []  # type: List[Tuple[int, int, str]]
    for mat in mat_list:
        # Due ot our preference to parse English expressions mixed
        # with numbers, '2 4' might be captured in the match part.
        # We separate those numbers here.

        num_num_span_list = num_num_split(mat.group(), mat.start())

        if len(num_num_span_list) > 1:
            for offset_start, unused_offset_end, span_st in num_num_span_list:
                # string is 'm-3'
                # re.search(r'^[mtb]\-', span_st):
                if re.search(r'^[mtb]\-', span_st):
                    continue

                # skip '1-12'
                if span_st.count('-') > 1 or \
                   (span_st.count('-') == 1 and not span_st.startswith('-')):
                    continue

                mat2 = NUMBER_PAT.search(span_st)
                if mat2:
                    num_se_list.append((offset_start + mat2.start(),
                                        offset_start + mat2.end(),
                                        mat2.group()))
        else:
            # get rid of 'm-2'
            # remove_invalid_num_spans = invalid_num_split(mat)
            num_se_list.extend(invalid_num_split(mat))

    for num_start, num_end, num_st in num_se_list:
        # norm_st = text2int.normalize_comma_period(mat.group())
        # 2.3.4, or section head
        # 2018-01-01 or date
        if is_invalid_number_phrase(num_st) or \
           is_invalid_number_word(num_st):
            continue

        norm_dict = number_to_norm_dict(num_st, num_start, num_end, line)
        result.append(norm_dict)
    return result
