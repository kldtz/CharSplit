# pylint: disable=too-many-lines
import copy
import logging
import re
from typing import Dict, List, Match, Optional, Pattern, Tuple

from kirke.utils import ebantdoc4, ebsentutils, strutils

from kirke.utils import text2int, mathutils

from kirke.utils.text2int import remove_hyphen_among_num_words


# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# pylint: disable=line-too-long
# (?:^| |\()
# want to avoid "rst", which is "rs" + "t" where "t" is just trillion
# CURRENCY_PAT_ST and CURRENCY_PAT ARE NOT USED due to backtracking issues
# the other currency prefix and suffix patterns are used instead.
CURRENCY_PAT_ST = r'((\bUSD\b|\bEUR\b|\bGBP\b|\bCNY\b|\bJPY\b|\bINR\b|\bRupees?\b|\bRs\b\.?)|[\$€£円¥₹]) *({})'.format(text2int.numeric_regex_st)
CURRENCY_PAT = re.compile(CURRENCY_PAT_ST, re.I)


CURRENCY_SYMBOL_PREFIX_ST = r'((\bUSD\b|\bEUR\b|\bGBP\b|\bCNY\b|\bJPY\b|\bINR\b|\bRupees?\b|\bRs\b\.?)|[\$€£円¥₹])'
CURRENCY_SYMBOL_PREFIX_PAT = re.compile(CURRENCY_SYMBOL_PREFIX_ST, re.I)
CURRENCY_SYMBOL_EXACT_PAT = re.compile(r'([\$€£円¥₹]|\bR[Ss]\.)')
CURRENCY_SYMBOL_SUFFIX_PAT_ST = r'((USD|EUR|GBP|CNY|JPY|INR|Rs|[dD]ollars?|u\.\s*s\.\s*dollars?|' \
                         r'[eE]uros?|[pP]ounds?|[yY]uans?|[yY]ens?|[rR]upees?)\b|[\$€£円¥₹])'
CURRENCY_SYMBOL_SUFFIX_PAT = re.compile(CURRENCY_SYMBOL_SUFFIX_PAT_ST)

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

# TODO
# WARNING, this is no longer used due to backtracking take took long
# a new version of extract_percents() is used.  Not this regex.
# TO_FIX
# pylint: disable=line-too-long
PERCENT_PAT_ST = r'({})\s*(%|percent)'.format(text2int.numeric_regex_st_with_b)
PERCENT_PAT = re.compile(PERCENT_PAT_ST, re.I)

# print('\nPERCENT_PAT_ST:')
# print(PERCENT_PAT_ST)


# pylint: disable=too-many-return-statements
def normalize_currency_unit(line: str) -> str:
    lc_line = line.lower()
    if lc_line in set(['$', 'usd', 'dollar', 'dollars']) or \
       re.search(r'u\.\s*s\.\s*dollars?', lc_line):
        return 'USD'
    elif lc_line in set(['€', 'eur', 'euro', 'euros']):
        return 'EUR'
    elif lc_line in set(['£', 'gbp', 'pound', 'pounds']):
        return 'GBP'
    elif lc_line in set(['円', 'cny', 'yuan', 'yuans']):
        return 'CNY'
    elif lc_line in set(['¥', 'jpy', 'yen', 'yens']):
        return 'JPY'
    elif lc_line in set(['₹', 'inr', 'rupee', 'rupees', 'rs', 'rs.']):
        return 'INR'

    return 'UNKNOWN_CURRENCY'


def currency_to_norm_dict(cx_mat: Match, line: str) -> Dict:
    # print('  currency cx_mat group: {} {} [{}]'.format(cx_mat.start(), cx_mat.end(), cx_mat.group()))
    # for gi, group in enumerate(cx_mat.groups(), 1):
    #    print("    cx_mat.group #{}: [{}]".format(gi, cx_mat.group(gi)))
    norm_unit = normalize_currency_unit(cx_mat.group(1))
    norm_value = text2int.extract_number(cx_mat.group(3)).get('value', -1)
    norm_dict = {'norm': {'unit': norm_unit,
                          'value': norm_value},
                 'text': line[cx_mat.start():cx_mat.end()],
                 'start': cx_mat.start(),
                 'end': cx_mat.end()}
    return norm_dict


def currency_to_norm_dict_symbol(num_start: int,
                                 num_end: int,
                                 currency_start: int,
                                 currency_end: int,
                                 line: str) -> Dict:
    num_st = line[num_start:num_end]
    norm_value = text2int.extract_number(num_st).get('value', -1)
    norm_unit = normalize_currency_unit(line[currency_start:currency_end].strip())
    if num_start < currency_end:
        norm_dict = {'norm': {'unit': norm_unit,
                              'value': norm_value},
                     'text': line[num_start:currency_end],
                     'start': num_start,
                     'end': currency_end}
    else:
        norm_dict = {'norm': {'unit': norm_unit,
                              'value': norm_value},
                     'text': line[currency_start:num_end],
                     'start': currency_start,
                     'end': num_end}

    return norm_dict

def space_same_len(matchobj: Match) -> str:
    return ' ' * len(matchobj.group())

def extract_currencies(line: str) -> List[Dict]:
    norm_line = remove_hyphen_among_num_words(line)
    result = []
    # Handle 'XXX dollars' using simplified regex to avoid nasty backtracking.
    # Removing currency symbols before extract_numbers() because numbers
    # requires a non-word before it.  All other operations are kept the same
    # using the original text, norm_line (without hyphens).
    norm_line_no_currency_symbol = re.sub(CURRENCY_SYMBOL_EXACT_PAT, space_same_len, norm_line)
    number_dict_list = extract_numbers(norm_line_no_currency_symbol)

    mat_list = CURRENCY_SYMBOL_PREFIX_PAT.finditer(norm_line)
    number_idx = 0
    for mat in mat_list:
        post_num_start, post_num_end, number_idx = find_post_start_end_in_dict_list(number_dict_list,
                                                                                    number_idx,
                                                                                    mat.end(),
                                                                                    line)
        if post_num_start != -1:
            norm_dict = currency_to_norm_dict_symbol(post_num_start, post_num_end,
                                                     mat.start(), mat.end(),
                                                     line)
            result.append(norm_dict)

    prefix_se_list = [(adict['start'], adict['end']) for adict in result]

    mat_list = CURRENCY_SYMBOL_SUFFIX_PAT.finditer(norm_line)
    number_idx = 0
    for mat in mat_list:
        prev_num_start, prev_num_end, number_idx = find_prev_start_end_in_dict_list(number_dict_list,
                                                                                    number_idx,
                                                                                    mat.start(),
                                                                                    line)
        if prev_num_start != -1:
            if not mathutils.is_overlap_with_se_list((prev_num_start, mat.end()),
                                                     prefix_se_list):
                # add it only if doesn't overlap with $33,000 before.
                # $33,000 has precedence over 33,000$.  The problematic case is '33,000 $ 22,000'
                # inside a table
                norm_dict = currency_to_norm_dict_symbol(prev_num_start, prev_num_end,
                                                         mat.start(), mat.end(),
                                                         line)
                result.append(norm_dict)

    return result


PERCENT_SYMBOL_PAT_ST = r'(%|\bpercent\b)'
PERCENT_SYMBOL_PAT = re.compile(PERCENT_SYMBOL_PAT_ST, re.I)

def find_prev_start_end_in_dict_list(number_dict_list: List[Dict],
                                     number_idx: int,
                                     percent_start: int,
                                     line: str) -> Tuple[int, int, int]:
    """Find the number in number_dict_list that appears BEFORE the
       the percent start index.

    This is an optimized version to find the number matched with percent_start.
    The percent_start was originally the "%" sign, but it can be currency also.

    Args:
      number_dict_list: the list of numbers found in the line
      number_idx: the index to the number inspected so far.
    Returns:
      start, end: non-negative offset if the pairing is found
      next_number_idx: the index to the next number to search
                       in the number_dict_list
    """
    list_len = len(number_dict_list)
    idx = number_idx
    ok_start, ok_end = -1, -1
    while idx < list_len:
        ndict = number_dict_list[idx]
        dstart, dend = ndict['start'], ndict['end']
        if dend < percent_start:
            # if the number appears before the symbol index, it is
            # a potential match, but not necessary a match
            ok_start, ok_end = dstart, dend
        elif dend == percent_start:
            # is it right before the index to symbo, it must be a match
            ok_start, ok_end = dstart, dend
            return ok_start, ok_end, idx + 1
        else:
            # now, check if the potential match is really a match
            # by inspecting if the span between them only has spaces
            if ok_end != -1 and \
               not line[ok_end:percent_start].strip():
                return ok_start, ok_end, idx
            return -1, -1, idx
        idx += 1

    # last one
    if ok_end != -1 and \
       not line[ok_end:percent_start].strip():
        return ok_start, ok_end, idx

    return -1, -1, list_len


def find_post_start_end_in_dict_list(number_dict_list: List[Dict],
                                     number_idx: int,
                                     percent_end: int,
                                     line: str) -> Tuple[int, int, int]:
    """Find the number in number_dict_list that appears AFTER the
       the percent start index.

    This is an optimized version to find the number matched with percent_start.
    The percent_start was originally the "%" sign, but it can be currency also.

    Args:
      number_dict_list: the list of numbers found in the line
      number_idx: the index to the number inspected so far.
    Returns:
      start, end: non-negative offset if the pairing is found
      next_number_idx: the index to the next number to search
                       in the number_dict_list
    """
    list_len = len(number_dict_list)
    idx = number_idx
    ok_start, ok_end = -1, -1
    while idx < list_len:
        ndict = number_dict_list[idx]
        dstart, dend = ndict['start'], ndict['end']
        if dstart < percent_end:
            pass
        elif dstart == percent_end:
            # is it right before the index to symbo, it must be a match
            ok_start, ok_end = dstart, dend
            return ok_start, ok_end, idx + 1
        else:
            between_span = line[percent_end:dstart].strip()
            if not between_span:
                return dstart, dend, idx
            return -1, -1, idx
        idx += 1
    return -1, -1, list_len


FRACTION_PAT_1 = re.compile(r'\b(\d{1,2})/(\d{1,3})(\s*(th|rd))*\b', re.I)

vulgar_fractions = ['½',
                    '⅓', '⅔',
                    '¼', '¾',
                    '⅕', '⅖', '⅗', '⅘',
                    '⅙', '⅚',
                    '⅐',
                    '⅛', '⅜', '⅝', '⅞',
                    '⅑',
                    '⅒']

vulgar_frac_val_list = [0.5,
                        0.33, 0.66,
                        0.25, 0.75,
                        0.2, 0.4, 0.6, 0.8,
                        round(1/6, 2), round(5/6, 2),
                        round(1/7, 2),
                        0.125, 0.375, 0.625, 0.875,
                        round(1/9, 2),
                        0.1]

# FRACTION_PAT_2 = re.compile(r'\b(\d+)\s*({})'.format('|'.join(vulgar_fractions)),
#                             re.I)

FIRST_19_ST_LIST = ["zero", "one", "two", "three", "four", "five", "six",
                    "seven", "eight", "nine", "ten", "eleven", "twelve",
                    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                    "eighteen", "nineteen"]

FIRST_19_ORD_ST_LIST = ["zeroth", "first", "half", "third", "fourth", "fifth", "sixth",
                        "seventh", "eighth", "ninth", "tenth", "eleventh", "twelveth",
                        "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth",
                        "eighteenth", "nineteenth",
                        # 'second' is for 'half'
                        'second']


FRACTION_PAT_2 = re.compile(r'\b({})[\-/\s]({})\b'.format('|'.join(FIRST_19_ST_LIST),
                                                          '|'.join(FIRST_19_ORD_ST_LIST)),
                            re.I)

def ordinal_word_to_number(line: str) -> int:
    """Convert an ordinal word (up-to 19) into a number."""
    lc_line = line.lower()
    if lc_line == 'second':
        return 2
    try:
        idx = FIRST_19_ORD_ST_LIST.index(lc_line)
    except ValueError:
        idx = -1
    return idx


def extract_fractions(line: str) -> List[Dict]:
    result = []  # type: List[Dict]

    mat_list = FRACTION_PAT_1.finditer(line)
    for mat in mat_list:
        val1 = int(mat.group(1))
        val2 = int(mat.group(2))

        if val2 >= 1 and val2 <= 100:
            norm_dict = {'start': mat.start(),
                         'end': mat.end(),
                         'text': mat.group(),
                         # cannot round this, because other use this
                         # to get back how many month, 1/60 -> 60 months
                         'norm': {'value': val1 / val2}}

            result.append(norm_dict)

    mat_list = FRACTION_PAT_2.finditer(line)
    for mat in mat_list:
        val1 = FIRST_19_ST_LIST.index(mat.group(1).lower())
        val2 = FIRST_19_ORD_ST_LIST.index(mat.group(2).lower())

        if val2 >= 1 and val2 <= 100:
            norm_dict = {'start': mat.start(),
                         'end': mat.end(),
                         'text': mat.group(),
                         # cannot round this, because other use this
                         # to get back how many month, 1/60 -> 60 months
                         'norm': {'value': val1 / val2}}
            result.append(norm_dict)

    mat_list = FRACTION_PAT_2.finditer(line)

    return result


FRACTION_PERCENT_PAT_1 = re.compile(r'\b(\d+) *(\d{1,2})/(\d{1,3})\s*(%|percent\b)',
                                    re.I)
FRACTION_PERCENT_PAT_2 = re.compile(r'\b(\d+) *({})\s*(%|percent\b)'.format('|'.join(vulgar_fractions)),
                                    re.I)

def extract_fraction_percents(line: str) -> List[Dict]:
    result = []  # type: List[Dict]

    mat_list = FRACTION_PERCENT_PAT_1.finditer(line)
    for mat in mat_list:
        val1 = int(mat.group(1))
        val2 = int(mat.group(2))
        val3 = int(mat.group(3))

        if val3 >= 1 and val3 <= 100:
            norm_dict = {'start': mat.start(),
                         'end': mat.end(),
                         'text': mat.group(),
                         'norm': {'value': val1 + round(val2 / val3, 2),
                                  'unit': '%'}}
            result.append(norm_dict)

    mat_list = FRACTION_PERCENT_PAT_2.finditer(line)
    for mat in mat_list:
        val1 = int(mat.group(1))
        val_idx = vulgar_fractions.index(mat.group(2))
        fval2 = vulgar_frac_val_list[val_idx]

        norm_dict = {'start': mat.start(),
                     'end': mat.end(),
                     'text': mat.group(),
                     'norm': {'value': val1 + fval2,
                              'unit': '%'}}
        result.append(norm_dict)

    return result

def percent_to_norm_dict(prev_num_start: int,
                         prev_num_end: int,
                         percent_end: int,
                         line: str) -> Dict:
    norm_value = -1
    num_st = line[prev_num_start:prev_num_end]
    norm_value = text2int.extract_number(num_st).get('value', -1)
    norm_dict = {'norm': {'unit': '%',
                          'value': norm_value},
                 'text': line[prev_num_start:percent_end],
                 'start': prev_num_start,
                 'end': percent_end}
    return norm_dict


def extract_percents(line: str) -> List[Dict]:
    norm_line = remove_hyphen_among_num_words(line)
    result = []
    # try detect '33 1/3%' first
    fraction_dict_list = extract_fraction_percents(line)
    for fraction_dict in fraction_dict_list:
        result.append(fraction_dict)

    prefix_se_list = [(adict['start'], adict['end']) for adict in result]

    number_dict_list = extract_numbers(line)
    mat_list = PERCENT_SYMBOL_PAT.finditer(norm_line)
    number_idx = 0
    for mat in mat_list:
        prev_num_start, prev_num_end, number_idx = \
            find_prev_start_end_in_dict_list(number_dict_list,
                                             number_idx,
                                             mat.start(),
                                             line)
        if prev_num_start != -1 and \
           not mathutils.is_overlap_with_se_list((prev_num_start, mat.end()),
                                                 prefix_se_list):

            norm_dict = percent_to_norm_dict(prev_num_start,
                                             prev_num_end,
                                             mat.end(),
                                             line)
            result.append(norm_dict)

    return result


TIME_DURATION_PAT_ST = r'\b(days?|weeks?|months?|years?|decades?)\b'
TIME_DURATION_PAT = re.compile(TIME_DURATION_PAT_ST, re.I)

def time_duration_to_norm_dict(prev_num_start: int,
                               prev_num_end: int,
                               duration_end: int,
                               line: str) -> Dict:
    norm_value = -1
    num_st = line[prev_num_start:prev_num_end]
    norm_value = text2int.extract_number(num_st).get('value', -1)

    duration_st = line[prev_num_end:duration_end].strip()
    if duration_st.endswith('s'):
        duration_st = duration_st[:-1]
    norm_dict = {'norm': {'unit': duration_st,
                          'value': norm_value},
                 'text': line[prev_num_start:duration_end],
                 'start': prev_num_start,
                 'end': duration_end}
    return norm_dict


def extract_time_durations(line: str) -> List[Dict]:
    # norm_line = remove_hyphen_among_num_words(line)
    # handling '18-month anniversary'
    norm_line = line.replace('-', ' ')
    result = []

    mat_list = list(TIME_DURATION_PAT.finditer(norm_line))

    number_dict_list = extract_number_paren_numbers(line)
    number_idx = 0
    for mat in mat_list:
        prev_num_start, prev_num_end, number_idx = \
            find_prev_start_end_in_dict_list(number_dict_list,
                                             number_idx,
                                             mat.start(),
                                             line)
        if prev_num_start != -1:
            norm_dict = time_duration_to_norm_dict(prev_num_start,
                                                   prev_num_end,
                                                   mat.end(),
                                                   line)
            result.append(norm_dict)

    # remember the start and end of above duration to avoid
    # detecting duplicates
    prefix_se_list = [(adict['start'], adict['end']) for adict in result]

    number_dict_list = extract_numbers(line)

    number_idx = 0
    for mat in mat_list:
        prev_num_start, prev_num_end, number_idx = \
            find_prev_start_end_in_dict_list(number_dict_list,
                                             number_idx,
                                             mat.start(),
                                             norm_line)
        if prev_num_start != -1 and \
           not mathutils.is_overlap_with_se_list((prev_num_start, mat.end()),
                                                 prefix_se_list):
            norm_dict = time_duration_to_norm_dict(prev_num_start,
                                                   prev_num_end,
                                                   mat.end(),
                                                   norm_line)
            result.append(norm_dict)

    return result


def nth_time_duration_to_norm_dict(prev_num_start: int,
                                   prev_num_end: int,
                                   num_dict: Dict,
                                   duration_end: int,
                                   line: str) -> Dict:
    norm_value = -1
    # num_st = line[prev_num_start:prev_num_end]
    # norm_value = text2int.extract_number(num_st).get('value', -1)
    norm_value = num_dict['norm']['value']

    duration_st = line[prev_num_end:duration_end].strip()
    if duration_st.endswith('s'):
        duration_st = duration_st[:-1]
    norm_dict = {'norm': {'unit': duration_st,
                          'value': norm_value},
                 'text': line[prev_num_start:duration_end],
                 'start': prev_num_start,
                 'end': duration_end}
    return norm_dict


def extract_nth_time_durations(line: str) -> List[Dict]:
    norm_line = remove_hyphen_among_num_words(line)
    result = []

    mat_list = list(TIME_DURATION_PAT.finditer(norm_line))

    number_dict_list = extract_ordinal_numbers(line)
    number_idx = 0
    for mat in mat_list:
        prev_num_start, prev_num_end, number_idx = \
            find_prev_start_end_in_dict_list(number_dict_list,
                                             number_idx,
                                             mat.start(),
                                             line)
        if prev_num_start != -1:
            ord_num_dict = number_dict_list[number_idx - 1]
            norm_dict = nth_time_duration_to_norm_dict(prev_num_start,
                                                       prev_num_end,
                                                       ord_num_dict,
                                                       mat.end(),
                                                       line)
            result.append(norm_dict)

    return result


def number_to_norm_dict(num_st: str,
                        start: int,
                        end: int,
                        line: str) -> Dict:
    # print('  number cx_mat group: {} {} [{}]'.format(cx_mat.start(), cx_mat.end(), cx_mat.group()))
    # for gi, group in enumerate(cx_mat.groups(), 1):
    #     print("    numb cx_mat.group #{}: [{}]".format(gi, cx_mat.group(gi)))
    norm_value = -1
    if num_st:
        norm_value = text2int.extract_number(num_st).get('value', -1)

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

D_DASH_D_REGEX = re.compile(r'\d+(\-\d+)+')

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
    """return True if The number is acronyms for 'b', 'm', or 't' or
       digits followed by those characters.  They are not
       valid numbers now.
    """

    if line.lower() in set(['b', 'm', 't']):
        return True

    if NUM_MBT_REGEX.search(line):
        return True

    # a date, 02-03, or phone number
    if D_DASH_D_WORD_REGEX.search(line):
        return True

    return False

FLOAT_WITH_WORD_REGEX = re.compile(r'\b(\d+\.\d+)\s+(\S+)')

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


DIGIT_ORDINAL_PAT_ST = r'\b(\d+)(st|nd|rd|th)\b'
DIGIT_ORDINAL_PAT = re.compile(DIGIT_ORDINAL_PAT_ST, re.I)

WORD_ORDINAL_PAT = re.compile('({})'.format('|'.join(FIRST_19_ORD_ST_LIST)), re.I)

def extract_ordinal_numbers(line: str) -> List[Dict]:
    result = []  # type: List[Dict]
    mat_list = DIGIT_ORDINAL_PAT.finditer(line)
    for mat in mat_list:
        adict = {'norm': {'value': int(mat.group(1)),
                          'ordinal': True},
                 'text': mat.group(),
                 'start': mat.start(),
                 'end': mat.end()}
        result.append(adict)

    mat_list = WORD_ORDINAL_PAT.finditer(line)
    for mat in mat_list:
        val1 = FIRST_19_ORD_ST_LIST.index(mat.group().lower())
        adict = {'norm': {'value': val1,
                          'ordinal': True},
                 'text': mat.group(),
                 'start': mat.start(),
                 'end': mat.end()}
        result.append(adict)

    return result


# pylint: disable=too-many-locals
def extract_numbers(line: str, is_ignore_currency_symbol: bool = False) -> List[Dict]:
    """Extract numbers.

    Args:
        line: the string to extract the numbers from.
        is_ignore_currency: if True, will capture the '3' in $3.

    """
    norm_line = remove_hyphen_among_num_words(line)
    if is_ignore_currency_symbol:
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


def get_nspace_char_at_after(idx: int, line: str) -> Tuple[int, str]:
    """Return the index + 1 and non-space character at or after idx in line."""
    len_line = len(line)
    if idx >= len_line:
        return len_line, ''

    while idx < len_line:
        if line[idx] != ' ':
            return idx+1, line[idx]
        idx += 1

    return len_line, ''


def extract_number_paren_numbers(line: str) -> List[Dict]:
    """Extract number (number), i.e. 'three (3)' or 3 (three) .

    Args:
        line: the string to extract the numbers from.

    """
    # this is really a hack
    line = line.replace(';', ' ')
    number_dict_list = extract_numbers(line)

    if len(number_dict_list) <= 1:
        return []

    # take any consecutive numbers and see if there is a '(' between them.
    len_num_list = len(number_dict_list)
    result = []  # type: List[Dict]
    prev_num_dict = number_dict_list[0]
    idx = 1
    while idx < len_num_list:
        num_dict = number_dict_list[idx]
        gap_st = line[prev_num_dict['end']:num_dict['start']].strip()

        if gap_st == '(':
            achar_idx_plus_1, achar = get_nspace_char_at_after(num_dict['end'], line)
            if  achar == ')':
                merged_dict = copy.copy(prev_num_dict)
                merged_dict['end'] = achar_idx_plus_1
                merged_dict['text'] = line[prev_num_dict['start']:
                                           achar_idx_plus_1]
                result.append(merged_dict)

                # move the index accordingly
                if idx + 1 < len_num_list:
                    prev_num_dict = number_dict_list[idx + 1]
                    idx += 1
        prev_num_dict = num_dict
        idx += 1
    return result



# pylint: disable=too-few-public-methods
class RegexContextGenerator:

    # pylint: disable=too-many-arguments
    def __init__(self,
                 num_prev_words: int,
                 num_post_words: int,
                 center_regex: Pattern,
                 candidate_type: str,
                 length_min: int = 0,
                 group_num: int = 1) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words
        self.center_regex = center_regex
        self.candidate_type = candidate_type
        self.length_min = length_min
        self.group_num = group_num


    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def get_candidates_from_text(self,
                                 nl_text: str,
                                 group_id: int = 0,
                                 # pylint: disable=line-too-long
                                 label_ant_list_param: Optional[List[ebsentutils.ProvisionAnnotation]] = None,
                                 label_list_param: Optional[List[bool]] = None,
                                 label: Optional[str] = None):
        # pylint: disable=line-too-long
        label_ant_list, label_list = [], []  # type: List[ebsentutils.ProvisionAnnotation], List[bool]
        if label_ant_list_param is not None:
            label_ant_list = label_ant_list_param
        if label_list_param is not None:
            label_list = label_list_param

        candidates = [] # type: List[Dict]
        group_id_list = [] # type: List[int]

        match_dict_list = []  # type: List[Dict]
        if self.candidate_type == 'NUMBER':
            # is_ignore_currency_symbol = True in order to
            # capture the number '3' in '$3'
            match_dict_list = extract_numbers(nl_text, is_ignore_currency_symbol=True)
        elif self.candidate_type == 'CURRENCY':
            match_dict_list = extract_currencies(nl_text)
        elif self.candidate_type == 'PERCENT':
            match_dict_list = extract_percents(nl_text)
        else:
            matches = self.center_regex.finditer(nl_text)
            for match in matches:
                match_start, match_end = match.span(self.group_num)
                match_str = match.group(self.group_num)
                match_dict_list.append({'start': match_start,
                                        'end': match_end,
                                        'text': match.group()})

        """
        matches = self.center_regex.finditer(nl_text)
        for match in matches:
            match_start, match_end = match.span(self.group_num)
            match_str = match.group(self.group_num)
        """

        for match_dict in match_dict_list:
            match_start, match_end = match_dict['start'], match_dict['end']
            match_str = match_dict['text']

            norm_dict = match_dict.get('norm', {})

            is_label = ebsentutils.check_start_end_overlap(match_start,
                                                           match_end,
                                                           label_ant_list)
            prev_n_words, prev_spans = strutils.get_prev_n_clx_tokens(nl_text,
                                                                      match_start,
                                                                      self.num_prev_words)
            post_n_words, post_spans = strutils.get_post_n_clx_tokens(nl_text,
                                                                      match_end,
                                                                      self.num_post_words)
            new_bow = '{} {} {}'.format(' '.join(prev_n_words),
                                        match_str,
                                        ' '.join(post_n_words))

            #update span based on window size
            new_start = match_start
            new_end = match_end
            if prev_spans:
                new_start = prev_spans[0][0]
            if post_spans:
                new_end = post_spans[-1][-1]

            """
            norm_dict = {}  # type: Dict
            if self.candidate_type == 'CURRENCY':
                norm_dict = currency_to_norm_dict(match, nl_text)
            elif self.candidate_type == 'NUMBER':
                # need to check for valid number because the extractor is
                # quite liberal.  Will catch sechead, such has 1.1.2
                norm_st = text2int.normalize_comma_period(match.group())
                if norm_st.count('.') >= 2:
                    continue
                norm_dict = number_to_norm_dict(match, nl_text)
            elif self.candidate_type == 'PERCENT':
                norm_dict = percent_to_norm_dict(match, nl_text)
            """

            a_candidate = {'candidate_type': self.candidate_type,
                           'bow_start': new_start,
                           'bow_end': new_end,
                           'text': new_bow,
                           'start': match_start,
                           'end': match_end,
                           'prev_n_words': ' '.join(prev_n_words),
                           'post_n_words': ' '.join(post_n_words),
                           'chars': match_str}
            if norm_dict:
                a_candidate['norm'] = norm_dict

            candidates.append(a_candidate)
            group_id_list.append(group_id)
            if is_label:
                a_candidate['label_human'] = label
                label_list.append(True)
            else:
                label_list.append(False)

        # remove any candidate that is >= min_length
        filtered_candidates = []  # type: List[Dict]
        filtered_label_list = []  # type: List[bool]
        filtered_group_id_list = []  # type: List[int]
        for candidate, cand_label, cand_group_id in zip(candidates,
                                                        label_list,
                                                        group_id_list):
            if len(candidate['chars']) >= self.length_min:
                filtered_candidates.append(candidate)
                filtered_label_list.append(cand_label)
                filtered_group_id_list.append(cand_group_id)
        return filtered_candidates, filtered_group_id_list, filtered_label_list

    # pylint: disable=too-many-locals
    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                label: Optional[str] = None) \
                                -> List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                              List[Dict],
                                              List[bool],
                                              List[int]]]:

        if 'length_min' not in self.__dict__:
            self.length_min = 0
        if 'group_num' not in self.__dict__:
            self.group_num = 1
        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):  # these are ebantdoc4
            label_list = []   # type: List[bool]

            #creates list of ants for a specific provision
            ant_list = antdoc.prov_annotation_list
            label_ant_list = []  # type: List[ebsentutils.ProvisionAnnotation]
            for ant in ant_list:
                if ant.label == label:
                    label_ant_list.append(ant)

            #gets text based on document type
            if antdoc.doc_format in set([ebantdoc4.EbDocFormat.html,
                                         ebantdoc4.EbDocFormat.html_nodocstruct,
                                         ebantdoc4.EbDocFormat.other]):
                nl_text = antdoc.text
            else:
                nl_text = antdoc.get_nl_text()

            if group_id % 10 == 0:
                logger.debug('RegexContextGenerator.documents_to_candidates(), group_id = %d',
                             group_id)

            candidates, group_id_list, label_list = self.get_candidates_from_text(nl_text,
                                                                                  group_id=group_id,
                                                                                  label_ant_list_param=label_ant_list,
                                                                                  label_list_param=label_list,
                                                                                  label=label)
            result.append((antdoc, candidates, label_list, group_id_list))
        return result
