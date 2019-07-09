
import logging
import re
# pylint: disable=unused-import
from typing import Dict, List, Match, Tuple, Union

from kirke.utils import mathutils, text2int
from kirke.utils.unicodeutils import normalize_dbcs_sbcs

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IS_DEBUG = False

UNITS_KANJI = ['零', '一', '二', '三',
               '四', '五', '六', '七',
               '八', '九']

SCALES_KANJI = {'十': (10, 0),
                '百': (100, 0),
                '千': (1000, 0),
                '万': (10000, 0),
                '億': (100000000, 0),
                '兆': (1000000000000, 0)}

def init_word_scale_increment_map() -> Dict[str, Tuple[int, int]]:
    word_scale_incr_map = {}
    for idx, word in enumerate(UNITS_KANJI):
        word_scale_incr_map[word] = (1, idx)
    for word, aval in SCALES_KANJI.items():
        word_scale_incr_map[word] = aval
    word_scale_incr_map['〇'] = (0, 0)
    word_scale_incr_map['壱'] = (1, 1)
    word_scale_incr_map['弐'] = (1, 2)
    word_scale_incr_map['参'] = (1, 3)
    # for 元年, but this is also a Chinese currency unit?
    word_scale_incr_map['元'] = (1, 1)
    # for synonyms
    # 拾　佰陌 仟阡 　仟阡　萬
    word_scale_incr_map['拾'] = (10, 0)
    word_scale_incr_map['佰'] = (100, 0)
    word_scale_incr_map['陌'] = (100, 0)
    word_scale_incr_map['仟'] = (1000, 0)
    word_scale_incr_map['阡'] = (1000, 0)
    word_scale_incr_map['萬'] = (10000, 0)

    # chinese
    word_scale_incr_map['两'] = (1, 2)
    word_scale_incr_map['亿'] = (100000000, 0)
    return word_scale_incr_map


# a dictionary of the word, with (scale, increment) as value
WORD_SCALE_INCR_MAP = init_word_scale_increment_map()

SCALE_TOKEN_SET = set(SCALES_KANJI.keys())

NUMERIC_WORDS = list(UNITS_KANJI + ['〇', '壱', '弐', '参',
                                    '元', '拾', '佰', '陌', '仟',
                                    '阡', '萬', '两', '亿'])

# NUMERIC_WORDS.extend([term for term in tens if term])
NUMERIC_WORDS.extend(SCALES_KANJI.keys())
NUMERIC_WORDS.sort(key=len, reverse=True)

# added a space to numeric expression
# because japanese OCR sometimes messed up with
# spaces
NUM_DIGIT_REGEX_ST = r'[-+]?[0-9,\.]*[0-9]|[-+]?[0-9 ]*[0-9]'

# '\d[\d\,\.]*' is very permissive
# works on 1,000,000.03
NUMERIC_REGEX_ST = r'(({})+・({})+|' \
                   r'({})|({})+|ゼロ)'.format('|'.join(NUMERIC_WORDS),
                                            '|'.join(NUMERIC_WORDS),
                                            NUM_DIGIT_REGEX_ST,
                                            '|'.join(NUMERIC_WORDS))

# ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5,
#                  'eighth':8, 'ninth':9, 'twelfth':12}
# ordinal_endings = [('ieth', 'y'), ('th', '')]

# one point 2 = simplified chinese 一点二, or 一分二
# ３０年月日
# '年５月'
# 翌月末日

# print('NUMERIC_REGEX_ST: {}'.format(NUMERIC_REGEX_ST))
# print('NUMERIC_WORDS_regex_st: {}'.format(NUMERIC_WORDS_regex_st))

NUM_REGEX = re.compile(NUMERIC_REGEX_ST)

# DBCS_ARABIC_NUMBERS = ['０', '１', '２', '３', '４'
#                        '５', '６', '７', '８', '９']
# full-width full stop
# '．'

def extract_numbers(line: str, is_norm_dbcs_sbcs=False) -> List[Dict]:
    # convert double-byte number to single-byte numbers
    # to simplify numeric normalization logic.
    if is_norm_dbcs_sbcs:
        line = normalize_dbcs_sbcs(line)
        # print('de-dbcs: [{}]'.format(line))

    mat_list = list(NUM_REGEX.finditer(line))

    num_span_list = []  # type: List[Tuple[int, int, str]]
    result = []  # type: List[Dict]
    for mat in mat_list:
        if mat.group().startswith(' '):
            num_prefix_spaces = len(re.search(r'^\s+', mat.group()).group())
            mat_start = mat.start() + num_prefix_spaces
            mat_end = mat.end()
            mat_stx = mat.group()[num_prefix_spaces:]
        else:
            mat_start = mat.start()
            mat_end = mat.end()
            mat_stx = mat.group()
        # '百' in '百分の二' should not be invalided
        # if mat_stx in set(['参', '仟', '阡', '千',
        #                    '万', '萬', '億', '兆', '万一']):
        if mat_stx in set(['参', '万一']):
            # part of a name or address
            # those scales shouldn't be by themself, should be
            # with another number specificier.
            # '参' is not a scale, but it goes with too many other
            # characters since it is a polysemy character.
            # can still remove in the future.
            # '万一' is word, meaning "just in case'
            continue
        numeric_span = (mat_start, mat_end, mat_stx)
        if IS_DEBUG:
            print('numeric_span: {}'.format(numeric_span))
        num_span_list.append(numeric_span)
        val = _text2number(mat_stx)
        adict = {'start': mat_start,
                 'end': mat_end,
                 'text': mat_stx,
                 'concept': 'number',
                 # the normalized value is in 'norm' field
                 'norm': {'value': val}}
        result.append(adict)
    return result


def extract_number(line: str, is_norm_dbcs_sbcs=False) -> Dict:
    """Extract the first number from line."""
    if is_norm_dbcs_sbcs:
        line = normalize_dbcs_sbcs(line)

    mat = NUM_REGEX.search(line)
    if mat:
        if mat.group().startswith(' '):
            num_prefix_spaces = len(re.search(r'^\s+', mat.group()).group())
            mat_start = mat.start() + num_prefix_spaces
            mat_end = mat.end()
            mat_stx = mat.group()[num_prefix_spaces:]
        else:
            mat_start = mat.start()
            mat_end = mat.end()
            mat_stx = mat.group()
        # '百' in '百分の二' should not be invalided
        # if mat_stx in set(['参', '仟', '阡', '千',
        #                    '万', '萬', '億', '兆', '万一']):
        if mat_stx in set(['参', '万一']):
            # part of a name or address
            # those scales shouldn't be by themself, should be
            # with another number specificier.
            # '参' is not a scale, but it goes with too many other
            # characters since it is a polysemy character.
            # can still remove in the future.
            # '万一' is word, meaning "just in case'
            return {}
        val = _text2number(mat_stx)
        adict = {'start': mat_start,
                 'end': mat_end,
                 'text': mat_stx,
                 'concept': 'number',
                 # the normalized value is in 'norm' field
                 'norm': {'value': val}}
        return adict
    return {}


def has_scale_token(tok_list: List[str]) -> bool:
    for tok in tok_list:
        if tok in SCALE_TOKEN_SET:
            return True
    return False


def digit_char_read_out(tok_list: List[str]) -> int:
    num_tok_list = []  # type: List[str]
    for tok in tok_list:
        if tok == '〇':
            num_tok_list.append('0')
        else:
            try:
                num_tok = UNITS_KANJI.index(tok)
                num_tok_list.append(str(num_tok))
            except ValueError:  # such as '元' is not in list
                # this error happens because we didn't remove
                # all currency symbol when trying to extract
                # numbers. is_ignore_currency_symbol=True
                pass
    return int(''.join(num_tok_list))


# https://stackoverflow.com/questions/493174/is-there-a-way-to-convert-number-words-to-integers
# pylint: disable=too-many-statements, too-many-return-statements
def _text2number(num_st: str) -> Union[int, float]:
    # assume num_st is normalize_dbcs_sbcs()
    # assuming the string is already single-byte, not double-byte
    # Because Japanese OCR has issues with spaces, we
    # assume the spaces between the nubmers doesn't matter.
    num_st = num_st.strip().replace(' ', '')
    if not num_st:
        return -1

    if num_st == 'ゼロ':
        return 0

    if '・' in num_st:
        return jp_float_word_num_to_float(num_st)

    is_negative = False
    if num_st[0] == '+':
        num_st = num_st[1:]
    elif num_st[0] == '-':
        num_st = num_st[1:]
        is_negative = True

    if '.' in num_st or ',' in num_st:
        norm_num_st = text2int.normalize_comma_period(num_st)
        return float(norm_num_st)
    elif num_st.isdigit():
        return int(num_st)

    tokens = list(num_st)
    # print('tokens: {}'.format(tokens))

    # handle '二零零四'
    # handle '一九五'
    if len(tokens) > 1 and not has_scale_token(tokens):
        return digit_char_read_out(tokens)

    current_thousand, current_hundred = 0, 0  # type: Tuple[float, float]
    current_ten, current = 0, 0  # type: Tuple[float, float]
    result = 0  # type: float

    # point_scale = 1  # type: Union[float, int]
    has_seen_point = False
    point_scale = 0.1
    for word in tokens:
        scale, increment = WORD_SCALE_INCR_MAP[word]

        if IS_DEBUG:
            print('  scale = {}, increment = {}, '
                  'has_seen_point = {}'.format(scale,
                                               increment,
                                               has_seen_point))
        if current < 1:  # current is never < 1 for non-float points
            # we don't want to reset 'current' in this case
            pass
        elif scale > 1:
            current = max(1, current)

        if IS_DEBUG:
            print('    current = {}'.format(current))

        if not has_seen_point:
            if scale == 0 and increment == 0:
                # zero
                pass
            elif scale == 1000:
                if current == 0:
                    current_thousand = 1000
                else:
                    current_thousand = current * scale
                current = 0
            elif scale == 100:
                if current == 0:
                    current_hundred = 100
                else:
                    current_hundred = current * scale
                current = 0
            elif scale == 10:
                if current == 0:
                    current_ten = 10
                else:
                    current_ten = current * scale
                current = 0
            elif scale > 1000:
                # the current from previous number is still valid
                pass
            else:
                current = increment

            if scale in set([10000, 100000000, 1000000000000]):
                if IS_DEBUG:
                    print('    in 10000, current = {}, thousand = {}, '
                          'hundred = {}, ten = {}, result = {}'.format(current,
                                                                       current_thousand,
                                                                       current_hundred,
                                                                       current_ten,
                                                                       result))
                # if scale > 1000:
                current = current_thousand + current_hundred + current_ten + current
                result += current * scale
                current_thousand, current_hundred = 0, 0
                current_ten, current = 0, 0
        else:
            result += point_scale * increment
            point_scale /= 10.0

        if IS_DEBUG:
            print('    current = {}, thousand = {}, hundred = {}, '
                  'ten = {}, result = {}'.format(current,
                                                 current_thousand,
                                                 current_hundred,
                                                 current_ten,
                                                 result))

    if not (current_thousand == 0 and \
            current_hundred == 0 and
            current_ten == 0 and
            current == 0):
        current = current_thousand + current_hundred + current_ten + current

    if is_negative:
        return -(result + current)
    return result + current

def jp_float_word_num_to_float(num_st: str) -> float:
    before_dot, after_dot = num_st.split('・')
    integer_part = _text2number(before_dot)
    mantissa = _text2number(after_dot)
    num_mantissa = len(after_dot)
    scale = pow(10, -num_mantissa)
    return integer_part + (mantissa * scale)


# fractions

FRACTION_PAT_ST = r'([0-9]+ +)?([0-9]+)/([0-9]+)|{}分の{}'.format(NUMERIC_REGEX_ST,
                                                                NUMERIC_REGEX_ST)
FRACTION_PAT = re.compile(FRACTION_PAT_ST, re.I)

def fraction_to_norm_dict(cx_mat: Match, line: str) -> Dict:
    if IS_DEBUG:
        print('  fraction cx_mat group: {} {} [{}]'.format(cx_mat.start(),
                                                           cx_mat.end(),
                                                           cx_mat.group()))
        for gi, unused_group in enumerate(cx_mat.groups(), 1):
            print("    perc cx_mat.group #{}: [{}]".format(gi, cx_mat.group(gi)))
    norm_value = -1.0

    try:
        if cx_mat.group(2):
            numerator = extract_number(cx_mat.group(2))['norm']['value']
            denominator = extract_number(cx_mat.group(3))['norm']['value']
            norm_value = numerator / denominator
            if cx_mat.group(1):
                norm_value += extract_number(cx_mat.group(1).strip())['norm']['value']
        elif cx_mat.group(4):
            numerator = extract_number(cx_mat.group(9))['norm']['value']
            denominator = extract_number(cx_mat.group(4))['norm']['value']
            norm_value = 0
            if denominator != 0:
                norm_value = numerator / denominator
    # pylint: disable=unused-variable
    except KeyError as e:
        # norm_value = -1
        logger.warning('failed to convert string to val in fraction_to_norm_dict(%s)',
                       cx_mat.group())

    norm_dict = {'norm': {'value': norm_value},
                 'text': line[cx_mat.start():cx_mat.end()],
                 'start': cx_mat.start(),
                 'concept': 'fraction',
                 'end': cx_mat.end()}
    return norm_dict


def extract_fractions(line: str, is_norm_dbcs_sbcs=False) -> List[Dict]:
    # convert double-byte number to single-byte numbers
    # to simplify numeric normalization logic.
    if is_norm_dbcs_sbcs:
        line = normalize_dbcs_sbcs(line)

    mat_list = list(FRACTION_PAT.finditer(line))

    num_span_list = []  # type: List[Tuple[int, int, str]]
    result = []  # type: List[Dict]
    for mat in mat_list:
        numeric_span = (mat.start(), mat.end(), mat.group())
        if IS_DEBUG:
            print('numeric_span: {}'.format(numeric_span))
        num_span_list.append(numeric_span)
        adict = fraction_to_norm_dict(mat, line)
        result.append(adict)
    return result


def extract_numbers_fractions(line: str, is_norm_dbcs_sbcs=False) -> List[Dict]:
    """Extract numbers and fractions.  If overlap, fractions win."""

    number_list = extract_numbers(line, is_norm_dbcs_sbcs)
    fraction_list = extract_fractions(line, is_norm_dbcs_sbcs)

    se_out_list = [(adict['start'], adict['end'], adict) for adict in fraction_list]

    fraction_se_list = [(adict['start'], adict['end']) for adict in fraction_list]
    for num_dict in number_list:
        if not mathutils.is_overlap_with_se_list((num_dict['start'], num_dict['end']),
                                                 fraction_se_list):
            se_out_list.append((num_dict['start'], num_dict['end'], num_dict))

    se_out_list.sort()
    result = [adict for start, end, adict in se_out_list]
    return result


def extract_roman_numbers(line: str, is_norm_dbcs_sbcs=False) -> List[Dict]:
    return text2int.extract_roman_numbers(line, is_norm_dbcs_sbcs)


def extract_numbers_fractions_romans(line: str, is_norm_dbcs_sbcs=False) -> List[Dict]:
    number_list = extract_numbers(line, is_norm_dbcs_sbcs)
    fraction_list = extract_fractions(line, is_norm_dbcs_sbcs)
    roman_num_list = text2int.extract_roman_numbers(line, is_norm_dbcs_sbcs)

    # add se_fractions for overlap check
    # add fractions
    se_out_list = [(adict['start'], adict['end'], adict) for adict in fraction_list]

    # check if number list overlap with fractions
    # add numbers
    fraction_se_list = [(adict['start'], adict['end']) for adict in fraction_list]
    for num_dict in number_list:
        if not mathutils.is_overlap_with_se_list((num_dict['start'], num_dict['end']),
                                                 fraction_se_list):
            se_out_list.append((num_dict['start'], num_dict['end'], num_dict))

    # add roman numbers
    for adict in roman_num_list:
        se_out_list.append((adict['start'], adict['end'], adict))

    se_out_list.sort()
    result = [adict for start, end, adict in se_out_list]
    return result
