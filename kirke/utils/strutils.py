#!/usr/bin/env python3

# pylint: disable=too-many-lines

import collections
import json
import logging
import os
import re
# pylint: disable=unused-import
from typing import Any, Dict, Generator, List, Match, Optional
from typing import Pattern, Set, Tuple, Union
import unicodedata
import urllib.parse

from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.regexp import RegexpTokenizer

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# https://github.com/python/typing/issues/182
# JSONType = Union[Dict[str, Any], List[Any]]
# pylint: disable=invalid-name
JSONType = Union[Dict, List[Any]]

# this include punctuations
CAP_SPACE_PAT = re.compile(r'^[A-Z\s\(\).,\[\]\-/\\{\}`\'"]+$')  # type: Pattern[str]

# pylint: disable=W0703, E1101

def sub_nltab_with_space(line: str) -> str:
    # return re.sub(r'[\s\t\r\n]+', ' ', line)
    return re.sub(r'[\s\t\r\n]', ' ', line)


def remove_space_nl(line: str) -> str:
    return re.sub(r'[\s\n]', '', line)


def loads(file_name: str) -> str:
    xst = ''
    try:
        with open(file_name, 'rt', newline='') as myfile:
            xst = myfile.read()
    except IOError as exc:
        logger.error("I/O error: %s in strutils.loads(%s)",
                     exc, file_name)
    except Exception as exc:  # handle other exceptions such as attribute errors
        # handle any other exception
        logger.error("Error %s", exc)
    return xst

def load_lines_with_offsets(file_name: str) -> Generator[Tuple[int, int, str], None, None]:
    offset = 0
    prev_line = None
    is_printed_empty_line = False
    with open(file_name, 'rt', newline='') as fin:
        for line in fin:
            orig_length = len(line)
            # replace non-breaking space with space for regex benefit
            new_line = line.replace('\xa0', ' ').strip()
            new_length = len(new_line)
            # remove the eoln char
            end = offset + new_length

            if offset != end:
                yield offset, end, new_line
                is_printed_empty_line = False
            else:  # is empty line
                if not prev_line and not is_printed_empty_line:
                    yield offset, end, new_line
                    is_printed_empty_line = True

            offset += orig_length
            prev_line = new_line


def dumps(xst: str, file_name: str) -> None:
    try:
        with open(file_name, 'wt') as myfile:
            myfile.write(xst)
            myfile.write(os.linesep)
    except IOError as exc:
        logger.error("I/O error(%s) in strutils.loads(%s)",
                     exc, file_name)
    # pylint: disable=W0703
    except Exception as exc:  # handle other exceptions such as attribute errors
        # handle any other exception
        logger.error("Error %s", exc)


def load_str_list(file_name: str) -> List[str]:
    st_list = []
    with open(file_name, 'rt', newline='') as fin:
        for line in fin:
            st_list.append(line.strip())
    return st_list


def load_non_empty_str_list(file_name: str) -> List[str]:
    st_list = []
    with open(file_name, 'rt', newline='') as fin:
        for line in fin:
            xline = line.strip()
            if xline:  # add only if not empty
                st_list.append(xline)
    return st_list


def load_lc_str_set(filename: str) -> Set[str]:
    aset = set([])
    with open(filename, 'rt') as fin:
        for line in fin:
            aset.add(line.strip().lower())
    return aset


def save_str_list(str_list: List[str], file_name: str) -> None:
    with open(file_name, 'wt') as fout:
        for line in str_list:
            fout.write(line)
            fout.write(os.linesep)

def load_json_list(file_name: str) -> JSONType:
    atext = loads(file_name)
    return json.loads(atext)

# This function is insufficient for Stanford CoreNLP.
# CoreNLP doesn't treate non-breaking space
# (https://en.wikipedia.org/wiki/Non-breaking_space) as space.
# Going to match first word instead to align CoreNLP output.
# st[i] == u'\u00a0' (non-breaking space)
def _get_num_prefix_space(line: str) -> int:
    i = 0
    st_size = len(line)
    while i < st_size:
        if line[i].isspace():
            i += 1
        else:
            break
    return i


def is_all_length_1_words(words: List[str]) -> bool:
    if not words:
        return False
    for word in words:
        if len(word) != 1:
            return False
    return True

WORD_LC_PAT = re.compile('^[a-z]+$')

# at least 2 char, starts with alpha
ALL_ALPHA_NUM_PAT = re.compile(r'^[a-zA-Z][a-zA-Z\d]+$')
ALL_ALPHAS_PAT = re.compile(r'^[a-zA-Z]+$')
ALL_ALPHAS_DOT_PAT = re.compile(r'^[a-zA-Z]+\.?$')
ANY_ALPHA_PAT = re.compile(r'[a-zA-Z]')


def is_word_all_lc(word: str) -> bool:
    return bool(WORD_LC_PAT.match(word))

# has more than 2 digits that's more than 3 width
TWO_GT_3_NUM_SEQ_PAT = re.compile(r'\d{3}.*\d{3}')
def has_likely_phone_number(line: str) -> bool:
    return bool(TWO_GT_3_NUM_SEQ_PAT.search(line))

def is_all_alphas(line: str) -> bool:
    return is_alpha_word(line)

def is_all_alphas_dot(line: str) -> bool:
    return bool(ALL_ALPHAS_DOT_PAT.match(line))

def is_alpha_word(line: str) -> bool:
    return bool(ALL_ALPHAS_PAT.match(line))

def is_both_alpha_and_num(line: str) -> bool:
    mat = ALL_ALPHA_NUM_PAT.match(line)
    if mat:
        if ANY_DIGIT_PAT.search(line) and ANY_ALPHA_PAT.search(line):
            return True
    return False


def count_word_category(word_list: List[str]) -> Tuple[int, int, int, int]:
    num_alpha, num_digit, num_dollar, num_other = 0, 0, 0, 0
    for word in word_list:
        if is_all_digits(word):
            num_digit += 1
        elif is_all_alphas(word) and len(word) >= 3:
            num_alpha += 1
        elif '$' in word:
            num_dollar += 1
        else:
            num_other += 1
    return num_alpha, num_digit, num_dollar, num_other


def is_all_dash_underline(line: str) -> bool:
    if not line:
        return False
    for xch in line:
        if not ((xch == '-') or (xch == '_')):
            return False
    return True


def is_all_caps_space(line: str) -> bool:
    return bool(CAP_SPACE_PAT.match(line))


def is_all_lower(line: str) -> bool:
    words = line.split()
    if not words:
        return False
    for word in words:
        if not word[0].islower():
            return False
    return True

def is_all_upper_words(words: List[str]) -> bool:
    if not words:
        return False
    for word in words:
        if not word.isupper():
            return False
    return True

def count_all_upper_words(line: str) -> int:
    words = line.split()
    count = 0
    for word in words:
        if word.isupper():
            count += 1
    return count


def is_cap_not_first_char(line: str) -> bool:
    if len(line) < 2:
        return False
    if not line[0].islower():
        return False
    for ach in line[1:]:
        if ach.isupper():
            return True
    return False


def bool_to_int(bool_val: bool) -> int:
    if bool_val:
        return 1
    return 0


def yesno_to_int(line: str) -> int:
    if line == 'yes':
        return 1
    elif line == 'no':
        return 0
    raise ValueError("Error in yesno_to_int({})".format(line))


def gen_ngram(word_list: List[str], max_n: int = 2) -> List[str]:
    result = []
    for i in range(len(word_list) - max_n + 1):
        ngram_words = [word_list[i + j] for j in range(max_n)]
        result.append(' '.join(ngram_words))
    return result



ALPHA_WORD_PAT = re.compile(r'[a-zA-Z]+')

ALPHANUM_WORD_PAT = re.compile(r'[a-zA-Z][a-zA-Z\d]+')

ALPHA_OR_NUM_WORD_PAT = re.compile(r'[a-zA-Z0-9]+')

ALL_PUNCT_PAT = re.compile(r"^[\(\)\.,\[\]\-/\\\{\}`'\"]+$")

ANY_PUNCT_PAT = re.compile(r"[\(\)\.,\[\]\-/\\\{\}`'\"]")

NUM_PERC_PAT = re.compile(r'^\s*\d+%\s*$')
NUM_PERIOD_PAT = re.compile(r'^\s*\d+\.\s*$')
DOLLAR_NUM_PAT = re.compile(r'^\s*\$\s*\d[\.\d]*\s*$')

def has_punct(line: str) -> bool:
    return bool(ANY_PUNCT_PAT.search(line))

def is_all_punct(line: str) -> bool:
    return bool(ALL_PUNCT_PAT.match(line))

def is_num_perc(line: str) -> bool:
    return bool(NUM_PERC_PAT.match(line))

def is_num_period(line: str) -> bool:
    return bool(NUM_PERIOD_PAT.match(line))

def is_dollar_num(line: str) -> bool:
    return bool(DOLLAR_NUM_PAT.match(line))


def get_alpha_words_gt_len1(line: str, is_lower: bool = True) -> List[str]:
    if is_lower:
        line = line.lower()
    return [word for word in ALPHA_WORD_PAT.findall(line) if len(word) > 1]


def get_alpha_words(line: str, is_lower: bool = True) -> List[str]:
    if is_lower:
        line = line.lower()
    return [word for word in ALPHA_WORD_PAT.findall(line)]


def get_alphanum_words_gt_len1(line: str, is_lower: bool = True) -> List[str]:
    if is_lower:
        line = line.lower()
    return [word for word in ALPHANUM_WORD_PAT.findall(line) if len(word) > 1]


def get_alphanum_words(line: str, is_lower=True) -> List[str]:
    if is_lower:
        line = line.lower()
    return [word for word in ALPHANUM_WORD_PAT.findall(line)]

def get_alpha_or_num_words(line: str, is_lower=True) -> List[str]:
    if is_lower:
        line = line.lower()
    return [word for word in ALPHA_OR_NUM_WORD_PAT.findall(line)]

def tokens_to_all_ngrams(word_list: List[str], max_n: int = 1) -> Set[str]:
    # unigram
    sent_wordset = set(word_list)
    # bigram and up
    for ngram_size in range(2, max_n+1):
        ngram_list = gen_ngram(word_list, max_n=ngram_size)
        sent_wordset |= set(ngram_list)

    return sent_wordset

# my non-standard version
#def is_punct_hacked(line):
#    if xst in '.,[]-/\{}':
#        return True
#    return False

def is_punct(line: str) -> bool:
    if line:
        return line[0] in r"().,[]-/\\{}`'\":;\?<>!"
    return False


def is_sent_punct(line: str) -> bool:
    return bool(line) and len(line) == 1 and line in r'.?!'


def is_not_sent_punct(line: str) -> bool:
    return bool(line) and is_punct(line) and not is_sent_punct(line)


def is_punct_not_period(line: str) -> bool:
    if line:
        return line[0] != '.' and is_punct(line)
    return False


def is_punct_notwork(line: str) -> bool:
    if line:
        return is_punct_core(line)
    return False

# '_' is for 'Title: ____________'.  It is end of sentence char
def is_eosent(line: str) -> bool:
    if line:
        return line[0] in '.!?_'
    return False

def is_punct_core(line: str) -> bool:
    return unicodedata.category(line[0]) == 'Po'

def is_lc(line: str) -> bool:
    return unicodedata.category(line[0]) == 'Ll'

def is_uc(line: str) -> bool:
    return unicodedata.category(line) == 'Lu'

def is_all_spaces(line: str) -> bool:
    if not line:
        return False
    for achar in line:
        if not is_space(achar):
            return False
    return True

def is_space(achr: str) -> bool:
    # print("is_space({}) = {}".format(line, unicodedata.category(line)))
    # return unicodedata.category(line) == 'Zs'
    # above doesn't seem to match '\n' or '\r'
    # there is also no-break space char???
    # return achr in '\n\r \t'
    # Below doesn't handle unicode, I am fine with it for now.
    # It's better than above.
    return achr.isspace()

# Ph.D Ph.D. I.B.M.  A. but not A
def is_acronym(input_word: str) -> bool:
    if '.' not in input_word:
        return False
    words = input_word.split('.')
    for word in words:
        if len(word) > 2:
            return False
    return True


# split into words after removing , ' "
def split_words(line: str) -> List[str]:
    words = re.split(r'[\s\,\'\"\-]+', line)  # lc_line.split()

    # now deal with period specifically
    result = []
    for word in words:
        if word:
            if word[-1] == '.':
                # is_acronym() has be before is_all_alphas(), otherwise, 'B.' would fail
                if is_acronym(word):
                    result.append(word)
                elif is_all_alphas(word[:-1]):
                    result.append(word[:-1])
                else:
                    result.append(word)
            else:   # 6.3
                result.append(word)
    return result


def is_all_title(line: str) -> bool:
    words = split_words(line)
    return is_all_title_words(words)


def is_all_title_words(words: List[str]) -> bool:
    if not words:
        return False
    for word in words:
        if not word[0].isalpha() or \
           not word[0].isupper():
            return False
    return True


ANY_ALPHA_PAT = re.compile(r'[a-z]', re.I)

def has_alpha(line: str) -> bool:
    return bool(ANY_ALPHA_PAT.search(line))


#
# ========== digits and numbers ==========
#
# 'digit' is an initeger
# 'number' is a floating point number, includint int

DIGIT_PAT = re.compile(r'^\s*\d+\s*$')
ALL_DIGITS_PAT = re.compile(r'^\d+$')
ANY_DIGIT_PAT = re.compile(r'\d')
PARENS_ALL_DIGITS_PAT = re.compile(r'^\(\d+\)$')


def is_all_digits(line: str) -> bool:
    return bool(ALL_DIGITS_PAT.match(line))


def is_parens_all_digits(line: str) -> bool:
    return bool(PARENS_ALL_DIGITS_PAT.match(line))


def is_all_digit_dot(line: str) -> bool:
    if not line:
        return False
    for xch in line:
        if not ((xch == '.') or (xch == '-') or xch.isdigit()):
            return False
    return True


def is_digit_st(line: str) -> bool:
    return bool(DIGIT_PAT.match(line))


def has_digit(line: str) -> bool:
    return bool(ANY_DIGIT_PAT.search(line))


def is_digit_core(line: str) -> bool:
    return unicodedata.category(line) == 'Nd'


def is_int(line: str) -> bool:
    try:
        int(line)
        return True
    except ValueError:
        return False


def to_int(line: str) -> int:
    try:
        result = int(line)
        return result
    except ValueError:
        raise ValueError("Error in to_int({})".format(line))

is_digits = is_digit_st

FLOAT_REGEX_ST = r'[-+]?\b[0-9]*\.?[0-9]+\b'
FLOAT_PAT = re.compile(FLOAT_REGEX_ST)

NUMBER_REGEX_ST = r'[-+]?\b[0-9,]*\.?[0-9]+'
NUMBER_PAT = re.compile(NUMBER_REGEX_ST + r'\b')

ALL_NUMBER_PAT = re.compile(r'^' + NUMBER_REGEX_ST + r'$')
CURRENCY_PAT = re.compile(r'^\$\s*' + NUMBER_REGEX_ST + r'$')
PERCENT_PAT = re.compile(r'^' + NUMBER_REGEX_ST + r'\s*%$')
# allowing '(201) 345-'
PHONE_NUMBER_PAT = re.compile(r'^(\(\d+\)|\d+\-\d*|\(\d+\)\s*\d+\-\d*)')
# allowing '(02/03/23]', but not for digit only prefix
NAIVE_DATE_PAT = re.compile(r'^(\d+$[\/\-]\d+[\/\-]\d+|\d+[\/\-]\d+)')


def find_numbers(line: str) -> List[str]:
    # return re.findall(r'(\d*\.\d+|\d+\.\d*|\d+)', line)
    return NUMBER_PAT.findall(line)

extract_numbers = find_numbers

def count_numbers(line: str) -> int:
    return len(find_numbers(line))

def find_number(line: str) -> Optional[Match[str]]:
    return NUMBER_PAT.search(line)

def has_number(line: str) -> bool:
    return bool(find_number(line))

def is_number(line: str) -> bool:
    return bool(ALL_NUMBER_PAT.search(line))

def is_currency(line: str) -> bool:
    return bool(CURRENCY_PAT.search(line))

def is_percent(line: str) -> bool:
    return bool(PERCENT_PAT.search(line))

def is_phone_number(line: str) -> bool:
    return bool(PHONE_NUMBER_PAT.search(line))

def is_naive_date(line: str) -> bool:
    return bool(NAIVE_DATE_PAT.search(line))

# http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float-in-python
def is_number_v2(line: str) -> bool:
    try:
        float(line)
        return True
    except ValueError:
        return False


# number, currency, percent,
# phone_ date, num_alpha,
# num_alphanum, num_bad_word,
# num_word,
# alphanum_words: str
# pylint: disable=too-many-locals
def remove_number_types(text: str) -> Tuple[int, int, int,
                                            int, int, int,
                                            int, int, int,
                                            str]:
    """This function is designed for tokenizing the text for table for
    classification purpose.

    A table contains a lot of condensed information which are not
    sentences.  We want to know if the content is of various numeric
    types.  One critical aspect we ant to capture is the number of
    words that are not even English words or numbers due to OCR errors.
    This is a reason why we are not using nltk's tokenizer.
    Not capturing them will make the tokenized result looks similar
    other sparse tables.
    """

    # words = re.split(r'[\s\,\'\"\-]+', text)
    # replace all parens
    text = re.sub(r'[\(\)\[\]]', ' ', text)
    # put together $ 23.33
    text = re.sub(r'\$\s+([\d.,]+)', r'$\1', text)
    words = text.split()
    alpha_words = []  # type: List[str]
    alphanum_words = []  # type: List[str]
    num_number, num_currency, num_percent = 0, 0, 0
    num_phone_number, num_date = 0, 0
    num_alpha_word, num_alphanum_word = 0, 0
    num_bad_word = 0

    # this will double count "$180,000,000" has 1 currency, 2 numbers
    for word in words:

        if not word.endswith('%'):
            # replace any punctuations at the end of a word
            word = re.sub(r'(\w)\W+$', r'\1', word)

        # skip words with acronyms, such A.
        if len(word) <= 1 and not is_number(word):
            continue

        # tried to handle "$", in $/Mwh
        # but doing so usually causes drop in recall

        # dates might have "/"
        if is_naive_date(word):
            num_date += 1
        else:
            tmp_word = word
            # pylint: disable=redefined-outer-name
            for word in tmp_word.split('/'):

                #if word == '$':
                    # adding this cause drop in recall
                    # alpha_words.append('SYMBOL_DOLLAR')
                    # num_alpha_word += 1
                #    pass

                # if ALL_MONTH_REGEX.match(word):
                #     word = 'SM_MONTH'

                if is_currency(word):
                    num_currency += 1
                elif is_percent(word):
                    num_percent += 1
                elif is_number(word):
                    num_number += 1
                elif is_phone_number(word):
                    num_phone_number += 1
                else:
                    # avoid all dashes and other stuff
                    if re.search(r'^[a-zA-Z_]+$', word):
                        alpha_words.append(word)
                        num_alpha_word += 1
                    elif re.search(r'^\w+$', word):
                        alphanum_words.append(word)
                        num_alphanum_word += 1
                    else:
                        num_bad_word += 1

    # once higher than 40, not meaningful
    num_number = min(num_number, 40)
    num_currency = min(num_currency, 40)
    num_percent = min(num_percent, 40)
    num_date = min(num_date, 40)
    num_phone_number = min(num_phone_number, 40)

    out_alphanum_words = list(alpha_words)
    out_alphanum_words.extend(alphanum_words)
    num_words = num_number + num_currency + \
                num_percent + num_phone_number + \
                num_date + num_alpha_word + \
                num_alphanum_word + num_bad_word

    return (num_number, num_currency,
            num_percent, num_phone_number,
            num_date, num_alpha_word,
            num_alphanum_word, num_bad_word,
            num_words,
            ' '.join(out_alphanum_words))


ROMAN_NUM_PAT = re.compile(r'^[ixv]+$', re.IGNORECASE)

def is_roman_number(line: str) -> bool:
    return bool(ROMAN_NUM_PAT.match(line))


NUM_ROMAN_PAT = re.compile(r'(([\(\“\”]?([\.\d]+|[ivx\d\.]+\b)[\)\“\”]?|'
                           r'[\(\“\”\.]?[a-z][\s\.\)\“\”])+)')

def is_header_number(line: str) -> bool:
    return bool(NUM_ROMAN_PAT.match(line))
# SECHEAD_NUMBER_PAT = re.compile('^\d[\d\.]+$')
# return SECHEAD_NUMBER_PAT.match(line)


# ========== telephone or SSN ==========
BIG_DASHED_DIGIT_PAT = re.compile(r'^(\d+)\-[\d\-]+$')
def is_dashed_big_number_st(line: str) -> bool:
    mat = BIG_DASHED_DIGIT_PAT.match(line)
    if mat:
        if int(mat.group(1)) > 20:
            return True
    return False


DASH_ONLY_LINE = re.compile(r'^\s*-+\s*$')

def is_dashed_line(line: str) -> bool:
    return bool(DASH_ONLY_LINE.match(line))


# this will match any substr, not just words
def are_all_substrs_in_st(substr_list: List[str], line: str) -> bool:
    lc_text = line.lower()
    return all(substr in lc_text for substr in substr_list)


def is_english_vowel(unich: str) -> bool:
    return unich in 'aeiouAEIOU'


def find_substr_indices(sub_strx: str, text: str) -> List[Tuple[int, int]]:
    return [(mat.start(), mat.end()) for mat in re.finditer(sub_strx, text)]


def find_all_indices(sub_strx: str, text: str) -> List[int]:
    return [mat.start() for mat in re.finditer(sub_strx, text)]


def count_date(line: str) -> int:
    return len(find_substr_indices(r'(\d{1,2}/\d{1,2}/(20|19)\d\d)', line))

# We encountered characters, 1, 2, 16, 31 in input to corenlp before.
# These characters passed through the processing and appeared as they are
# in the JSON output.  Unfortunately, these are not valid characters in JSON and
# should be escaped according to json.org.  Replace them for now
BAD_JSON_CTRL_CHARS = set([0, 1, 2, 3, 4, 5, 6, 7,
                           # 8, \b
                           # 9, \t
                           # 10 \n
                           11,
                           # 12 \f
                           # 13 \r
                           14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                           31,
                           127])

IGNORABLE_JSON_CTRL_CHARS = ''.join([chr(chx) for chx in BAD_JSON_CTRL_CHARS])
IGNORABLE_JSON_CTRL_PAT = re.compile(r'[' + IGNORABLE_JSON_CTRL_CHARS + ']')

def replace_ignorable_json_ctrl_chars(line: str) -> str:
    line = re.sub(IGNORABLE_JSON_CTRL_PAT, ' ', line)
    # pylint: disable=fixme
    # TODO, in future, regex these
    line = line.replace('\u201c', '"')
    line = line.replace('\u201d', '"')
    line = line.replace('\u2019', "'")
    return line


def corenlp_normalize_text(doc_text: str) -> str:
    line = replace_ignorable_json_ctrl_chars(doc_text)
    line = urllib.parse.quote(line)
    return line

def is_space_or_nl(xch: str) -> bool:
    return (xch == ' ' or
            xch == '\n' or
            xch == '\r')

def is_nl(xch: str) -> bool:
    return xch == '\n' or xch == '\r'


def is_double_quote(xch: str) -> bool:
    return xch in '“"”'


def dict_to_sorted_list(adict: Dict[Any, Any]) -> List[str]:
    return ['{}={}'.format(attr, value) for attr, value in sorted(adict.items())]


def space_repl_same_length(mat: Match[str]) -> str:
    return ' ' * len(mat.group())

def replace_dot3plus_with_spaces(line: str) -> str:
    return re.sub(r'([\.\-_][\.\-_][\.\-_]+)', space_repl_same_length, line)


# nobody is calling this?
NEXT_TOKEN_PAT = re.compile(r'\s*\S+')  # type: Pattern[str]

def find_next_token(line: str) -> Optional[Match[str]]:
    return NEXT_TOKEN_PAT.match(line)

def find_next_not_space_idx(line: str, idx: int) -> int:
    """Find the index of the start of next non-space char.

    Assume we are after end of a word, maybe a space, or a ','
    """
    if idx < 0 or idx >= len(line):
        return len(line)

    # if not line[idx].isspace():  # such as comma
    #    return idx

    for i in range(idx + 1, len(line)):
        if not line[i].isspace():
            return i
    # reached the end of line without find a space
    # return current position
    return idx


def find_previous_word(line: str, idx: int) -> Tuple[int, int, str]:
    """Find previous word.

    If the current index is an alphanum, it will get the previous word,
    not the current one.
    """
    if idx < 0 or idx >= len(line):
        return -1, -1, ''
    found_space = False  # in the middle of a word
    for i in range(idx, -1, -1):
        ch = line[i]
        if ch.isspace():
            found_space = True  # found end of a word
        elif ch.isalnum():
            if not found_space:
                continue
            end_idx = i + 1
            # go find the begin of a word
            for j in range(i-1, -1, -1):
                ch2 = line[j]
                if ch2.isspace():
                    return j+1, end_idx, line[j+1:end_idx]
                elif ch2.isalnum():
                    continue
                else:  # punctuation or anything else
                    return j+1, end_idx, line[j+1:end_idx]
            # this can only be reached if j == -1
            return 0, end_idx, line[0:end_idx]
        else:
            found_space = True  # any non-alphanum is a space
    return -1, -1, ''


# primitive version of getting words using regex

SIMPLE_WORD_PAT = re.compile(r'(\w+)')

def get_simple_words(text: str) -> List[Tuple[int, int, str]]:
    matches = SIMPLE_WORD_PAT.finditer(text)
    spans = [(m.start(), m.end(), m.group()) for m in matches]
    return spans


def get_prev_n_words(text: str, start: int, num_words: int) \
    -> Tuple[List[str], List[Tuple[int, int]]]:
    num_chars = num_words * 20  # avg word len is 7
    first_offset = max(0, start - num_chars)
    prev_text = text[first_offset:start]
    words_and_spans = get_simple_words(prev_text)[-num_words:]
    words = [x[-1] for x in words_and_spans]
    spans = [(x+first_offset, y+first_offset) for [x, y, z] in words_and_spans]
    return words[-num_words:], spans[-num_words:]


def get_post_n_words(text: str, end: int, num_words: int) \
    -> Tuple[List[str], List[Tuple[int, int]]]:
    num_chars = num_words * 20  # avg word len is 7
    last_offset = min(len(text), end + num_chars)
    post_text = text[end:last_offset]
    words_and_spans = get_simple_words(post_text)[:num_words]
    words = [x[-1] for x in words_and_spans]
    spans = [(x+end, y+end) for [x, y, z] in words_and_spans]
    return words[:num_words], spans[:num_words]


def get_lc_prev_n_words(text: str, start: int, num_words: int) \
    -> Tuple[List[str], List[Tuple[int, int]]]:
    words, spans = get_prev_n_words(text, start, num_words)
    return [word.lower() for word in words], spans


def get_lc_post_n_words(text: str, end: int, num_words: int) \
    -> Tuple[List[str], List[Tuple[int, int]]]:
    words, spans = get_post_n_words(text, end, num_words)
    return [word.lower() for word in words], spans


def has_quote(line: str) -> bool:
    return bool(re.search('[“"”]', line))

# adding period to tokens reduced 0.004 in F1 for effective_date.
# period by itself is also dangerous because it can be a part of abbreviation or
# floating point number.  Maybe better tokenizer in future.
# add \n to token reduced 0.01 in F1 for effective date.
# SIMPLE_WORD_QUOTE_PAT = re.compile(r'([“"”:\.\(\)]|\w+)')
SIMPLE_WORD_QUOTE_PAT = re.compile(r'([“"”:\(\)\n]|[\d,?]+|\d+\.\d+|\w+)')

# please not that because CountVectorizer does some word filtering,
# we must transform 1 char punctuations to alphabetized words, otherwise
# CountVectorizer just ignore them
def get_simple_words_with_quote(text: str,
                                is_quote: bool = False) \
                                -> List[Tuple[int, int, str]]:
    # this is the default
    if not is_quote:
        matches = SIMPLE_WORD_PAT.finditer(text)
        return [(m.start(), m.end(), m.group()) for m in matches]

    matches = SIMPLE_WORD_QUOTE_PAT.finditer(text)
    spans = []  # type: List[Tuple[int, int, str]]
    for mat in matches:
        word = mat.group()
        if word in '“"”':
            word = 'WxxQT'
        elif word == '(':
            word = 'WxxLP'
        elif word == ')':
            word = 'WxxRP'
        elif word == ':':
            word = 'WxxCL'
        elif word == '\n':
            word = 'WxxNL'
        elif re.match(r'[\d\.,]+', word):
            word = 'WxxDIGIT'
        # elif word == '.':
        #    word = 'WxxPD'
        spans.append((mat.start(), mat.end(), word))
    return spans


def get_prev_n_words_with_quote(text: str,
                                start: int,
                                num_words: int,
                                is_lower: bool = True,
                                is_quote: bool = False) \
    -> Tuple[List[str], List[Tuple[int, int]]]:
    num_chars = num_words * 20  # avg word len is 7
    first_offset = max(0, start - num_chars)
    prev_text = text[first_offset:start]
    if is_lower:
        prev_text = prev_text.lower()
    words_and_spans = get_simple_words_with_quote(prev_text,
                                                  is_quote)[-num_words:]
    words = [x[-1] for x in words_and_spans]
    spans = [(x+first_offset, y+first_offset) for [x, y, z] in words_and_spans]
    return words, spans


def get_post_n_words_with_quote(text: str,
                                end: int,
                                num_words: int,
                                is_lower: bool = True,
                                is_quote: bool = False) \
    -> Tuple[List[str], List[Tuple[int, int]]]:
    num_chars = num_words * 20  # avg word len is 7
    last_offset = min(len(text), end + num_chars)
    post_text = text[end:last_offset]
    if is_lower:
        post_text = post_text.lower()
    words_and_spans = get_simple_words_with_quote(post_text,
                                                  is_quote)[:num_words]
    words = [x[-1] for x in words_and_spans]
    spans = [(x+end, y+end) for [x, y, z] in words_and_spans]
    return words, spans


# from kirke/ebrules/dateutils.py, in other branches
MONTH_LIST = ['January', 'February', 'March', 'April', 'May',
              'June', 'July', 'August', 'September', 'October',
              'November', 'December',
              # for OCR misspelling?
              'M ay']
MONTH_ABBR_LIST = [r'Jan\.?', r'Feb\.?', r'Mar\.?', r'Apr\.?',
                   r'Jun\.?', r'Jul\.?', r'Sep\.?', r'Sept\.?', r'Oct\.?',
                   r'Nov\.?', r'Dec\.?']
ALL_MONTH_LIST = MONTH_LIST + MONTH_ABBR_LIST

ALL_MONTH_PAT = '|'.join(ALL_MONTH_LIST)

ALL_MONTH_REGEX = re.compile(r'\b(' + ALL_MONTH_PAT + r')\b', re.I)


def remove_ignorable_token(se_word_list: List[Tuple[int, int, str]]) \
    -> List[Tuple[int, int, str]]:
    """Remove ignorable token.

    Will remove 'a', 'the', and multiple 'SM_DIGIT', from 1,222 -> 1 111
    Will remove len(token) == 1
    """
    result = []
    prev_token = ''
    for se_token in se_word_list:
        unused_start, unused_end, token = se_token
        # skip 1 char words, and article 'a' or 'the'
        if len(token) == 1 or token == 'the' or token == 'an':
            pass
        elif token != prev_token:
            result.append(se_token)
        prev_token = token
    return result

WWPLUS_PAT = re.compile(r'\w\w+')

def get_regex_wwplus(line: str) -> List[str]:
    return WWPLUS_PAT.findall(line)

# For digits, we take , and .
# we take multiple \n as one
# patterns:
#    special characters: [“"”:\(\), quotes, left right parens, colon semicolon
#    digits: we take , and . with numbers
#    dollar and percent: $ => 'SM_DOLLAR', so are 'dollar' and 'dollars'
#    month, year
#    remove len(1) characters, and 'the', but kept the prepositions

SIMPLE_WORD_TOKEN_PAT = re.compile(r'([“"”:;\(\)]|\n+|\b[\d,\.]+\b|(\w\.)+|\w+)')

TREEBANK_WORD_TOKENIZER = TreebankWordTokenizer()

# please note that because CountVectorizer does some word filtering,
# we must transform 1 char punctuations to alphabetized words, otherwise
# CountVectorizer just ignore them
def get_clx_tokens(text: str) -> List[Tuple[int, int, str]]:
    '''
    Normalization is performed on words, so that
      - 'january' -> SM_MONTH
      - '1ddd' to '2ddd' are mapped SM_YEAR
      - d+ -> SM_DIGIT
      - mixed digit + alpha, such as 'form 10k' or '1st day', are kept as is
      - multiple token of same thing are collapse into just 1 token
      - 'a' and 'the' are removed, but not any other stop words
    stop words are kept.  We don't deal with stop words here.
    '''
    spans = []  # type: List[Tuple[int, int, str]]
    text = text.replace('"', '``')
    tok_spans = TREEBANK_WORD_TOKENIZER.span_tokenize(text)
    for start, end in tok_spans:
        word = text[start:end]
        if word in '``“"”':
            word = 'WxxQT'
        elif word == '(':
            word = 'WxxLP'
        elif word == ')':
            word = 'WxxRP'
        elif word == ';':
            word = 'WxxSCL'
        elif '\n' in word:
            word = 'WxxNL'
        elif ALL_MONTH_REGEX.match(word):
            word = 'SM_MONTH'
        elif re.match(r'\b[12]\d\d\d\b', word):
            word = 'SM_YEAR'
        elif re.match(r'\b\d+[a-zA-Z]+\b', word):
            # keep as is
            pass
        elif re.match(r'\b(\d+\.\d|\.\d+|\d+)\b', word):
            word = 'SM_DIGIT'
        elif word == ',':
            word = 'WxxCM'
        spans.append((start, end, word))
    spans = [x for x in spans if not (len(x[2]) < 2 or x[2] == 'the' or x[2] == 'an')]
    return spans


def get_prev_n_clx_tokens(text: str,
                          start: int,
                          num_words: int,
                          is_lower: bool = True) \
    -> Tuple[List[str], List[Tuple[int, int]]]:
    """Get n tokens for classification purpose.

    Normalization is performed on words, so that
      - 'january' -> SM_MONTH
      - '1ddd' to '2ddd' are mapped SM_YEAR
      - d+ -> SM_DIGIT
      - mixed digit + alpha, such as 'form 10k' or '1st day', are kept as is
      - multiple token of same thing are collapse into just 1
    stop words are kept.  We don't deal with stop words here.
    """
    num_chars = num_words * 20  # avg word len is 7
    first_offset = max(0, start - num_chars)
    prev_text = text[first_offset:start]
    if is_lower:
        prev_text = prev_text.lower()
    words_and_spans = get_clx_tokens(prev_text)[-num_words:]
    words = [x[-1] for x in words_and_spans]
    spans = [(x+first_offset, y+first_offset) for [x, y, z] in words_and_spans]
    return words, spans


def get_post_n_clx_tokens(text: str,
                          end: int,
                          num_words: int,
                          is_lower: bool = True) \
    -> Tuple[List[str], List[Tuple[int, int]]]:
    """Get n tokens for classification purpose.

    Normalization is performed on words, so that
      - 'january' -> SM_MONTH
      - '1ddd' to '2ddd' are mapped SM_YEAR
      - d+ -> SM_DIGIT
      - multiple token of same thing are collapse into just 1
    stop words are kept.  We don't deal with stop words here.
    """
    num_chars = num_words * 20  # avg word len is 7
    last_offset = min(len(text), end + num_chars)
    post_text = text[end:last_offset]
    if is_lower:
        post_text = post_text.lower()
    words_and_spans = get_clx_tokens(post_text)[:num_words]
    words = [x[-1] for x in words_and_spans]
    spans = [(x+end, y+end) for [x, y, z] in words_and_spans]
    return words, spans


def to_pos_neg_count(bool_list: List[bool]) -> str:
    pos_neg_counter = collections.Counter()  # type: collections.Counter
    pos_neg_counter.update(bool_list)
    return "num_pos = {}, num_neg = {}".format(pos_neg_counter.get(True, 0),
                                               pos_neg_counter.get(False, 0))


# https://stackoverflow.com/questions/9518806/how-to-split-a-string-on-whitespace-and-retain-offsets-and-lengths-of-words
def using_split2(line, _len=len) -> List[Tuple[int, int, str]]:
    words = line.split()
    index = line.index
    offsets = []  # type: List[Tuple[int, int, str]]
    append = offsets.append
    running_offset = 0
    for word in words:
        word_offset = index(word, running_offset)
        word_len = _len(word)
        running_offset = word_offset + word_len
        append((word_offset, running_offset, word))
    return offsets


# this probably should go into strutils
def split_with_offsets(line: str) -> List[Tuple[int, int, str]]:
    ret = [(mat.start(), mat.end(), mat.group())
           for mat in re.finditer(r'\S+', line)]
    return ret

def split_with_offsets_xparens(line: str) -> List[Tuple[int, int, str]]:
    ret = [(mat.start(), mat.end(), mat.group())
           for mat in re.finditer(r'\S+', line)]
    out_list = []
    for se_matst in ret:
        start, unused_end, mat_st = se_matst
        parts = re.split(r'([\(\)])', mat_st)  # capture separators also
        if len(parts) > 1:
            offset = start
            for part in parts:
                if part:
                    out_list.append((offset, offset + len(part), part))
                    offset += len(part)
        else:
            out_list.append(se_matst)
    return out_list


def find_itemized_paren_mats(line: str) -> List[Match[str]]:
    result = list(re.finditer(r'\(?\s*([\divx]+|[a-z])\s*\)\s*', line, re.I))
    return result


def find_non_space_index(line: str) -> int:
    """Return index of the first non-space character in line.

    This handles non-breaking spaces because of isspace()
    """
    if not line:
        return -1
    idx = 0
    line_len = len(line)
    while line[idx].isspace():
        idx += 1
        if idx >= line_len:
            return -1
    return idx


if __name__ == '__main__':
    print(str(_get_num_prefix_space("   abc")))   # 3
    print(str(_get_num_prefix_space("abc")))      # 0
    print(str(_get_num_prefix_space(" abc")))     # 1
    print(str(_get_num_prefix_space("\n\nabc")))  # 2
