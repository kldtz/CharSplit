#!/usr/bin/env python

import json
import logging
import os
import re
from typing import List, Set, Dict, Tuple
import unicodedata
import urllib

# this include punctuations
CAP_SPACE_PAT = re.compile(r'^[A-Z\s\(\).,\[\]\-/\\{\}`\'"]+$')

# pylint: disable=W0703, E1101

def remove_nltab(line):
    # return re.sub(r'[\s\t\r\n]+', ' ', line)
    return re.sub(r'[\s\t\r\n]', ' ', line)

def loads(file_name):
    xst = ''
    try:
        with open(file_name, 'rt', newline='') as myfile:
            xst = myfile.read()
    except IOError as exc:
        logging.error("I/O error(%s) in strutils.loads(%s): %s",
                      file_name, str(exc.errno), str(exc.strerror))
    except Exception as exc:  # handle other exceptions such as attribute errors
        # handle any other exception
        logging.error("Error '%s' occured. Arguments %s.", exc.message, str(exc.args))
    return xst

def load_lines_with_offsets(file_name: str) -> List[str]:
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


def dumps(xst, file_name):
    try:
        with open(file_name, 'wt') as myfile:
            myfile.write(xst)
            myfile.write(os.linesep)
    except IOError as exc:
        logging.error("I/O error(%s) in strutils.loads(%s): %s",
                      file_name, str(exc.errno), str(exc.strerror))
    # pylint: disable=W0703
    except Exception as exc:  # handle other exceptions such as attribute errors
        # handle any other exception
        logging.error("Error '%s' occured. Arguments %s.", exc.message, str(exc.args))


def load_str_list(file_name):
    st_list = []
    with open(file_name, 'rt', newline='') as fin:
        for line in fin:
            st_list.append(line.strip())
    return st_list


def load_lc_str_set(filename):
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

def load_json_list(file_name: str) -> List:
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

# http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float-in-python
def is_number(line: str) -> bool:
    try:
        float(line)
        return True
    except ValueError:
        return False

# handle "5.5." as a number at the end of a sentence
"""
def is_relaxed_number(line: str) -> bool:
    if line.endswith('.'):
        return is_number(line[:-1])
    return is_number(line)
    """

# SECHEAD_NUMBER_PAT = re.compile('^\d[\d\.]+$')

NUM_ROMAN_PAT = re.compile(r'(([\(\“\”]?([\.\d]+|[ivx\d\.]+\b)[\)\“\”]?|[\(\“\”\.]?[a-z][\s\.\)\“\”])+)')

def is_header_number(line: str) -> bool:
    return NUM_ROMAN_PAT.match(line)
# return SECHEAD_NUMBER_PAT.match(line)


ROMAN_NUM_PAT = re.compile(r'^[ixv]+$', re.IGNORECASE)

def is_roman_number(line: str) -> bool:
    return ROMAN_NUM_PAT.match(line)


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

ALL_DIGITS_PAT = re.compile(r'^\d+$')
ALL_ALPHAS_PAT = re.compile(r'^[a-zA-Z]+$')
ALL_ALPHAS_DOT_PAT = re.compile(r'^[a-zA-Z]+\.?$')

ALL_ALPHA_NUM_PAT = re.compile(r'^[a-zA-Z\d]+$')
ANY_DIGIT_PAT = re.compile(r'\d')
ANY_ALPHA_PAT = re.compile(r'[a-zA-Z]')


PARENS_ALL_DIGITS_PAT = re.compile(r'^\(\d+\)$')

def is_all_digits(line: str) -> bool:
    mat = ALL_DIGITS_PAT.match(line)
    return mat

def is_parens_all_digits(line: str) -> bool:
    mat = PARENS_ALL_DIGITS_PAT.match(line)
    return mat

def is_all_digit_dot(line: str) -> bool:
    if not line:
        return False
    for xch in line:
        if not ((xch == '.') or (xch == '-') or xch.isdigit()):
            return False
    return True

def is_all_length_1_words(words):
    if not words:
        return False
    for word in words:
        if len(word) != 1:
            return False
    return True

WORD_LC_PAT = re.compile('^[a-z]+$')

def is_word_all_lc(word):
    mat = WORD_LC_PAT.match(word)
    return mat

# has more than 2 digits that's more than 3 width
TWO_GT_3_NUM_SEQ_PAT = re.compile(r'\d{3}.*\d{3}')
def has_likely_phone_number(line):
    return TWO_GT_3_NUM_SEQ_PAT.search(line)
    
def is_all_alphas(line: str) -> bool:
    return is_alpha_word(line)

def is_all_alphas_dot(line: str) -> bool:
    return ALL_ALPHAS_DOT_PAT.match(line)

def is_alpha_word(line: str) -> bool:
    mat = ALL_ALPHAS_PAT.match(line)
    return mat

def has_digit(line: str) -> bool:
    return ANY_DIGIT_PAT.search(line)

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
            num_dollar + 1
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
    return CAP_SPACE_PAT.match(line)

def is_all_lower(line: str) -> bool:
    words = line.split()
    if not words:
        return False
    for word in words:
        if not word[0].islower():
            return False
    return True


def bool_to_int(bxx: bool) -> int:
    if bxx:
        return 1
    return 0


def yesno_to_int(line: str) -> int:
    if line == 'yes':
        return 1
    elif line == 'no':
        return 0
    raise ValueError("Error in yesno_to_int({})".format(line))

# pylint: disable=W0105
"""
def str_to_boolint_not_used(st):
    if isinstance(st, str):
        if st == 'yes':
            return 1
        elif st == 'no':
            return 0
        return to_int(st)
    elif isinstance(st, int):
        return st
    else:
        raise ValueError("str is not 'yes' or 'no' in str_to_booint({})".format(st))
"""


def gen_ngram(word_list: List[str], max_n=2) -> List[str]:
    result = []
    for i in range(len(word_list) - max_n + 1):
        ngram_words = [word_list[i + j] for j in range(max_n)]
        result.append(' '.join(ngram_words))
    return result



ALPHA_WORD_PAT = re.compile(r'[a-zA-Z]+')

ALPHANUM_WORD_PAT = re.compile(r'[a-zA-Z0-9]+')

ALL_PUNCT_PAT = re.compile(r"^[\(\)\.,\[\]\-/\\\{\}`'\"]+$")

ANY_PUNCT_PAT = re.compile(r"[\(\)\.,\[\]\-/\\\{\}`'\"]")

NUM_PERC_PAT = re.compile(r'^\s*\d+%\s*$')
NUM_PERIOD_PAT = re.compile(r'^\s*\d+\.\s*$')
DOLLAR_NUM_PAT = re.compile(r'^\s*\$\s*\d[\.\d]*\s*$')

def has_punct(line: str) -> bool:
    return ANY_PUNCT_PAT.search(line)

def is_all_punct(line: str) -> bool:
    return ALL_PUNCT_PAT.match(line)

def is_num_perc(line: str) -> bool:
    return NUM_PERC_PAT.match(line)

def is_num_period(line: str) -> bool:
    return NUM_PERIOD_PAT.match(line)

def is_dollar_num(line: str) -> bool:
    return DOLLAR_NUM_PAT.match(line)


def get_alpha_words_gt_len1(line: str, is_lower=True) -> List[str]:
    if is_lower:
        line = line.lower()
    return [word for word in ALPHA_WORD_PAT.findall(line) if len(word) > 1]


def get_alpha_words(line: str, is_lower=True) -> List[str]:
    if is_lower:
        line = line.lower()
    return [word for word in ALPHA_WORD_PAT.findall(line)]


def get_alphanum_words_gt_len1(line: str, is_lower=True) -> List[str]:
    if is_lower:
        line = line.lower()
    return [word for word in ALPHANUM_WORD_PAT.findall(line) if len(word) > 1]


def get_alphanum_words(line: str, is_lower=True) -> List[str]:
    if is_lower:
        line = line.lower()
    return [word for word in ALPHANUM_WORD_PAT.findall(line)]


def tokens_to_all_ngrams(word_list: List[str], max_n=1) -> Set[str]:
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
    else:
        return False

def is_sent_punct(line: str) -> bool:
    return line and len(line) == 1 and line in r'.?!'

def is_not_sent_punct(line: str) -> bool:
    return line and is_punct(line) and not is_sent_punct(line)

def is_punct_not_period(line: str) -> bool:
    if line:
        return line[0] != '.' and is_punct(line)
    else:
        return False
    
def is_punct_notwork(line: str) -> bool:
    if line:
        return is_punct_core(line)
    else:
        return False

# '_' is for 'Title: ____________'.  It is end of sentence char
def is_eosent(line: str) -> bool:
    if line:
        return line[0] in '.!?_'
    else:
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

def is_all_title_words(line: str) -> bool:
    words = split_words(line)
    has_alpha_word = False
    for word in words:
        if is_all_alphas(word):
            has_alpha_word = True
            if not word.isupper():
                return False
    return is_alpha_word

def is_all_title(words: List[str]) -> bool:
    has_alpha_word = False
    for word in words:
        if is_all_alphas(word):
            has_alpha_word = True
            if not word.isupper():
                return False
    return is_alpha_word
    
ANY_ALPHA_PAT = re.compile(r'[a-z]', re.I)

def has_alpha(line: str):
    return ANY_ALPHA_PAT.search(line)


DIGIT_PAT = re.compile(r'^\d+$')
def is_digit_st(line: str) -> bool:
    return DIGIT_PAT.match(line)

# to detect telephone or SSN
BIG_DASHED_DIGIT_PAT = re.compile(r'^(\d+)\-[\d\-]+$')
def is_dashed_big_number_st(line: str) -> bool:
    mat = BIG_DASHED_DIGIT_PAT.match(line)
    if mat:
        if int(mat.group(1)) > 20:
            return True
    return False

def extract_numbers(line: str):
    return re.findall(r'(\d*\.\d+|\d+\.\d*|\d+)', line)

def count_numbers(line: str):
    return len(extract_numbers(line))

NUM_10_PAT = re.compile(r'(\d*\.\d+|\d+\.\d*|\d+)')
def find_number(line: str):
    return NUM_10_PAT.search(line)

    
def is_digit_core(line: str) -> bool:
    return unicodedata.category(line) == 'Nd'

DASH_ONLY_LINE = re.compile(r'^\s*-+\s*$')

def is_dashed_line(line: str) -> bool:
    return DASH_ONLY_LINE.match(line)


# this will match any substr, not just words
def are_all_substrs_in_st(substr_list: List[str], line: str) -> bool:
    lc_text = line.lower()
    return all(substr in lc_text for substr in substr_list)

def is_english_vowel(unich: str) -> bool:
    return unich in 'aeiouAEIOU'


def find_substr_indices(sub_strx, text):
    result = []
    for m in re.finditer(sub_strx, text):
        result.append((m.start(), m.end()))
    return result

def count_date(line):
    return len(find_substr_indices(r'(\d{1,2}/\d{1,2}/(20|19)\d\d)', line))

def count_number(line):
    return len(find_substr_indices(r'(\d+)', line))
    # return len(find_substr_indices(r'(\d*\.\d+|\d+\.\d*|\d+)', line))


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
    # TODO, in future, regex these
    line = line.replace('\u201c', '"')
    line = line.replace('\u201d', '"')
    line = line.replace('\u2019', "'")
    return line


def corenlp_normalize_text(doc_text):
    line = replace_ignorable_json_ctrl_chars(doc_text)
    line = urllib.parse.quote(line)
    return line

def is_space_or_nl(xch):
    return (xch == ' ' or
            xch == '\n' or
            xch == '\r')

def is_nl(xch):
    return xch == '\n' or xch == '\r'


def is_double_quote(xch):
    return xch in '“"”'


def dict_to_sorted_list(adict):
    return ['{}={}'.format(attr, value) for attr, value in sorted(adict.items())]


def space_repl_same_length(mat):
    return ' ' * len(mat.group())

def replace_dot3plus_with_spaces(line: str) -> str:
    return re.sub(r'([\.\-_][\.\-_][\.\-_]+)', space_repl_same_length, line)


NEXT_TOKEN_PAT = re.compile(r'\s*\S+')

def find_next_token(line: str):
    return NEXT_TOKEN_PAT.match(line)


if __name__ == '__main__':
    print(str(_get_num_prefix_space("   abc")))   # 3
    print(str(_get_num_prefix_space("abc")))      # 0
    print(str(_get_num_prefix_space(" abc")))     # 1
    print(str(_get_num_prefix_space("\n\nabc")))  # 2
