#!/usr/bin/env python

import json
import logging
import os
import re

# pylint: disable=W0703, E1101

def remove_nltab(xst):
    # return re.sub(r'[\s\t\r\n]+', ' ', xst)
    return re.sub(r'[\s\t\r\n]', ' ', xst)


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


def save_str_list(str_list, file_name):
    with open(file_name, 'wt') as fout:
        for xst in str_list:
            fout.write(xst)
            fout.write(os.linesep)


def load_json_list(file_name):
    atext = loads(file_name)
    return json.loads(atext)

# This function is insufficient for Stanford CoreNLP.
# CoreNLP doesn't treate non-breaking space
# (https://en.wikipedia.org/wiki/Non-breaking_space) as space.
# Going to match first word instead to align CoreNLP output.
# st[i] == u'\u00a0' (non-breaking space)
def _get_num_prefix_space(xst):
    i = 0
    st_size = len(xst)
    while i < st_size:
        if xst[i].isspace():
            i += 1
        else:
            break
    return i

# http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float-in-python
def is_number(xst):
    try:
        float(xst)
        return True
    except ValueError:
        return False


def is_int(xst):
    try:
        int(xst)
        return True
    except ValueError:
        return False


def to_int(xst):
    try:
        result = int(xst)
        return result
    except ValueError:
        raise ValueError("Error in to_int({})".format(xst))

def is_all_digit_dot(xst):
    if not xst:
        return False
    for xch in xst:
        if not ((xch == '.') or (xch == '-') or xch.isdigit()):
            return False
    return True


def is_all_dash_underline(xst):
    if not xst:
        return False
    for xch in xst:
        if not ((xch == '-') or (xch == '_')):
            return False
    return True


def bool_to_int(bxx):
    if bxx:
        return 1
    return 0


def yesno_to_int(xst):
    if xst == 'yes':
        return 1
    elif xst == 'no':
        return 0
    raise ValueError("Error in yesno_to_int({})".format(xst))

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


def gen_ngram(word_list, max_n=2):
    result = []
    for i in range(len(word_list) - max_n + 1):
        ngram_words = [word_list[i + j] for j in range(max_n)]
        result.append(' '.join(ngram_words))
    return result


def tokens_to_all_ngrams(word_list, max_n=1):
    # unigram
    sent_wordset = set(word_list)
    # bigram and up
    for ngram_size in range(2, max_n+1):
        ngram_list = gen_ngram(word_list, max_n=ngram_size)
        sent_wordset |= set(ngram_list)

    return sent_wordset


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
    return re.sub(IGNORABLE_JSON_CTRL_PAT, ' ', line)

if __name__ == '__main__':
    print(str(_get_num_prefix_space("   abc")))   # 3
    print(str(_get_num_prefix_space("abc")))      # 0
    print(str(_get_num_prefix_space(" abc")))     # 1
    print(str(_get_num_prefix_space("\n\nabc")))  # 2
