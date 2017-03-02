#!/usr/bin/env python

import sys
import logging
import json
import os
import re

def remove_nltab(st):
    # return re.sub(r'[\s\t\r\n]+', ' ', st)
    return re.sub(r'[\s\t\r\n]', ' ', st)
    
def loads(file_name):
    st = ''
    try:
        with open(file_name, 'rt') as myfile:
            st = myfile.read()
    except IOError as e:
        logging.error("I/O error({}) in strutils.loads({}): {}".format(file_name, e.errno, e.strerror))
    except:   # handle other exceptions such as attribute errors
        logging.error("Unexpected error in strutils.loads({}): {}".format(file_name, sys.exc_info()[0]))
    return st


def dumps(st, file_name):
    try:
        with open(file_name, 'wt') as myfile:
            myfile.write(st)
            myfile.write(os.linesep)
    except IOError as e:
        logging.error("I/O error({}) in strutils.dumps({}): {}".format(file_name, e.errno, e.strerror))
    except:   # handle other exceptions such as attribute errors
        logging.error("Unexpected error in strutils.dumps({}): {}".format(file_name, sys.exc_info()[0]))


def save_str_list(str_list, file_name):
    with open(file_name, 'wt') as fout:
        for st in str_list:
            fout.write(st)
            fout.write(os.linesep)
            
    # with open("sents.pos.out.{}".format(time.strftime("%Y-%m-%d-%H-%M")), 'wt') as posout:
    #            for sent_words_st in positive_sent_words_st_list:
    #                posout.write(sent_words_st)
    #                posout.write(os.linesep)
        

def load_json_list(file_name):
    atext = loads(file_name)
    return json.loads(atext)

# This function is insufficient for Stanford CoreNLP.
# CoreNLP doesn't treate non-breaking space
# (https://en.wikipedia.org/wiki/Non-breaking_space) as space.
# Going to match first word instead to align CoreNLP output.
# st[i] == u'\u00a0' (non-breaking space)
def _get_num_prefix_space(st):
    i = 0
    st_size = len(st)
    while i < st_size:
        if st[i].isspace():
            i += 1
        else:
            break
    return i

# http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float-in-python
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def to_int(s):
    try:
        result = int(s)
        return result
    except ValueError:
        raise ValueError("Error in to_int({})".format(s))

def is_all_digit_dot(st):
    if not st:
        return False
    for ch in st:
        if not ((ch == '.') or (ch == '-') or ch.isdigit()):
            return False
    return True

def is_all_dash_underline(st):
    if not st:
        return False
    for ch in st:
        if not ((ch == '-') or (ch == '_')):
            return False
    return True

def bool_to_int(bx):
    if bx:
        return 1
    else:
        return 0
    
def yesno_to_int(st):
    if st == 'yes':
        return 1
    elif st == 'no':
        return 0
    raise ValueError("Error in yesno_to_int({})".format(st))

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


def gen_ngram(word_list, n=2):
    result = []
    for i in range(len(word_list)-n+1):
        ngram_words = [word_list[i+j] for j in range(n)]
        result.append(' '.join(ngram_words))
    return result


def tokens_to_all_ngrams(word_list, n=1):
    # unigram
    sent_wordset = set(word_list)
    # bigram and up
    for ngram_size in range(2, n+1):
        ngram_list = gen_ngram(word_list, n=ngram_size)
        sent_wordset |= set(ngram_list)

    return sent_wordset


if __name__ == '__main__':
    print(str(_get_num_prefix_space("   abc")))  # 3
    print(str(_get_num_prefix_space("abc")))     # 0
    print(str(_get_num_prefix_space(" abc")))    # 1
    print(str(_get_num_prefix_space("\n\nabc"))) # 2
