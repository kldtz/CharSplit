#!/usr/bin/env python3

from kirke.utils import strutils
from typing import List

def get_sklearn_stopwords():
    file_name = 'dict/sklearn_stopwords.txt'
    stop_words = []
    with open(file_name, 'rt') as fin:
        for word in fin:
            stop_words.append(word.strip())
    return frozenset(stop_words)


def get_corenlp_puncts():
    # punctuation after corenlp's tokenization
    file_name = 'dict/eb_corenlp_punct.txt'
    punct_words = []
    with open(file_name, 'rt') as fin:
        for word in fin:
            punct_words.append(word.strip())
    return frozenset(punct_words)


STOPWORDS = get_sklearn_stopwords()
PUNCTWORDS = get_corenlp_puncts()
PUNCT_STOPWORDS = STOPWORDS.union(PUNCTWORDS)


def get_nonstopwords_gt_len1(line: str) -> List[str]:
    words_gt_len1 = strutils.get_alphanum_words_gt_len1(line)
    return [word for word in words_gt_len1 if word not in STOPWORDS]


def get_nonstopwords_nolc_gt_len1(line: str) -> List[str]:
    words_gt_len1 = strutils.get_alphanum_words_gt_len1(line, is_lower=False)
    return [word for word in words_gt_len1 if word.lower() not in STOPWORDS]


def is_stopword(word):
    return word in STOPWORDS


def is_punctword(word):
    return word in PUNCTWORDS


def is_punct_or_stopword(word):
    return word in PUNCT_STOPWORDS

def count_stopword(word_list):
    count = 0
    for word in word_list:
        if word in STOPWORDS:
            count += 1
    return count


# pylint: disable=R0911
def is_eb_stopword(word, is_lower=False):
    if not word:  # empty
        return True

    if is_lower:  # lower if needed
        word = word.lower()
        
    if word in PUNCT_STOPWORDS:
        return True
    if len(word) < 2:  # one char words are ambiguous
        return True
    if len(word) == 2 and word[1] == '.':
        return True
    if strutils.is_all_digit_dot(word):
        return True
    if strutils.is_all_dash_underline(word):
        return True
    return False


# mode = {lower, keep, keep_both}, 0, 1, 2
def str_to_tokens(sent, mode=0, is_remove_punct=False):
    sent_tokens = []
    for word in sent.split():
        if is_remove_punct and is_punctword(word):
            continue
        if mode == 0:  # lower
            word = word.lower()
        # elif mode == 1:
        #    pass
        sent_tokens.append(word)
        if mode == 2:  # keep_both
            lcword = word.lower()
            if lcword != word:
                sent_tokens.append(lcword)
    return sent_tokens


# mode = {lower, keep, keep_both}, 0, 1, 2
def get_sent_tokens_list(sent_list, mode=0):
    sent_tokens_list = []
    for sent in sent_list:
        sent_tokens_list.append(str_to_tokens(sent, mode=mode))
    return sent_tokens_list


def tokens_remove_stopwords(token_list, is_lower=False):
    return [token for token in token_list if not is_eb_stopword(token, is_lower)]


def tokens_remove_punct(token_list):
    return [token for token in token_list if not is_punctword(token)]

def is_title_non_stopwords(word_list: List[str]) -> bool:
    if not word_list:  # no word, return False
        return False
    for word in tokens_remove_stopwords(word_list, is_lower=True):
        if not word[0].isupper():
            return False
    return True

def is_line_title_non_stopwords(line: str) -> bool:
    words = str_to_tokens(line, mode=1)
    print("words = {}".format(words))
    return is_title_non_stopwords(words)



# mode = {lower, keep, keep_both}, 0, 1, 2
def str_remove_stopwords(sent, mode=0):
    nostop_token_list = []
    token_list = str_to_tokens(sent, mode=mode)
    nostop_token_list = tokens_remove_stopwords(token_list)
    return ' '.join(nostop_token_list)


def str_lc_and_keep_case(sent):
    # mode = {lower, keep, keep_both}, 0, 1, 2
    return ' '.join(str_to_tokens(sent, mode=2))


# mode = {lower, keep, keep_both}, 0, 1, 2
def remove_stopwords(sent_list, mode=0):
    return [str_remove_stopwords(sent, mode=mode).strip() for sent in sent_list]


# pylint: disable=W0105
"""
# mode = {lower, keep, keep_both}, 0, 1, 2
def not_used_remove_stopwords_and_count(sent_list, mode=0):
    word_count_map = defaultdict(int)
    sent_tokens_list = get_sent_tokens_list(sent_list, mode)
    nostop_sent_st_list = []
    for sent_tokens in sent_tokens_list:
        sent_nostop_tokens = tokens_remove_stopwords(sent_tokens)
        for word in sent_nostop_tokens:
            word_count_map[word] += 1
        nostop_sent_st_list.append(' '.join(sent_nostop_tokens))
    return nostop_sent_st_list, word_count_map
"""
