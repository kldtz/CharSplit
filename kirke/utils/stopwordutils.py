#!/usr/bin/env python

from kirke.utils import strutils


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


def is_stopword(word):
    return word in STOPWORDS


def is_punctword(word):
    return word in PUNCTWORDS


def is_punct_or_stopword(word):
    return word in PUNCT_STOPWORDS


# pylint: disable=R0911
def is_eb_stopword(word):
    if not word:  # empty
        return True
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


def tokens_remove_stopwords(token_list):
    return [token for token in token_list if not is_eb_stopword(token)]


def tokens_remove_punct(token_list):
    return [token for token in token_list if not is_punctword(token)]


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