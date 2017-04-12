
from collections import defaultdict
import operator
import logging


from kirke.utils import stopwordutils, strutils

# NUMBER_OF_TOP_WORDS = 25000
NUMBER_OF_TOP_WORDS = 50000

def eb_doc_to_all_ngrams(sent_st):
    ngram_size = 2

    # keep case
    orig_word_list = stopwordutils.str_to_tokens(sent_st, mode=1, is_remove_punct=True)
    # sent_wordset = strutils.tokens_to_all_ngrams(orig_word_list, max_n=ngram_size)

    # add lower case version
    # lc_word_list = stopwordutils.str_to_tokens(sent_st, mode=0, is_remove_punct=True)
    # sent_wordset |= strutils.tokens_to_all_ngrams(lc_word_list, max_n=ngram_size)

    nostop_orig_word_list = stopwordutils.tokens_remove_stopwords(orig_word_list)
    sent_wordset = strutils.tokens_to_all_ngrams(nostop_orig_word_list, max_n=ngram_size)

    # nostop_lc_word_list = stopwordutils.tokens_remove_stopwords(lc_word_list)
    # sent_wordset |= strutils.tokens_to_all_ngrams(nostop_lc_word_list, max_n=ngram_size)

    return sent_wordset


def doc_label_list_to_vocab(doc_list, label_list, tokenize, debug_mode=False):
    word_freq_map = defaultdict(int)
    # TODO, remove, for debug only
    positive_st_count = 0
    debug_mode = False
    pos_word_freq_map = defaultdict(int)
    for doc_st, label_tf in zip(doc_list, label_list):
        doc_tokens = tokenize(doc_st)
        if label_tf:
            positive_st_count += 1
        if debug_mode and label_tf:
            print("\npos doc_st #{}: [[{}]]".format(positive_st_count, doc_st))
            print("  ngrams: {}".format(doc_tokens))
        for word in doc_tokens:
            word_freq_map[word] += 1
            if label_tf:
                pos_word_freq_map[word] += 1

    logging.debug("len(word_freq_map) = {}".format(len(word_freq_map)))
    logging.debug("len(pos_word_freq_map) = {}".format(len(pos_word_freq_map)))
    logging.debug("positive_sent_count = {}".format(positive_st_count))

    vocabs = set([])
    positive_vocabs = set([])
    word_count = 0

    for word, freq in sorted(pos_word_freq_map.items(), key=operator.itemgetter(1), reverse=True):
        word_count += 1
        if freq < 5:
            break
        vocabs.add(word)
        positive_vocabs.add(word)
    logging.debug("len(positive vocab) = {}".format(len(vocabs)))

    for word, freq in sorted(word_freq_map.items(), key=operator.itemgetter(1), reverse=True):
        if word_count > NUMBER_OF_TOP_WORDS:
            logging.debug("skipping word with freq less than {}".format(freq))
            break
        if word not in vocabs:
            word_count += 1
            # print('adding vocab: [{}], freq= {}'.format(word, freq))
            vocabs.add(word)

    return vocabs, positive_vocabs

    
