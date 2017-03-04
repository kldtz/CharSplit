#!/usr/bin/env python

import logging
import math
import sys
import time
from collections import defaultdict

from kirke.utils import stopwordutils, strutils

# we tried 10%, but this seems to be safer to avoid
# sudden increase in vocab size when corpus size
# increases drastically
DEFAULT_INFOGAIN_VOCAB_SIZE = 100000

def entropy_by_freq_list(freq_list):
    total = 0
    for word, freq in freq_list.items():
        total += freq
    result = 0.0
    for val, count in freq_list.items():
        prob = count / total
        # avoid prob = 0, in which math.log throw exception
        if count != 0:
            result += prob * math.log(prob, 2)
        # it's possible that there is no instance for True
        #else:
        #    print("count = 0, val = {}".format(val))
    return -result

def to_cond_count_map(label_count_map, cond_dist):
    cond_count_map = defaultdict(lambda: defaultdict(int))
    for cond, cond_count in cond_dist.items():
        label_count = label_count_map.get(cond, 0)
        if label_count != 0:
            cond_count_map[cond][True] = label_count
        cond_count_map[cond][False] = cond_count - label_count
    return cond_count_map


def info_gain(cond_count_map, entropy_class):
    col_entropy = column_entropy(cond_count_map)
    return entropy_class - col_entropy


def column_entropy(cond_count_map):
    val_count_map = defaultdict(int)
    val_class_freq_map = defaultdict(lambda: defaultdict(int))
    total = 0
    for class_label, word_exists_freq_map  in cond_count_map.items():
        for word_exists, freq in word_exists_freq_map.items():
            val_count_map[word_exists] += freq
            # print("val_count_map[{}] = {}".format(word_exists, freq))
            val_class_freq_map[word_exists][class_label] = freq
            total += freq
    # print("val_class_freq_map = {}".format(val_class_freq_map))

    result = 0.0
    for k, v in val_count_map.items():
        ratio = v / total
        # print("v = {}, total= {}".format(v, total))
        # print("val_class_freq_map[{}] = {}".format(k, val_class_freq_map[k]))
        entropy_i = entropy_by_freq_list(val_class_freq_map[k])
        # print("%s ratio = %.2f" % (k, v / total_map[label]))
        # print(" entropy = %.2f" % entropy_i)
        result += ratio * entropy_i
    return result


def eb_doc_to_all_ngrams(sent_st):
    ngram_size = 3
    # keep case
    orig_word_list = stopwordutils.str_to_tokens(sent_st, mode=1, is_remove_punct=True)
    sent_wordset = strutils.tokens_to_all_ngrams(orig_word_list, n=ngram_size)

    # add lower case version
    lc_word_list = stopwordutils.str_to_tokens(sent_st, mode=0, is_remove_punct=True)
    sent_wordset |= strutils.tokens_to_all_ngrams(lc_word_list, n=ngram_size)

    nostop_orig_word_list = stopwordutils.tokens_remove_stopwords(orig_word_list)
    sent_wordset |= strutils.tokens_to_all_ngrams(nostop_orig_word_list, n=ngram_size)

    nostop_lc_word_list = stopwordutils.tokens_remove_stopwords(lc_word_list)
    sent_wordset |= strutils.tokens_to_all_ngrams(nostop_lc_word_list, n=ngram_size)

    return sent_wordset


# doc is a st
# tokenize is the tokenizing function
def doc_label_list_to_vocab(doc_list, label_list, tokenize, debug_mode=False):
    sent_wordset_list = []
    word_label_count_map = defaultdict(lambda: defaultdict(int))
    cond_dist = defaultdict(int)
    vocabs = set([])
    
    for doc_st, label_val in zip(doc_list, label_list):
        doc_tokens = tokenize(doc_st)
        cond_dist[label_val] += 1

        vocabs |= doc_tokens
        for word in doc_tokens:
            word_label_count_map[word][label_val] += 1

    # print("word_docids_map[Change] = {}".format(word_docids_map['Change']))
    logging.debug("igain.vocab size = {}".format(len(vocabs)))
                
    # vocab, cond_dist, word_label_count_map

    entropy_of_class = entropy_by_freq_list(cond_dist)
    # logging.info('cond dist = {}'.format(cond_dist))
    # logging.info('class entropy = {:.3f}'.format(entropy_of_class))

    # print('vocabs = ' + str(vocabs))
    #for col_name in vocabs:
    #    cond_freq_dist = to_cond_freq_dist(col_name, word_docids_map[col_name], label_list)
    #    ent = igain.column_entropy(col_name, cond_freq_dist)
    #    # print('entropy(%s) = %.3f' % (col_name, ent))

    result = []
    start_time = time.time()
    orig_start_time = start_time
    word_cond_freq_dist_map = {}
    for i, col_name in enumerate(vocabs):
        if i % 100000 == 0:
            now_time = time.time()
            partial_diff = now_time - start_time
            total_diff = now_time - orig_start_time
            logging.debug("i = {}, took {:.4f} seconds, total = {:.4f} seconds".format(i,
                                                                                      partial_diff,
                                                                                      total_diff,
                                                                                      file=sys.stderr))
            start_time= time.time()
        cond_count_map = to_cond_count_map(word_label_count_map[col_name], cond_dist)
        # "{}".format(cond_count_map)
        word_cond_freq_dist_map[col_name] = cond_count_map
                                                                             
        ig = info_gain(cond_count_map, entropy_of_class)
        # print('info_gain(%s) = %.3f' % (col_name, ig))
        result.append((ig, col_name))

    # take top 5% of all the vocab
    # wanted_vocab_size = len(vocabs) * 0.10
    wanted_vocab_size = DEFAULT_INFOGAIN_VOCAB_SIZE
    if len(vocabs) < DEFAULT_INFOGAIN_VOCAB_SIZE:
        wanted_vocab_size = len(vocabs)
    top_ig_ngram_list = []
    # i = 0
    if debug_mode:    
        for ig, word in sorted(result, reverse=True):
            cond_count_map = word_cond_freq_dist_map[word]
            #print("cond_count_map = {}".format(cond_count_map))
            #if i > 200000:
            #    break
            #i += 1
            #if ((cond_count_map['True'].get(True, 0) == 1 and cond_count_map['False'].get(True, 0) == 0) or
            #    (cond_count_map[True].get(True, 0) == 1 and cond_count_map[False].get(True, 0) == 0)):
            #    break


            cond_count_map_st = "YES-word={}, NO-word={}, YES-xx-word= {}, NO-xx-word={}".format(cond_count_map['True'].get(True, 0),
                                                                                                 cond_count_map['False'].get(True, 0),
                                                                                                 cond_count_map['True'].get(False, 0),
                                                                                                 cond_count_map['False'].get(False, 0))                                                                             
            print(word, ig, cond_count_map_st, sep='\t')

    top_ig_ngram_list = [word for ig, word in sorted(result, reverse=True)][:wanted_vocab_size]

    return top_ig_ngram_list


def read_eb_sents_to_word_label_count_map(file_name):
    cond_dist = defaultdict(int)
    vocab = set([])
    label = 'class'

    # sent_list = []
    label_list = []
    sent_wordset_list = []
    word_label_count_map = defaultdict(lambda: defaultdict(int))
    
    # first get all the vocab
    with open(file_name, 'rt') as csvfile:
        fieldnames = ['class', 'sent']
        reader = csv.DictReader(csvfile, fieldnames=fieldnames, delimiter='\t')

        docid = 0
        for row in reader:
            label_val = row[label]
            cond_dist[label_val] += 1
            label_list.append(label_val)

            # print("label_val= {}".format(label_val))
            # print("sent= {}".format(row['sent']))
            sent_st = row['sent']
            # sent_list.append(sent_st)

            # keep case
            orig_word_list = stopwordutils.str_to_tokens(sent_st, mode=1, is_remove_punct=True)
            sent_wordset = strutils.tokens_to_all_ngrams(orig_word_list)
            
            # add lower case version
            lc_word_list = stopwordutils.str_to_tokens(sent_st, mode=0, is_remove_punct=True)
            sent_wordset |= strutils.tokens_to_all_ngrams(lc_word_list)

            nostop_orig_word_list = stopwordutils.tokens_remove_stopwords(orig_word_list)
            sent_wordset |= strutils.tokens_to_all_ngrams(nostop_orig_word_list)

            nostop_lc_word_list = stopwordutils.tokens_remove_stopwords(lc_word_list)
            sent_wordset |= strutils.tokens_to_all_ngrams(nostop_lc_word_list)
            
            for word in sent_wordset:
                vocab.add(word)
                word_label_count_map[word][label_val] += 1

            docid += 1
    # print("word_docids_map[Change] = {}".format(word_docids_map['Change']))
    # print("vocab_size = {}".format(len(vocab)))
                
    return vocab, cond_dist, word_label_count_map

def read_csv_to_word_label_count_map(file_name, label_name='class'):
    cond_dist = defaultdict(int)
    word_label_count_map = defaultdict(lambda: defaultdict(int))
    with open(file_name, 'rt') as csvfile:
        reader = csv.DictReader(csvfile)

        fieldnames = list(reader.fieldnames)
        fieldnames.remove(label_name)
        for row in reader:
            label_val = row[label_name]
            cond_dist[label_val] += 1            
            for word in fieldnames:
                if row[word] == 'true':
                    word_label_count_map[word][label_val] += 1
    return fieldnames, cond_dist, word_label_count_map


