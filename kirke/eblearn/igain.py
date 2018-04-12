#!/usr/bin/env python3

from collections import defaultdict, namedtuple
import csv
import logging
import math
import operator
import sys
import time

from kirke.utils import stopwordutils, strutils

# we tried 10%, but this seems to be safer to avoid
# sudden increase in vocab size when corpus size
# increases drastically
# DEFAULT_INFOGAIN_VOCAB_SIZE = 400000
# DEFAULT_INFOGAIN_VOCAB_SIZE = 100000
# DEFAULT_INFOGAIN_VOCAB_SIZE = 50000
DEFAULT_INFOGAIN_VOCAB_SIZE = 25000

def entropy_by_freq_list(freq_list):
    # print("freq_list = {}".format(freq_list))
    total = 0
    for _, freq in freq_list.items():
        total += freq
    result = 0.0
    for _, count in freq_list.items():
        prob = count / total
        # avoid prob = 0, in which math.log throw exception
        if count != 0:
            result += prob * math.log(prob, 2)
        # it's possible that there is no instance for True
        #else:
        #    print("count = 0, val = {}".format(val))
    return -result

def entropy_by_freq_list_flat(freq_list):
    # print("flat freq_list = {}".format(freq_list))
    total = 0
    for freq in freq_list:
        total += freq
    result = 0.0
    for count in freq_list:
        prob = count / total
        # avoid prob = 0, in which math.log throw exception
        if count != 0:
            result += prob * math.log(prob, 2)
        # it's possible that there is no instance for True
        #else:
        #    print("count = 0, val = {}".format(val))
    return -result

CondCountTuple = namedtuple('CondCountTuple', ['condTTrue', 'condTFalse',
                                               'condFTrue', 'condFFalse'])

def to_cond_count_map(true_count_map,
                      false_count_map,
                      cond_dist):
    cond_count_map = defaultdict(lambda: defaultdict(int))
    # print("\ntrue_count_map = {}".format(true_count_map))
    # print("false_count_map = {}".format(false_count_map))
    # print("cond_dist = {}".format(cond_dist))
    for cond, cond_count in cond_dist.items():
        # print("cond=[{}], count= {}".format(cond, cond_count))
        if cond:
            label_count = true_count_map
        else:
            label_count = false_count_map
        if label_count != 0:
            cond_count_map[cond][True] = label_count
        cond_count_map[cond][False] = cond_count - label_count
    # print("cond_count_map = {}".format(cond_count_map))
    #result = CondCountTuple(cond_count_map[True].get(True, 0),
    #                        cond_count_map[True].get(False, 0),
    #                        cond_count_map[False].get(True, 0),
    #                        cond_count_map[False].get(False, 0))
    # print("resut = {}".format(result))
    return cond_count_map


def to_cond_count_tuple(true_count_map,
                        false_count_map,
                        cond_dist):
    try:
        cond_count_map = {}
        # print("\ntrue_count_map = {}".format(true_count_map))
        # print("false_count_map = {}".format(false_count_map))
        # print("cond_dist = {}".format(cond_dist))
        for cond, cond_count in cond_dist.items():
            # print("cond=[{}], count= {}".format(cond, cond_count))
            if cond:
                label_count = true_count_map
            else:
                label_count = false_count_map

            if label_count != 0:
                cond_count_map[cond] = (label_count, cond_count - label_count)
            else:
                cond_count_map[cond] = (0, cond_count - label_count)
        # print("cond_count_map = {}".format(cond_count_map))
        result = CondCountTuple(cond_count_map[True][0],
                                cond_count_map[True][1],
                                cond_count_map[False][0],
                                cond_count_map[False][1])
        # print("resut = {}".format(result))
        return result
    except KeyError:
        unused_error_type, error_instance, traceback  = sys.exc_info()
        error_instance.user_message = 'No positive examples found during training.'
        error_instance.__traceback__ = traceback
        raise error_instance


def info_gain(cond_count_map, entropy_class):
    col_entropy = column_entropy(cond_count_map)
    return entropy_class - col_entropy

def info_gain_flat(cond_count_map, entropy_class):
    col_entropy = column_entropy_flat(cond_count_map)
    return entropy_class - col_entropy

def column_entropy_flat(cond_count_map):
    # print("float count_count_map = {}".format(cond_count_map))
    # val_class_freq_map[word_exists][class_label] = freq
    val_class_freq_map_wordt_classt = cond_count_map.condTTrue
    val_class_freq_map_wordt_classf = cond_count_map.condTFalse
    val_class_freq_map_wordf_classt = cond_count_map.condFTrue
    val_class_freq_map_wordf_classf = cond_count_map.condFFalse

    val_count_map_wordt = cond_count_map.condTTrue + cond_count_map.condFTrue
    val_count_map_wordf = cond_count_map.condTFalse + cond_count_map.condFFalse

    total = (cond_count_map.condTTrue + cond_count_map.condTFalse +
             cond_count_map.condFTrue + cond_count_map.condFFalse)
    # print('flat total = {}'.format(total))

    # wordt
    ratio = val_count_map_wordt / total
    entropy_i = entropy_by_freq_list_flat([val_class_freq_map_wordt_classt,
                                           val_class_freq_map_wordf_classt])
    result = ratio * entropy_i
    #print("flat %s ratio = %.2f" % (True, ratio))
    #print("flat entropy = %.2f" % entropy_i)

    # wordf
    ratio = val_count_map_wordf / total
    entropy_i = entropy_by_freq_list_flat([val_class_freq_map_wordt_classf,
                                           val_class_freq_map_wordf_classf])

    #print("flat2 %s ratio = %.2f" % (False, ratio))
    #print("flat2 entropy = %.2f" % entropy_i)
    result += ratio * entropy_i
    return result


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
    for k, count in val_count_map.items():
        ratio = count / total
        #print("yy total= {}".format(total))
        #print("yy val_class_freq_map[{}] = {}".format(k, val_class_freq_map[k]))
        entropy_i = entropy_by_freq_list(val_class_freq_map[k])
        #print("yy %s ratio = %.2f" % (k, ratio))
        #print("yy entropy = %.2f" % entropy_i)
        result += ratio * entropy_i
    return result


def eb_doc_to_all_ngrams(sent_st):
    ngram_size = 3
    # keep case
    orig_word_list = stopwordutils.str_to_tokens(sent_st, mode=1, is_remove_punct=True)
    sent_wordset = strutils.tokens_to_all_ngrams(orig_word_list, max_n=ngram_size)

    # add lower case version
    lc_word_list = stopwordutils.str_to_tokens(sent_st, mode=0, is_remove_punct=True)
    sent_wordset |= strutils.tokens_to_all_ngrams(lc_word_list, max_n=ngram_size)

    nostop_orig_word_list = stopwordutils.tokens_remove_stopwords(orig_word_list)
    sent_wordset |= strutils.tokens_to_all_ngrams(nostop_orig_word_list, max_n=ngram_size)

    nostop_lc_word_list = stopwordutils.tokens_remove_stopwords(lc_word_list)
    sent_wordset |= strutils.tokens_to_all_ngrams(nostop_lc_word_list, max_n=ngram_size)

    return sent_wordset


# doc is a st
# tokenize is the tokenizing function, such as eb_doc_to_all_ngrams() above
# pylint: disable=R0914
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def doc_label_list_to_vocab(doc_list, label_list, tokenize, debug_mode=False, provision='default'):

    word_freq_map = defaultdict(int)
    for doc_st, label_val in zip(doc_list, label_list):
        doc_tokens = tokenize(doc_st)
        for word in doc_tokens:
            word_freq_map[word] += 1
    logging.debug("len(word_freq_map) = %d", len(word_freq_map))

    vocabs = set([])
    word_count = 0
    vocab_size_times_10 = DEFAULT_INFOGAIN_VOCAB_SIZE * 15
    for word, freq in sorted(word_freq_map.items(), key=operator.itemgetter(1), reverse=True):
        word_count += 1
        if word_count > vocab_size_times_10:
            logging.debug("skipping word with freq less than %d", freq)
            break
        # print('adding vocab: [{}], freq= {}'.format(word, freq))
        vocabs.add(word)
    word_freq_map = None  # free that memory

    # sent_wordset_list = []
    word_true_count_map = defaultdict(int)
    word_false_count_map = defaultdict(int)
    cond_dist = defaultdict(int)

    for doc_st, label_val in zip(doc_list, label_list):
        doc_tokens = tokenize(doc_st)
        cond_dist[label_val] += 1
        # print('label_val = [{}]'.format(label_val))

        for word in doc_tokens:
            if word not in vocabs:  # skip word that doesn't occur enough
                continue
            if label_val:
                word_true_count_map[word] += 1
            else:
                word_false_count_map[word] += 1
            # word_freq_map[word] += 1

    # count number of freq
    #print("cond_dist = {}".format(cond_dist))  # False, True
    #wf_count_map = defaultdict(int)
    #for word, freq in word_freq_map.items():
    #    wf_count_map[freq] += 1

    #for freq, count in sorted(wf_count_map.items(), key=operator.itemgetter(1)):
    #    print("wf_count_map[{}] = {}".format(freq, count))

    # print("word_docids_map[Change] = {}".format(word_docids_map['Change']))
    logging.debug("igain.vocab size = %d", len(vocabs))

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
    for i, col_name in enumerate(vocabs, 1):
        if i % 100000 == 0:
            now_time = time.time()
            partial_diff = now_time - start_time
            total_diff = now_time - orig_start_time
            logging.debug("i = %d, took %.4f seconds, total = %.4f seconds",
                          i, partial_diff, total_diff)
            start_time = time.time()
        #cond_count_map = to_cond_count_map(word_true_count_map[col_name],
        #                                   word_false_count_map[col_name],
        #                                   cond_dist)

        cond_count_tuple = to_cond_count_tuple(word_true_count_map[col_name],
                                               word_false_count_map[col_name],
                                               cond_dist)

        word_cond_freq_dist_map[col_name] = cond_count_tuple

        #igain = info_gain(cond_count_map, entropy_of_class)
        igain = info_gain_flat(cond_count_tuple, entropy_of_class)
        #if igain != igain2:
        #    print('info_gain(%s) = %.5f, igain2 = %.5f\n' % (col_name, igain, igain2))
        result.append((igain, col_name))

    # take top 5% of all the vocab
    # wanted_vocab_size = len(vocabs) * 0.10
    wanted_vocab_size = DEFAULT_INFOGAIN_VOCAB_SIZE
    if len(vocabs) < DEFAULT_INFOGAIN_VOCAB_SIZE:
        wanted_vocab_size = len(vocabs)
    top_ig_ngram_list = []
    # i = 0
    if debug_mode:
        with open("/tmp/{}.igain.vocab.tsv".format(provision), 'wt') as fout:
            for igain, word in sorted(result, reverse=True):
                cond_count_map = word_cond_freq_dist_map[word]
                print(word, igain, cond_count_map, sep='\t', file=fout)

    top_ig_ngram_list = [word for ig, word in sorted(result, reverse=True)][:wanted_vocab_size]

    # pylint: disable=pointless-string-statement
    """
    MAX_NUM_SKIP_UNIGRAM = 175
    count_top_ig_unigram = 0
    top_ig_unigram_list = []
    count_top_ig_skipgram = 0
    for ig, word in sorted(result, reverse=True):
        if count_top_ig_unigram < MAX_NUM_SKIP_UNIGRAM:
            if ' ' not in word:
                top_ig_unigram_list.append(word)
                count_top_ig_unigram += 1
        if count_top_ig_skipgram > wanted_vocab_size:
            break
        top_ig_ngram_list.append(word)
        count_top_ig_skipgram += 1

    print('len(top_ig_ngram_list_old = {}, {}'.
          format(len(top_ig_ngram_list_old), top_ig_ngram_list_old[3]))
    print('len(top_ig_ngram_list = {}, {}'.
          format(len(top_ig_ngram_list), top_ig_ngram_list[3]))
    print('len(count_to_ig_unigram_list = {}, {}'.
          format(len(top_ig_unigram_list), top_ig_unigram_list))
    return top_ig_ngram_list, top_ig_unigram_list
    """

    return top_ig_ngram_list


# pylint: disable=C0103
def read_eb_sents_to_word_label_count_map(file_name):
    cond_dist = defaultdict(int)
    vocab = set([])
    label = 'class'

    # sent_list = []
    label_list = []
    # sent_wordset_list = []
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


# pylint: disable=fixme
# TODO, jsahw
# Is this function ever called?
# pylint: disable=C0103
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
