import os
import nltk
import re
import math

from nltk.tokenize import word_tokenize

from collections import namedtuple

from kirke.utils import strutils, stopwordutils

POSS_PAT = re.compile(r'(\'|’)s\b')

PUNCT_PAT = re.compile(r'[\.,\?;:\{\}\(\)\[\]\/\\\'’"`”“\*#@_ ]')

ALPHAS_PAT = re.compile(r'^[a-zA-Z]+$')

ANY_DIGIT_PAT = re.compile('.*\d+.*')

def remove_punct(line):
    return re.sub(PUNCT_PAT, ' ', line)

def is_alphas(line):
    mat = re.match(ALPHAS_PAT, line)
    return mat

def line2bigrams(line):
    tokens = line2words(line)
    bigrams = nltk.bigrams(tokens)
    return bigrams

def line2words(line):
    raw = re.sub(POSS_PAT, ' ', line)
    raw = remove_punct(raw)

    tokens = nltk.word_tokenize(raw)
    tokens = [word for word in tokens if is_alphas(word)]
    tokens = [word.lower() for word in tokens]
    return tokens

def line2char_trigram(line):
    line = line.lower().replace('\t',' ')

    trigrams = []
    # print("line: {}".format(line))
    for i in range(len(line)-2):
        trigram = line[i:i+3]
        if trigram.strip():  # not all spaces
            digit_mat = ANY_DIGIT_PAT.match(trigram)
            if not ('  ' in trigram or
                    '__' in trigram or
                    '██' in trigram):
                trigrams.append(trigram)
    return trigrams

def files2ngram(adir, suffix='.txt', n=2):
    corpus_freq = nltk.FreqDist()
    for filename in os.listdir(adir):
        if filename.endswith(suffix):
            # print('filename = {}'.format(filename))
            with open('{}/{}'.format(adir, filename), 'rt') as fin:
                raw = fin.read()

                # remove all possessive's
                raw = re.sub(POSS_PAT, ' ', raw)
                raw = remove_punct(raw)

                tokens = nltk.word_tokenize(raw)
                tokens = [word for word in tokens if is_alphas(word)]
                tokens = [word.lower() for word in tokens]

                bigrams = nltk.bigrams(tokens)

                #compute frequency distribution for all the bigrams in the text
                fdist = nltk.FreqDist(bigrams)

                corpus_freq |= fdist

    total_bigram = 0
    adict = {}
    for k, v in corpus_freq.items():
        if v >= 2:
            # print("{}\t{} {}".format(v, k[0], k[1]))
            total_bigram += v
            adict[k] = v

    result = {}
    for k, v in adict.items():
        if v >= 2:
            xlog = math.log(v / total_bigram)
            print("{}\t{} {}\t{}".format(v, k[0], k[1], xlog))
            result[k] = xlog

    return result


def files2unigram(adir, suffix='.txt'):
    corpus_freq = nltk.FreqDist()
    for filename in os.listdir(adir):
        if filename.endswith(suffix):
            # print('filename = {}'.format(filename))
            with open('{}/{}'.format(adir, filename), 'rt') as fin:
                raw = fin.read()

                tokens = line2words(raw)

                #compute frequency distribution for all the bigrams in the text
                fdist = nltk.FreqDist(tokens)

                corpus_freq |= fdist

    total_bigram = 0
    adict = {}
    for k, v in corpus_freq.items():
        if v >= 2:
            # print("{}\t{} {}".format(v, k[0], k[1]))
            total_bigram += v
            adict[k] = v

    result = {}
    for k, v in adict.items():
        if v >= 2:
            xlog = math.log(v / total_bigram)
            print("{}\t{}\t{}".format(v, k, xlog))
            result[k] = xlog

    return result


def files2char_ngram(adir, suffix='.txt', n=3):
    corpus_freq = nltk.FreqDist()
    for filename in os.listdir(adir):
        if filename.endswith(suffix):
            # print('filename = {}'.format(filename))
            with open('{}/{}'.format(adir, filename), 'rt') as fin:

                trigrams = []
                for line in fin:
                    line = line.strip()

                    line_trigrams = line2char_trigram(line)
                    trigrams.extend(line_trigrams)

                        
                #compute frequency distribution for all the bigrams in the text
                fdist = nltk.FreqDist(trigrams)

                corpus_freq |= fdist

    total_bigram = 0
    adict = {}
    for k, v in corpus_freq.items():
        if v >= 2:
            total_bigram += v
            adict[k] = v

    result = {}
    for k, v in adict.items():
        if v >= 2:
            xlog = math.log(v / total_bigram)
            print("{}\t{}\t{}".format(v, k, xlog))
            result[k] = xlog

    return result


def load_ngram_score(filename):
    adict = {}
    with open(filename, 'rt') as fin:
        for line in fin:
            line = line.strip()
            freq, ngram_st, score = line.split('\t')
            adict[ngram_st] = float(score)
    return adict

unigram_score_map = load_ngram_score('dict/unigram300.tsv')
bigram_score_map = load_ngram_score('dict/bigram300.tsv')
char_trigram_score_map = load_ngram_score('dict/trigram300.char.tsv')


def bi2st(bigram):
    return '{} {}'.format(bigram[0], bigram[1])

def check_english(filename):
    with open(filename, 'rt') as fin:
        for line in fin:
            line = line.strip()

            if not line:
                print()
                continue

            words = line2words(line)
            if words:
                scores = [unigram_score_map.get(word, -10.75)
                          for word in words]
                # print("scores = {}".format(scores))
                avg_uni_score = sum(scores) / len(scores)
            else:
                avg_uni_score = -10.75
                
            bigrams = [bi2st(bigram) for bigram in line2bigrams(line)]
            if bigrams:
                # print("bigrams = {}".format(list(bigrams)))
                scores = [bigram_score_map.get(bigram, -11.5)
                          for bigram in bigrams]
                # print("scores = {}".format(scores))
                avg_bi_score = sum(scores) / len(scores)
            else:
                avg_bi_score = -11.5

            char_trigrams = line2char_trigram(line)
            if char_trigrams:
                # print("char_trigrams = {}".format(list(char_trigrams)))
                scores = [char_trigram_score_map.get(trigram, -12.25)
                          for trigram in char_trigrams]
                # print("scores = {}".format(scores))
                avg_tri_score = sum(scores) / len(scores)
            else:
                avg_tri_score = -12.25

            print('{}\t{}\t\t{}\t{}'.format(avg_tri_score,
                                            avg_uni_score, avg_bi_score, line))


def classify_english_sentence_v1(line, debug_mode=False):
    words = line2words(line)

    if len(words) <= 5:
        return False

    if '.....' in line:
        return False
    
    if words:
        scores = [unigram_score_map.get(word, -10.75)
                  for word in words]
        # print("scores = {}".format(scores))
        avg_uni_score = sum(scores) / len(scores)
    else:
        avg_uni_score = -10.75

    bigrams = [bi2st(bigram) for bigram in line2bigrams(line)]
    if bigrams:
        # print("bigrams = {}".format(list(bigrams)))
        scores = [bigram_score_map.get(bigram, -11.5)
                  for bigram in bigrams]
        # print("scores = {}".format(scores))
        avg_bi_score = sum(scores) / len(scores)
    else:
        avg_bi_score = -11.5

    char_trigrams = line2char_trigram(line)
    if char_trigrams:
        # print("char_trigrams = {}".format(list(char_trigrams)))
        scores = [char_trigram_score_map.get(trigram, -12.25)
                  for trigram in char_trigrams]
        # print("scores = {}".format(scores))
        avg_tri_score = sum(scores) / len(scores)
    else:
        avg_tri_score = -12.25

    if avg_tri_score < -11.0:
        return False
    
    result = ((len(words) >= 15 and avg_uni_score >= -8.5) or
              (avg_tri_score >= -8.0 and avg_uni_score >= -8.0) or
              (avg_tri_score >= -8.0 and avg_uni_score >= -8.2 and
               avg_bi_score >= -10.93))

    if debug_mode:
        print("eng diff {}\t{}\t{}\t{}\t{}\txxx".format(avg_tri_score,
                                                        avg_uni_score,
                                                        avg_bi_score,
                                                        result,
                                                        line))

    return result

# MINNKOTA PPA PROJECT TURBINES OTHER'S  PROJECT  TURBINES,  IF ANY PURCHASER'S  CHECK METER
def is_double_space_3_times(line):
    # most likely a table row
    if len(line) < 120 and line.count('  ') >= 3:
        return True
        """
        start_end_indices = strutils.find_substr_indices('  ', line)
        prev_start = start_end_indices[0][0]
        count_diff_small = 0
        for start_end in start_end_indices[1:]:
            start = start_end[0]
            if start - prev_start < 25:
                count_diff_small += 1
            prev_start = start
        if count_diff_small >= 2:
            return True
        """
    return False

def is_math_equation(line):
    operator_count = len(strutils.find_substr_indices(r' [\+\-\*/%x] ', line))
    eq_count = line.count('=')
    digit_count = len(strutils.find_substr_indices(r'[0-9]+', line))
    if len(line) < 150:
        if operator_count + eq_count >= 10:   # something else is going on, not equation
            return False
        if (eq_count > 0 or (operator_count + eq_count >= 3) or
            (operator_count + eq_count >= 2 and digit_count >= 2)):
            return True
    return False

# handle (a), a), 1), (1), 1., iii. ix.  ix)
ITEM_PREFIX_PAT = re.compile(r'^\(?([a-z]|\d{1,2}|[ixv]+)[\)\.]\s.+')

def is_itemize_prefix(line):
    return ITEM_PREFIX_PAT.match(line)


def classify_english_sentence(line: str, debug_mode=False) -> bool:
    words = line2words(line)

    if debug_mode:
        print('len(words) =', len(words))
        
    # if a line is itemized, it is probably NOT inside a table
    # it can be non-English sentence though, but assume not the case
    if is_itemize_prefix(line) and len(words) >= 4:
        return True

    if len(words) <= 5:
        return False

    if '.....' in line:
        return False

    # replace multiple underscores with just one, to keep the 'other' count work
    line = re.sub(r'__+', '_', line)
    line = re.sub(r'\.\.+', '.', line)
    # we don't bother with resetting 'words'

    colon_count = line.count(':')
    if len(line) < 100 and colon_count >= 2:
        return False

    # table row with date
    if strutils.count_date(line) > 0 and line.count('  ') >= 2:
        return False

    if is_double_space_3_times(line):
        return False

    if is_math_equation(line):
        return False

    stopword_count = stopwordutils.count_stopword(words)
    cht_count = count_char_type(line)

    if debug_mode:
        print("charcount\t{}".format(cht_count))
        print("eng diff, vowel/cons = {}, alph/len = {}, other/len = {}, digit/len = {}".format(
            cht_count.num_vowel / cht_count.num_consonant,
            cht_count.num_alpha / cht_count.length,
            cht_count.num_other / cht_count.length,
            cht_count.num_digit / cht_count.length))
        print("eng diff, stop/len = {}".format(stopword_count / len(words)))

    if cht_count.num_alpha / cht_count.length < 0.3:
        return False
    
    if cht_count.num_digit / cht_count.length > 0.3:
        return False

    if len(words) <= 14 and stopword_count >= 3:
        return True

    if len(words) > 10 and stopword_count / len(words) >= 0.12:
        return True

    # lowest seen so far if 0.48
    if cht_count.num_vowel / cht_count.num_consonant < 0.46:
        return False
    if cht_count.num_alpha / cht_count.length < 0.66:  # lowest seen 0.69
        return False
    if cht_count.num_other / cht_count.length > 0.14:  # max seen, 0.11
        return False

    return True
    
    
def classify_english_line(filename, debug_mode=False):

    with open(filename, 'rt') as fin:
        for line in fin:
            line = line.strip()

            if not line:
                print()
                continue

            is_english_sentence = classify_english_sentence(line)
            is_english_sent2 = classify_english_sent2(line)

            if is_english_sentence != is_english_sent2:
                print("eng diff")
                print("eng diff")
                print("eng diff!!!!!!!!!!!!!!!! eng= {}, eng2={}".format(is_english_sentence,
                                                                         is_english_sent2))
                # to print out debug info
                classify_english_sentence(line, debug_mode=True)
                classify_english_sent2(line, debug_mode=True)
                print("eng diff [{}]".format(line))
            
            print('{}\t{}'.format(is_english_sentence, line))


def classify_english_lines(adir):
    #check_english('data-test-tmp/test1.txt', unigram_scores, bigram_scores,
    #              char_trigram_scores)
    

    suffix = '.txt'
    filenames = [filename for filename in os.listdir(adir)]

    for filename in sorted(filenames):
        if filename.endswith(suffix) and len(filename) <= 10:
            # print('filename = {}'.format(filename))
            # with open('{}/{}'.format(adir, filename), 'rt') as fin:
            classify_english_line('{}/{}'.format(adir,filename))

CharTypeCount = namedtuple('CharTypeCount', ['num_space', 'num_uc', 'num_lc', 'num_alpha', 'num_digit', 'num_punct', 'num_other', 'num_consonant', 'num_vowel', 'length'])

def count_char_type(line):
    # we don't count beginning and end spaces
    line = line.strip()
    num_space, num_uc, num_lc, num_alpha = 0, 0, 0, 0
    num_digit, num_punct, num_other = 0, 0, 0
    num_consonant, num_vowel = 0, 0
    length = len(line)
    if line:
        for ch in line:
            if strutils.is_uc(ch):
                num_uc += 1
                if strutils.is_english_vowel(ch):
                    num_vowel += 1
                else:
                    num_consonant += 1
            elif strutils.is_lc(ch):
                num_lc += 1
                if strutils.is_english_vowel(ch):
                    num_vowel += 1
                else:
                    num_consonant += 1                
            elif strutils.is_space(ch):
                num_space += 1
            elif strutils.is_digit_core(ch):
                num_digit += 1
            elif strutils.is_punct_core(ch):
                num_punct += 1
            else:
                num_other += 1  # parenthesis, hyphen
        num_alpha = num_uc + num_lc
        num_other += num_punct   # other include punct
    return CharTypeCount(num_space, num_uc, num_lc, num_alpha,
                         num_digit, num_punct, num_other,
                         num_consonant, num_vowel, length)


MONTH_ST_LIST = ['January', 'February', 'March', 'April', 'May',
                 'June', 'July', 'August', 'September', 'October',
                 'November', 'December',
                 'Jan', 'Feb', 'Mar', 'Apr', 'May',
                 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                 'Nov', 'Dec']

date_pat = re.compile(r'({})\b.*\d\d\d\d\b'.format('|'.join(MONTH_ST_LIST)),
                      re.IGNORECASE)

# the xxx day of xxx 2016
date2_pat = re.compile(r'\bday of\b.*\d\d\d\d\b', re.IGNORECASE)

date3_pat = re.compile(r'\bdated\b', re.IGNORECASE)

def is_date_line(line):
    if len(line) > 50:  # don't want to match line "BY AND BETWEEN" in title page
        return False
    mat = date_pat.search(line) or date2_pat.search(line)
    return mat

def has_date(line):
    if (date_pat.search(line) or
        date2_pat.search(line) or
        date3_pat.search(line)):
        return True
    return False


skip_template_pat = re.compile(r'(ACT OF 1933|ACT of 1934|all rights? reserved)', re.IGNORECASE)

# bad example of above regex
"""[This Convertible Promissory Note is one of a series of duly
authorized and issued convertible promissory notes of All Fuels &
Energy Company, a Delaware corporation (the “Company”), designated
its 8% Convertible Promissory Notes due September 1, 2013 (the
“Note”), issued to Equity Highrise, Inc. (together with its permitted
successors and assigns, the “Holder”) in accordance with exemptions
from registration under the Securities Act of 1933, as amended (the
“Securities Act”), pursuant to a Securities Purchase Agreement, dated
August 15, 2011 (the “Securities Purchase Agreement”) between the
Company and the Holder.  Capitalized terms not otherwise defined
herein shall have the meanings ascribed to them in the Securities
Purchase Agreement.]
"""

def is_skip_template_line(line):
    if skip_template_pat.search(line) and not has_date(line):
        return True
    return False

