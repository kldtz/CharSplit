from datetime import datetime
import logging
import time
# pylint: disable=unused-import
from typing import Dict, List, Optional, Set

import numpy as np
from scipy import sparse

from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

from kirke.sampleutils.tablegen import fix_rate_table_text

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AddrLineTransformer(BaseEstimator, TransformerMixin):

    fit_count = 0
    transform_count = 0

    def __init__(self) -> None:
        self.name = 'AddrLineTransformer'
        self.version = '1.0'
        self.min_max_scaler = preprocessing.MinMaxScaler()

    # this returns a matrix
    # pylint: disable=unused-argument, invalid-name
    def candidates_to_matrix(self,
                             span_candidate_list: List[Dict],
                             y: Optional[List[bool]],
                             fit_mode: bool = False):
        numeric_matrix = np.zeros(shape=(len(span_candidate_list),
                                         1))
        for i, span_candidate in enumerate(span_candidate_list):
            prob = span_candidate['has_addr']
            numeric_matrix[i] = prob

        if fit_mode:
            self.min_max_scaler.fit(numeric_matrix)

        numeric_matrix = self.min_max_scaler.transform(numeric_matrix)
        return numeric_matrix

    # return self
    def fit(self,
            span_candidate_list: List[Dict],
            # pylint: disable=invalid-name
            y: Optional[List[bool]] = None):
        start_time = time.time()
        self.candidates_to_matrix(span_candidate_list, y, fit_mode=True)
        end_time = time.time()
        AddrLineTransformer.fit_count += 1
        logger.debug("%s fit called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, AddrLineTransformer.fit_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return self

    # return X_out
    # not sure what sparse matrix is represented in typing; use List for now
    def transform(self,
                  span_candidate_list: List[Dict]) -> List:
        start_time = time.time()
        # pylint: disable=invalid-name
        X_out = self.candidates_to_matrix(span_candidate_list, [], fit_mode=False)
        end_time = time.time()
        AddrLineTransformer.transform_count += 1
        logger.debug("%s transform called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, AddrLineTransformer.transform_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return X_out


class SurroundWordTransformer(BaseEstimator, TransformerMixin):

    fit_count = 0
    transform_count = 0

    def __init__(self) -> None:
        self.name = 'SurroundWordTransformer'
        self.version = '1.0'
        self.start = datetime.now()
        # change n-gram to 3 lowered F1 for effective date by 0.02
        self.prev_words_vectorizer = CountVectorizer(min_df=2,
                                                     ngram_range=(1, 2),
                                                     lowercase=False)
        self.post_words_vectorizer = CountVectorizer(min_df=2,
                                                     ngram_range=(1, 2),
                                                     lowercase=False)
        self.min_max_scaler = preprocessing.MinMaxScaler()

    # span_candidate_list should be a list of dictionaries
    # pylint: disable=unused-argument, invalid-name
    def candidates_to_matrix(self,
                             span_candidate_list: List[Dict],
                             y: Optional[List[bool]],
                             fit_mode: bool = False):
        prev_words_list = []
        post_words_list = []
        numeric_matrix = np.zeros(shape=(len(span_candidate_list), 2))
        for i, span_candidate in enumerate(span_candidate_list):
            prev_words_list.append(span_candidate.get('prev_n_words', ''))
            post_words_list.append(span_candidate.get('post_n_words', ''))
            numeric_matrix[i, 0] = span_candidate.get('candidate_percent10', 1.0)
            numeric_matrix[i, 1] = span_candidate.get('doc_percent', 1.0)

        if fit_mode:
            self.prev_words_vectorizer.fit(prev_words_list)
            self.post_words_vectorizer.fit(post_words_list)
            self.min_max_scaler.fit(numeric_matrix)
            return self

        # prev_matrxi and post_matrix are spare_matrix, not sure what typing should be
        prev_matrix = []  # type: List
        post_matrix = []  # type: List
        prev_matrix = self.prev_words_vectorizer.transform(prev_words_list)
        # print("prev_matrix.shape: {}".format(prev_matrix.shape))
        post_matrix = self.post_words_vectorizer.transform(post_words_list)
        numeric_matrix = self.min_max_scaler.transform(numeric_matrix)

        # print("post_matrix.shape: {}".format(post_matrix.shape))
        X_out = sparse.hstack((prev_matrix, post_matrix, numeric_matrix))
        return X_out


    # return self
    def fit(self,
            span_candidate_list: List[Dict],
            # pylint: disable=invalid-name
            y: Optional[List[bool]] = None):
        start_time = time.time()
        self.candidates_to_matrix(span_candidate_list, y, fit_mode=True)
        end_time = time.time()
        SurroundWordTransformer.fit_count += 1
        logger.debug("%s fit called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, SurroundWordTransformer.fit_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return self

    # return X_out
    # not sure what sparse matrix is represented in typing; use List for now
    def transform(self,
                  span_candidate_list: List[Dict]) -> List:
        start_time = time.time()
        # pylint: disable=invalid-name
        X_out = self.candidates_to_matrix(span_candidate_list, [], fit_mode=False)
        end_time = time.time()
        SurroundWordTransformer.transform_count += 1
        logger.debug("%s transform called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, SurroundWordTransformer.transform_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return X_out


class SimpleTextTransformer(BaseEstimator, TransformerMixin):
    fit_count = 0
    transform_count = 0

    def __init__(self) -> None:
        self.name = 'SimpleTextTransformer'
        self.version = '1.0'
        self.words_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2))

    # span_candidate_list should be a list of dictionaries
    # pylint: disable=unused-argument, invalid-name
    def candidates_to_matrix(self,
                             span_candidate_list: List[Dict],
                             y: Optional[List[bool]],
                             fit_mode: bool = False):
        words_list = []
        for span_candidate in span_candidate_list:
            words_list.append(span_candidate.get('text', ''))

        if fit_mode:
            self.words_vectorizer.fit(words_list)
            return self

        X_out = self.words_vectorizer.transform(words_list)
        return X_out


    # return self
    def fit(self,
            span_candidate_list: List[Dict],
            # pylint: disable=invalid-name
            y: Optional[List[bool]] = None):
        start_time = time.time()
        self.candidates_to_matrix(span_candidate_list, y, fit_mode=True)
        end_time = time.time()
        SimpleTextTransformer.fit_count += 1
        logger.debug("%s fit called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, SimpleTextTransformer.fit_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return self

    # return X_out
    # not sure what sparse matrix is represented in typing; use List for now
    def transform(self,
                  span_candidate_list: List[Dict]) -> List:
        start_time = time.time()
        X_out = self.candidates_to_matrix(span_candidate_list, [], fit_mode=False)
        end_time = time.time()
        SimpleTextTransformer.transform_count += 1
        logger.debug("%s transform called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, SimpleTextTransformer.transform_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return X_out


# pylint: disable=too-many-instance-attributes
class CharacterTransformer(BaseEstimator, TransformerMixin):
    fit_count = 0
    transform_count = 0

    def __init__(self) -> None:
        self.name = 'CharacterTransformer'
        self.version = '1.0'
        self.char_vectorizer = CountVectorizer(analyzer='char', min_df=2, ngram_range=(1, 2))
        self.generic_char_vectorizer = CountVectorizer(analyzer='word',
                                                       min_df=2,
                                                       token_pattern=r'(?u)[^\s]+',
                                                       ngram_range=(1, 3))
        self.first_char_vectorizer = CountVectorizer(analyzer='word',
                                                     token_pattern=r'(?u)[^\s]+',
                                                     ngram_range=(1, 2))
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.cat_vectorizer = DictVectorizer()
        self.start = datetime.now()

    # span_candidate_list should be a list of dictionaries
    # pylint: disable=unused-argument, invalid-name, too-many-locals
    def candidates_to_matrix(self,
                             span_candidate_list: List[Dict],
                             y: Optional[List[bool]],
                             fit_mode: bool = False):
        all_cands = []
        generic_chars_list = []
        all_first_chars = []
        numeric_matrix = np.zeros(shape=(len(span_candidate_list), 29))
        cat_dict_list = []
        for i, span_candidate in enumerate(span_candidate_list):
            cat_dict = {}
            chars = span_candidate.get('chars', '')
            all_cands.append(chars)
            if len(chars) > 1:
                all_first_chars.append("{} {}".format('FIRST-'+chars[0], 'SECOND-'+chars[1]))
            else:
                all_first_chars.append('FIRST-'+chars[0])
            chars_list = list(chars)
            cat_dict[span_candidate['candidate_type']] = 1
            cat_dict_list.append(cat_dict)
            # total length
            numeric_matrix[i, 0] = len(chars_list)
            # number of alpha character
            numeric_matrix[i, 1] = len([x for x in chars_list if x.isalpha()])
            # number of digits
            numeric_matrix[i, 2] = len([x for x in chars_list if x.isdigit()])
            # number of punct
            numeric_matrix[i, 3] = len([x for x in chars_list
                                        if not x.isalpha() and \
                                        not x.isdigit() and \
                                        not x.isspace()])
            # first char alpha
            numeric_matrix[i, 4] = chars[0].isalpha()
            # first char digit
            numeric_matrix[i, 5] = chars[0].isdigit()
            # sections divided by hyphens
            numeric_matrix[i, 6] = len([x for x in chars.split('-') if x])
            # sections divided by periods
            numeric_matrix[i, 7] = len([x for x in chars.split('.') if x])
            # sections divided by spaces
            numeric_matrix[i, 8] = len([x for x in chars.split(' ') if x])
            #first char is punct
            numeric_matrix[i, 9] = not chars[0].isalpha() and not chars[0].isdigit()
            # no alpha characters
            # pylint: disable=len-as-condition
            numeric_matrix[i, 10] = len([x for x in chars_list if x.isalpha()]) == 0
            match_len = 2
            for j in range(11, 29):
                # individual length features
                # pylint: disable=simplifiable-if-statement
                if len(chars_list) == match_len:
                    numeric_matrix[i, j] = True
                else:
                    numeric_matrix[i, j] = False
                match_len += 1
            generic_alpha_list = ['ALPHA' if x.isalpha() else x for x in chars_list]
            generic_num_list = ['NUM' if x.isdigit() else x for x in generic_alpha_list]
            generic_char_list = ['SPACE' if x.isspace() else x for x in generic_num_list]
            generic_chars = " ".join(generic_char_list)
            generic_chars_list.append(generic_chars)

        if fit_mode:
            self.char_vectorizer.fit(all_cands) # character ngram vectorizer
            self.generic_char_vectorizer.fit(generic_chars_list) # generic ngram vectorizer
            self.first_char_vectorizer.fit(all_first_chars) # vectorizer of first character
            self.min_max_scaler.fit(numeric_matrix)
            self.cat_vectorizer.fit(cat_dict_list)
            return self

        chars_out = self.char_vectorizer.transform(all_cands)
        generic_out = self.generic_char_vectorizer.transform(generic_chars_list)
        first_char_out = self.first_char_vectorizer.transform(all_first_chars)
        numeric_matrix = self.min_max_scaler.transform(numeric_matrix)
        cat_out = self.cat_vectorizer.transform(cat_dict_list)
        X_out = sparse.hstack((chars_out, numeric_matrix, generic_out, first_char_out, cat_out))
        return X_out


    # return self
    def fit(self,
            span_candidate_list: List[Dict],
            # pylint: disable=invalid-name
            y: Optional[List[bool]] = None):
        start_time = time.time()
        self.candidates_to_matrix(span_candidate_list, y, fit_mode=True)
        end_time = time.time()
        CharacterTransformer.fit_count += 1
        logger.debug("%s fit called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, CharacterTransformer.fit_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return self

    # return X_out
    # not sure what sparse matrix is represented in typing; use List for now
    def transform(self,
                  span_candidate_list: List[Dict]) -> List:
        start_time = time.time()
        X_out = self.candidates_to_matrix(span_candidate_list, [], fit_mode=False)
        end_time = time.time()
        CharacterTransformer.transform_count += 1
        logger.debug("%s transform called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, CharacterTransformer.transform_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return X_out


class TableTextTransformer(BaseEstimator, TransformerMixin):
    fit_count = 0
    transform_count = 0

    def __init__(self) -> None:
        self.name = 'TableTextTransformer'
        self.version = '1.0'
        self.words_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2))
        # min_df is set to 1 because not enough training data when doing
        # cross validation on small set
        self.sechead_vectorizer = CountVectorizer(min_df=1,
                                                  ngram_range=(1, 2))
        self.row_header_vectorizer = CountVectorizer(min_df=1,
                                                     ngram_range=(1, 2))
        self.pre_table_vectorizer = CountVectorizer(min_df=1,
                                                    ngram_range=(1, 2))
        self.min_max_scaler = preprocessing.MinMaxScaler()

        self.pos_word_set = set([])  # type: Set[str]


    # span_candidate_list should be a list of dictionaries
    # pylint: disable=unused-argument, invalid-name, too-many-locals, too-many-statements
    def candidates_to_matrix(self,
                             span_candidate_list: List[Dict],
                             y: Optional[List[bool]],
                             fit_mode: bool = False):

        pre_table_words_list = []  # type: List[str]
        sechead_words_list = []  # type: List[str]
        words_list = []  # type: List[str]
        row_header_words_list = []  # type: List[str]

        # must get all the positive words first
        if fit_mode:
            pos_cand_text_list = []  # type: List[str]
            pos_words_vectorizer = CountVectorizer(min_df=1,
                                                   ngram_range=(1, 2))
            if y:
                bool_label_list = y  # type: List[bool]
            else:
                bool_label_list = [False for span_candidate in span_candidate_list]
            for span_candidate, is_label in zip(span_candidate_list, bool_label_list):
                if is_label:
                    # table_text = fix_rate_table_text(span_candidate['text'])
                    table_text_alphanum = span_candidate['text_alphanum']
                    print("jj table_text_alphanum: [%s]" %
                          table_text_alphanum.replace('\n', '~'))
                    pos_cand_text_list.append(table_text_alphanum)
            pos_words_vectorizer.fit(pos_cand_text_list)
            print("pos_words_vectorizer.vocab = {}".format(pos_words_vectorizer.vocabulary_))
            self.pos_word_set = set(pos_words_vectorizer.vocabulary_.keys())
        # doesn't matter if min_df is 1 or 2
        pos_word_tokenizer = self.words_vectorizer.build_tokenizer()

        numeric_matrix = np.zeros(shape=(len(span_candidate_list), 36))
        for i, span_candidate in enumerate(span_candidate_list):
            # table_text = fix_rate_table_text(span_candidate['text'])
            table_text_alphanum = span_candidate['text_alphanum']
            print("text_alphanum: [{}]".format(table_text_alphanum))
            words_list.append(table_text_alphanum)
            row_header_text = fix_rate_table_text(span_candidate['row_header_text'])
            row_header_words_list.append(row_header_text)
            pre_table_words_list.append(span_candidate['pre_table_text'])
            sechead_words_list.append(fix_rate_table_text(span_candidate['sechead_text']))

            numeric_matrix[i, 0] = 1.0 if span_candidate['is_abbyy_original'] else 0.0
            numeric_matrix[i, 1] = 1.0 if span_candidate['is_in_exhibit'] else 0.0
            numeric_matrix[i, 2] = span_candidate['doc_percent']
            numeric_matrix[i, 3] = span_candidate['num_word']
            numeric_matrix[i, 4] = span_candidate['num_number']
            numeric_matrix[i, 5] = span_candidate['num_currency']
            numeric_matrix[i, 6] = span_candidate['num_percent']
            numeric_matrix[i, 7] = span_candidate['num_phone_number']
            numeric_matrix[i, 8] = span_candidate['num_date']
            numeric_matrix[i, 9] = span_candidate['num_alpha_word']
            numeric_matrix[i, 10] = span_candidate['num_alphanum_word']
            numeric_matrix[i, 11] = span_candidate['num_bad_word']
            numeric_matrix[i, 12] = 1.0 if span_candidate['has_number'] else 0.0
            numeric_matrix[i, 13] = 1.0 if span_candidate['has_currency'] else 0.0
            numeric_matrix[i, 14] = 1.0 if span_candidate['has_percent'] else 0.0
            numeric_matrix[i, 15] = 1.0 if span_candidate['has_phone_number'] else 0.0
            numeric_matrix[i, 16] = 1.0 if span_candidate['has_date'] else 0.0
            numeric_matrix[i, 17] = span_candidate['num_alpha_word']
            numeric_matrix[i, 18] = 1.0 if span_candidate['is_num_alpha_word_le10'] else 0.0
            numeric_matrix[i, 19] = 1.0 if span_candidate['is_num_alpha_word_le20'] else 0.0
            numeric_matrix[i, 20] = span_candidate['num_word_div_100']
            numeric_matrix[i, 21] = span_candidate['num_alpha_word_div_100']
            numeric_matrix[i, 22] = span_candidate['perc_number_word']
            numeric_matrix[i, 23] = span_candidate['perc_currency_word']
            numeric_matrix[i, 24] = span_candidate['perc_percent_word']
            numeric_matrix[i, 25] = span_candidate['perc_phone_word']
            numeric_matrix[i, 26] = span_candidate['perc_date_word']
            numeric_matrix[i, 27] = span_candidate['perc_alpha_word']
            numeric_matrix[i, 28] = span_candidate['perc_alphanum_word']
            numeric_matrix[i, 29] = span_candidate['perc_bad_word']
            numeric_matrix[i, 30] = span_candidate['len_pre_table_text']
            numeric_matrix[i, 31] = span_candidate['num_rows']
            numeric_matrix[i, 32] = span_candidate['num_cols']
            numeric_matrix[i, 33] = span_candidate['num_period_cap']
            numeric_matrix[i, 34] = 1.0 if span_candidate['has_dollar_div'] else 0.0

            # how compute the % words found in positive examples
            table_word_list = pos_word_tokenizer(table_text_alphanum)
            pos_word_list = [word for word in table_word_list if word in self.pos_word_set]
            numeric_matrix[i, 35] = len(pos_word_list) / len(table_word_list)

        if fit_mode:
            self.words_vectorizer.fit(words_list)
            self.pre_table_vectorizer.fit(pre_table_words_list)
            self.sechead_vectorizer.fit(sechead_words_list)
            self.row_header_vectorizer.fit(row_header_words_list)
            self.min_max_scaler.fit(numeric_matrix)

            print("----- fit:")
            print("pretable_vocab = {}".format(self.pre_table_vectorizer.vocabulary_))
            print("sechead_vocab = {}".format(self.sechead_vectorizer.vocabulary_))
            print("row_header_vocab = {}".format(self.row_header_vectorizer.vocabulary_))
            print("pos_word_list = {}".format(self.pos_word_set))
            return self

        words_out = self.words_vectorizer.transform(words_list)
        row_header_out = self.row_header_vectorizer.transform(row_header_words_list)
        numeric_matrix = self.min_max_scaler.transform(numeric_matrix)
        sechead_out = self.sechead_vectorizer.transform(sechead_words_list)
        pre_table_out = self.pre_table_vectorizer.transform(pre_table_words_list)
        X_out = sparse.hstack((words_out,
                               row_header_out,
                               numeric_matrix,
                               sechead_out,
                               pre_table_out))
        return X_out


    # return self
    def fit(self,
            span_candidate_list: List[Dict],
            # pylint: disable=invalid-name
            y: Optional[List[bool]] = None):
        start_time = time.time()
        self.candidates_to_matrix(span_candidate_list, y, fit_mode=True)
        end_time = time.time()
        TableTextTransformer.fit_count += 1

        logger.debug("%s fit called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, TableTextTransformer.fit_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return self

    # return X_out
    # not sure what sparse matrix is represented in typing; use List for now
    def transform(self,
                  span_candidate_list: List[Dict]) -> List:
        start_time = time.time()
        X_out = self.candidates_to_matrix(span_candidate_list, [], fit_mode=False)
        end_time = time.time()
        TableTextTransformer.transform_count += 1
        logger.debug("%s transform called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, TableTextTransformer.transform_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return X_out
