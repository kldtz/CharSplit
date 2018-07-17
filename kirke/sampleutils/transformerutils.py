from datetime import datetime
import logging
import time
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse

from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

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
            numeric_matrix = self.min_max_scaler.fit_transform(numeric_matrix)
        else:
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
            numeric_matrix = self.min_max_scaler.fit_transform(numeric_matrix)
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
        X_out = self.candidates_to_matrix(span_candidate_list, [], fit_mode=False)
        end_time = time.time()
        SurroundWordTransformer.transform_count += 1
        logger.debug("%s transform called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, SurroundWordTransformer.transform_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return X_out

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
            numeric_matrix = self.min_max_scaler.fit_transform(numeric_matrix)
            self.cat_vectorizer.fit_transform(cat_dict_list)
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
        X_out = self.candidates_to_matrix(span_candidate_list, [], fit_mode=False)
        end_time = time.time()
        SurroundWordTransformer.transform_count += 1
        logger.debug("%s transform called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, SurroundWordTransformer.transform_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return X_out
