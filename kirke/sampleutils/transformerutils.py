
import logging
import time
from typing import Dict, List, Optional

import numpy as np
from scipy import sparse

from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class AddrLineTransformer(BaseEstimator, TransformerMixin):

    fit_count = 0
    transform_count = 0

    def __init__(self) -> None:
        self.name = 'AddrLineTransformer'
        self.version = '1.0'
        self.min_max_scaler = preprocessing.MinMaxScaler()

    # pylint: disable=unused-argument, invalid-name
    def samples_to_matrix(self,
                          span_sample_list: List[Dict],
                          y: Optional[List[bool]],
                          fit_mode: bool = False):
        numeric_matrix = np.zeros(shape=(len(span_sample_list),
                                         1))
        for i, span_sample in enumerate(span_sample_list):
            prob = span_sample['has_addr']
            numeric_matrix[i] = prob
        if fit_mode:
            numeric_matrix = self.min_max_scaler.fit_transform(numeric_matrix)
        else:
            numeric_matrix = self.min_max_scaler.transform(numeric_matrix)

        return numeric_matrix

    # return self
    def fit(self,
            span_sample_list: List[Dict],
            # pylint: disable=invalid-name
            y: Optional[List[bool]] = None):
        start_time = time.time()
        self.samples_to_matrix(span_sample_list, y, fit_mode=True)
        end_time = time.time()
        AddrLineTransformer.fit_count += 1
        logging.debug("%s fit called #%d, len(span_sample_list) = %d, took %.0f msec",
                      self.name, AddrLineTransformer.fit_count, len(span_sample_list),
                      (end_time - start_time) * 1000)
        return self

    # return X_out
    # not sure what sparse matrix is represented in typing; use List for now
    def transform(self,
                  span_sample_list: List[Dict]) -> List:
        start_time = time.time()
        # pylint: disable=invalid-name
        X_out = self.samples_to_matrix(span_sample_list, [], fit_mode=False)
        end_time = time.time()
        AddrLineTransformer.transform_count += 1
        logging.debug("%s transform called #%d, len(span_sample_list) = %d, took %.0f msec",
                      self.name, AddrLineTransformer.transform_count, len(span_sample_list),
                      (end_time - start_time) * 1000)
        return X_out


class SurroundWordTransformer(BaseEstimator, TransformerMixin):

    fit_count = 0
    transform_count = 0

    def __init__(self) -> None:
        self.name = 'SurroundWordTransformer'
        self.version = '1.0'
        self.prev_words_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2))
        self.post_words_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2))

    # span_sample_list should be a list of dictionaries
    # pylint: disable=unused-argument, invalid-name
    def samples_to_matrix(self,
                          span_sample_list: List[Dict],
                          y: Optional[List[bool]],
                          fit_mode: bool = False):
        prev_words_list = []
        post_words_list = []
        for span_sample in span_sample_list:
            prev_words_list.append(span_sample.get('prev_n_words', ''))
            post_words_list.append(span_sample.get('post_n_words', ''))

        if fit_mode:
            self.prev_words_vectorizer.fit(prev_words_list)
            self.post_words_vectorizer.fit(post_words_list)
            return self

        # prev_matrxi and post_matrix are spare_matrix, not sure what typing should be
        prev_matrix = []  # type: List
        post_matrix = []  # type: List
        prev_matrix = self.prev_words_vectorizer.transform(prev_words_list)
        # print("prev_matrix.shape: {}".format(prev_matrix.shape))
        post_matrix = self.post_words_vectorizer.transform(post_words_list)
        # print("post_matrix.shape: {}".format(post_matrix.shape))
        X_out = sparse.hstack((prev_matrix, post_matrix))

        return X_out


    # return self
    def fit(self,
            span_sample_list: List[Dict],
            # pylint: disable=invalid-name
            y: Optional[List[bool]] = None):
        start_time = time.time()
        self.samples_to_matrix(span_sample_list, y, fit_mode=True)
        end_time = time.time()
        SurroundWordTransformer.fit_count += 1
        logging.debug("%s fit called #%d, len(span_sample_list) = %d, took %.0f msec",
                      self.name, SurroundWordTransformer.fit_count, len(span_sample_list),
                      (end_time - start_time) * 1000)
        return self

    # return X_out
    # not sure what sparse matrix is represented in typing; use List for now
    def transform(self,
                  span_sample_list: List[Dict]) -> List:
        start_time = time.time()
        # pylint: disable=invalid-name
        X_out = self.samples_to_matrix(span_sample_list, [], fit_mode=False)
        end_time = time.time()
        SurroundWordTransformer.transform_count += 1
        logging.debug("%s transform called #%d, len(span_sample_list) = %d, took %.0f msec",
                      self.name, SurroundWordTransformer.transform_count, len(span_sample_list),
                      (end_time - start_time) * 1000)
        return X_out


class SimpleTextTransformer(BaseEstimator, TransformerMixin):
    fit_count = 0
    transform_count = 0

    def __init__(self) -> None:
        self.name = 'SimpleTextTransformer'
        self.version = '1.0'
        self.words_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2))

    # span_sample_list should be a list of dictionaries
    # pylint: disable=unused-argument, invalid-name
    def samples_to_matrix(self,
                          span_sample_list: List[Dict],
                          y: Optional[List[bool]],
                          fit_mode: bool = False):
        words_list = []
        for span_sample in span_sample_list:
            words_list.append(span_sample.get('text', ''))

        if fit_mode:
            self.words_vectorizer.fit(words_list)
            return self

        X_out = self.words_vectorizer.transform(words_list)
        return X_out


    # return self
    def fit(self,
            span_sample_list: List[Dict],
            # pylint: disable=invalid-name
            y: Optional[List[bool]] = None):
        start_time = time.time()
        self.samples_to_matrix(span_sample_list, y, fit_mode=True)
        end_time = time.time()
        SurroundWordTransformer.fit_count += 1
        logging.debug("%s fit called #%d, len(span_sample_list) = %d, took %.0f msec",
                      self.name, SurroundWordTransformer.fit_count, len(span_sample_list),
                      (end_time - start_time) * 1000)
        return self

    # return X_out
    # not sure what sparse matrix is represented in typing; use List for now
    def transform(self,
                  span_sample_list: List[Dict]) -> List:
        start_time = time.time()
        X_out = self.samples_to_matrix(span_sample_list, [], fit_mode=False)
        end_time = time.time()
        SurroundWordTransformer.transform_count += 1
        logging.debug("%s transform called #%d, len(span_sample_list) = %d, took %.0f msec",
                      self.name, SurroundWordTransformer.transform_count, len(span_sample_list),
                      (end_time - start_time) * 1000)
        return X_out
