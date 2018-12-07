from datetime import datetime
import logging
import time
from typing import Dict, List, Optional, Set

import numpy as np
from scipy import sparse

from nltk import FreqDist

from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

from kirke.eblearn import ebattrvec
from kirke.eblearn import igain, bigramutils
from kirke.utils import stopwordutils, strutils
from kirke.eblearn.ebtransformerbase import EbTransformerBase

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_NUM_BI_TOPGRAM_WORDS = 175

#pylint: disable=too-many-instance-attributes
class SentTransformer(EbTransformerBase):

    fit_count = 0
    transform_count = 0

    def __init__(self) -> None:
        self.name = 'SentTransformer'
        self.version = '1.2'
        self.binary_attr_list = ebattrvec.DEFAULT_BINARY_ATTR_LIST
        self.numeric_attr_list = ebattrvec.DEFAULT_NUMERIC_ATTR_LIST
        self.categorical_attr_list = ebattrvec.DEFAULT_CATEGORICAL_ATTR_LIST

        self.cols_to_keep = []

        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')

        self.n_top_positive_words = [] # type: List[str]
        self.vocab_id_map = {}  # type: Dict[str, int]
        self.positive_vocab = set([])  # type: Set[str]

        self.vocabulary = {}  # type: Dict[str, int]

        self.sechead_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2))

    # pylint: disable=too-many-locals
    def gen_bi_topgram_matrix(self, sent_st_list, fit_mode=False):
        # for each sentence, find which top words it contains.  Then generate all pairs of these,
        # and generate the sparse matrix row entries for the rows it contains.

        indptr = [0]
        indices = []
        data = []
        for sent_st in sent_st_list:
            sent_words = set(sent_st.split())
            found_words = [common_word for common_word
                           in self.n_top_positive_words if common_word in sent_words]
            for index_w1, tmp_w1 in enumerate(found_words):
                for index_w2 in range(index_w1 + 1, len(found_words)):
                    tmp_w2 = found_words[index_w2]
                    col_name = ",".join((tmp_w1, tmp_w2))
                    if fit_mode:
                        index = self.vocabulary.setdefault(col_name, len(self.vocabulary))
                        indices.append(index)
                        data.append(1)
                    else:
                        if col_name in self.vocabulary:
                            index = self.vocabulary[col_name]
                            indices.append(index)
                            data.append(1)
            indptr.append(len(indices))
        if fit_mode:
            bi_topgram_matrix = sparse.csr_matrix((data, indices, indptr), dtype=int)
        else:
            bi_topgram_matrix = sparse.csr_matrix((data, indices, indptr),
                                                  shape=(len(sent_st_list), len(self.vocabulary)),
                                                  dtype=int)
        return bi_topgram_matrix

    def gen_top_ngram_matrix(self, sent_st_list, tokenize):
        # for each sentence, find which top words it contains.  Then generate all pairs of these,
        # and generate the sparse matrix row entries for the rows it contains.

        indptr = [0]
        indices = []
        data = []
        perc_positive_ngram_list = []
        for sent_st in sent_st_list:
            sent_wordset = tokenize(sent_st)
            count_pos_ngram = 0
            for ngram in sent_wordset:
                if ngram in self.positive_vocab:
                    count_pos_ngram += 1
            if sent_wordset:
                perc_positive_ngram_list.append(count_pos_ngram / len(sent_wordset))
            else:
                perc_positive_ngram_list.append(0.0)

            for ngram in sent_wordset:
                index = self.vocab_id_map.get(ngram)
                if index:
                    indices.append(index)
                    data.append(1)
            indptr.append(len(indices))

        top_ig_ngram_matrix = sparse.csr_matrix((data, indices, indptr),
                                                shape=(len(sent_st_list), len(self.vocab_id_map)),
                                                dtype=int)
        return top_ig_ngram_matrix, perc_positive_ngram_list

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def candidates_to_matrix(self,
                             span_candidate_list: List[Dict],
                             y: Optional[List[bool]],
                             fit_mode: bool = False):

        num_rows = len(span_candidate_list)
        numeric_matrix = np.zeros(shape=(num_rows,
                                         len(self.binary_attr_list) +
                                         len(self.numeric_attr_list)))
        categorical_matrix = np.zeros(shape=(num_rows,
                                             len(self.categorical_attr_list)))

        # populates numeric and categorical matrices
        for i, span_candidate in enumerate(span_candidate_list):
            cand_attrvec = span_candidate['attrvec']
            for ibin, binary_attr in enumerate(self.binary_attr_list):
                numeric_matrix[i, ibin] = strutils.bool_to_int(cand_attrvec.get_val(binary_attr))
            for inum, numeric_attr in enumerate(self.numeric_attr_list):
                # pylint: disable=line-too-long
                numeric_matrix[i, len(self.binary_attr_list) + inum] = cand_attrvec.get_val(numeric_attr)
            for icat, cat_attr in enumerate(self.categorical_attr_list):
                categorical_matrix[i, icat] = cand_attrvec.get_val(cat_attr)
        if fit_mode:
            numeric_matrix = self.min_max_scaler.fit_transform(numeric_matrix)
            categorical_matrix = self.one_hot_encoder.fit_transform(categorical_matrix)
        else:
            numeric_matrix = self.min_max_scaler.transform(numeric_matrix)
            categorical_matrix = self.one_hot_encoder.transform(categorical_matrix)

        # handles bag of words
        sent_st_list = []
        positive_sent_st_list = []  # only populated if fit_mode
        sechead_st_list = []
        if fit_mode:
            for span_candidate, label in zip(span_candidate_list, y):
                sent_st_list.append(span_candidate['text'])
                if label:
                    positive_sent_st_list.append(span_candidate['text'])
                sechead_st_list.append(span_candidate['attrvec'].sechead)
        else:
            for span_candidate in span_candidate_list:
                sent_st_list.append(span_candidate['text'])
                sechead_st_list.append(span_candidate['attrvec'].sechead)

        nostop_sent_st_list = stopwordutils.remove_stopwords(sent_st_list, mode=2)

        if fit_mode:
            logger.info("starting computing info_gain")
            ##### not used???
            igain_vocab = igain.doc_label_list_to_vocab(sent_st_list,
                                                        y,
                                                        tokenize=bigramutils.eb_doc_to_all_ngrams,
                                                        debug_mode=False)

            logger.info("starting computing unigram and bigram")
            unused_vocab, positive_vocab = bigramutils.doc_label_list_to_vocab(sent_st_list,
                                                                               y,
                                                                               # pylint: disable=line-too-long
                                                                               tokenize=bigramutils.eb_doc_to_all_ngrams)

            vocab_id_map = {}
            for vid, vocab in enumerate(igain_vocab):
                vocab_id_map[vocab] = vid
            self.vocab_id_map = vocab_id_map
            self.positive_vocab = positive_vocab

            DEBUG_MODE = True
            if DEBUG_MODE:
                with open("/tmp/v3344_vocabs.tsv", "wt") as fvcabout:
                    for word in igain_vocab:
                        print(word, file=fvcabout)
                print('wrote {}'.format("/tmp/v3344_vocabs.tsv"))

                with open("/tmp/v3344_posvocabs.tsv", "wt") as fvcabout:
                    for word in positive_vocab:
                        print(word, file=fvcabout)
                print('wrote {}'.format("/tmp/v3344_posvocabs.tsv"))

            try:
                self.sechead_vectorizer.fit(sechead_st_list)
            except ValueError:
                self.sechead_vectorizer = CountVectorizer(vocabulary=['dummy'])

            logger.info("starting computing bi_topgram")
            #pylint: disable=line-too-long
            nostop_positive_sent_st_list = stopwordutils.remove_stopwords(positive_sent_st_list, mode=0)
            filtered_list = []
            for nostop_positive_sent in nostop_positive_sent_st_list:
                for tmp_w in nostop_positive_sent.split():
                    if len(tmp_w) > 3:
                        filtered_list.append(tmp_w)
            fdistribution = FreqDist(filtered_list)
            self.n_top_positive_words = [item[0] for item in
                                         fdistribution.most_common(MAX_NUM_BI_TOPGRAM_WORDS)]

        bow_matrix, perc_positive_ngrams = self.gen_top_ngram_matrix(sent_st_list,
                                                                     # pylint: disable=line-too-long
                                                                     tokenize=bigramutils.eb_doc_to_all_ngrams)
        sechead_matrix = self.sechead_vectorizer.transform(sechead_st_list)

        bi_topgram_matrix = self.gen_bi_topgram_matrix(nostop_sent_st_list, fit_mode=fit_mode)

        perc_pos_ngram_matrix = np.zeros(shape=(num_rows, 1))
        for instance_i, perc_pos_ngram in enumerate(perc_positive_ngrams):
            perc_pos_ngram_matrix[instance_i, 0] = perc_pos_ngram


        comb_matrix = sparse.hstack((numeric_matrix,
                                     perc_pos_ngram_matrix,
                                     categorical_matrix,
                                     bow_matrix,
                                     sechead_matrix))
        sparse_comb_matrix = sparse.csr_matrix(comb_matrix)
        nozero_sparse_comb_matrix = self.remove_zero_column(sparse_comb_matrix, fit_mode=fit_mode)

        X = sparse.hstack((nozero_sparse_comb_matrix, bi_topgram_matrix), format='csr')

        return X

    def fit(self,
            span_candidate_list: List[Dict],
            # pylint: disable=invalid-name
            y: Optional[List[bool]] = None):
        start_time = time.time()
        self.candidates_to_matrix(span_candidate_list, y, fit_mode=True)
        end_time = time.time()
        SentTransformer.fit_count += 1
        logger.debug("%s fit called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, SentTransformer.fit_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return self

    def transform(self,
                  span_candidate_list: List[Dict]) -> List:
        start_time = time.time()
        X_out = self.candidates_to_matrix(span_candidate_list, [], fit_mode=False)
        end_time = time.time()
        SentTransformer.fit_count += 1
        logger.debug("%s fit called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, SentTransformer.fit_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return X_out


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
        SurroundWordTransformer.transform_count += 1
        logger.debug("%s transform called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, SurroundWordTransformer.transform_count, len(span_candidate_list),
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
