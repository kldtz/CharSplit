from datetime import datetime
import logging
import time
# pylint: disable=unused-import
from typing import Dict, List, Optional, Set

import numpy as np
from scipy import sparse

from nltk import FreqDist

from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

from kirke.sampleutils.tablegen import fix_rate_table_text

from kirke.eblearn import ebattrvec
from kirke.eblearn import igain, bigramutils
from kirke.utils import stopwordutils, strutils
from kirke.eblearn.ebtransformerbase import EbTransformerBase

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_NUM_BI_TOPGRAM_WORDS = 175

CJK_SET = set(['zh', 'ja', 'ko'])


#pylint: disable=too-many-instance-attributes
class SentTransformer(EbTransformerBase):

    fit_count = 0
    transform_count = 0

    def __init__(self, provision: str) -> None:
        super().__init__(provision)
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
                             *,
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
            # mypy complaining about 'y'
            # Argument 2 to "zip" has incompatible type "Optional[List[bool]]"; expected
            # "Iterable[bool]"
            for span_candidate, label in zip(span_candidate_list, y):  # type: ignore
                attrvec = span_candidate['attrvec']
                sent_st = attrvec.bag_of_words
                sent_st_list.append(sent_st)
                if label:
                    positive_sent_st_list.append(sent_st)
                sechead_st_list.append(attrvec.sechead)
        else:
            for span_candidate in span_candidate_list:
                attrvec = span_candidate['attrvec']
                sent_st = attrvec.bag_of_words
                sent_st_list.append(sent_st)
                sechead_st_list.append(attrvec.sechead)

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

            DEBUG_MODE = False
            if DEBUG_MODE:
                with open("/tmp/v3344_vocabs.tsv", "wt") as fvcabout:
                    for word in igain_vocab:
                        print(word, file=fvcabout)
                print('wrote {}'.format("/tmp/v3344_vocabs.tsv"))

                with open("/tmp/v3344_posvocabs.tsv", "wt") as fvcabout:
                    for word in sorted(positive_vocab):
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
            # This should not be triggered.
            # if self.lang == '':
            #     raise Exception('SentTransformer {} has no lang specified.'.format(self.provision))
            if self.lang in CJK_SET:
                for nostop_positive_sent in nostop_positive_sent_st_list:
                    for tmp_w in nostop_positive_sent.split():
                        # if len(tmp_w) > 3:
                        filtered_list.append(tmp_w)
            else:
                for nostop_positive_sent in nostop_positive_sent_st_list:
                    for tmp_w in nostop_positive_sent.split():
                        if len(tmp_w) > 3:
                            filtered_list.append(tmp_w)

            # The words in FreqDist at the same frequency is unordered.
            # To make the classification result consistent, take the wanted
            # frequency and perform cut-off based on that frequency.
            fdistribution = FreqDist(filtered_list)
            wfreq_list = fdistribution.most_common(MAX_NUM_BI_TOPGRAM_WORDS * 2)
            if  MAX_NUM_BI_TOPGRAM_WORDS < len(wfreq_list):
                cut_off = wfreq_list[MAX_NUM_BI_TOPGRAM_WORDS][1]
            else:
                cut_off = 1
            ntop_positive_words = []
            for wfreq in wfreq_list:
                word, freq = wfreq
                if freq >= cut_off:
                    ntop_positive_words.append(word)
            self.n_top_positive_words = sorted(ntop_positive_words)
            if DEBUG_MODE:
                with open("/tmp/v3344_n_top_positive_words", 'wt') as faaout:
                    for word in self.n_top_positive_words:
                        print(word, file=faaout)

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

        is_debug = False
        if is_debug:
            print('  shape of bow_matrix: {}'.format(bow_matrix.shape))
            print('  shape of sechead_matrix: {}'.format(sechead_matrix.shape))
            print('  shape of bi_topgram_matrix: {}'.format(bi_topgram_matrix.shape))
            print('  shape of numeric_matrix: {}'.format(numeric_matrix.shape))
            print('  shape of perc_pos_ngram_matrix: {}'.format(perc_pos_ngram_matrix.shape))
            print('  shape of categorical_matrix: {}'.format(categorical_matrix.shape))
            print('  shape of sparse_comb_matrix: {}'.format(sparse_comb_matrix.shape))
            # pylint: disable=line-too-long
            print('  shape of nozero_sparse_comb_matrix: {}'.format(nozero_sparse_comb_matrix.shape))
            print('  shape of X: {}'.format(X.shape))

        return X

    # pylint: disable=arguments-differ
    def fit(self,
            span_candidate_list: List[Dict],
            # pylint: disable=invalid-name
            y: Optional[List[bool]] = None):
        start_time = time.time()
        self.candidates_to_matrix(span_candidate_list, y=y, fit_mode=True)
        end_time = time.time()
        SentTransformer.fit_count += 1
        logger.debug("%s fit called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, SentTransformer.fit_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return self

    # pylint: disable=arguments-differ
    def transform(self,
                  span_candidate_list: List[Dict]) -> List:
        start_time = time.time()
        X_out = self.candidates_to_matrix(span_candidate_list,
                                          y=[],
                                          fit_mode=False)
        end_time = time.time()
        SentTransformer.transform_count += 1
        logger.debug("%s transform called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, SentTransformer.transform_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return X_out


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
                             *,
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
        self.candidates_to_matrix(span_candidate_list, y=y, fit_mode=True)
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
        X_out = self.candidates_to_matrix(span_candidate_list, y=[], fit_mode=False)
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
                             *,
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
        self.candidates_to_matrix(span_candidate_list, y=y, fit_mode=True)
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
        X_out = self.candidates_to_matrix(span_candidate_list, y=[], fit_mode=False)
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
                             *,
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
        self.candidates_to_matrix(span_candidate_list, y=y, fit_mode=True)
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
        X_out = self.candidates_to_matrix(span_candidate_list, y=[], fit_mode=False)
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
                             *,
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
        self.candidates_to_matrix(span_candidate_list, y=y, fit_mode=True)
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
        X_out = self.candidates_to_matrix(span_candidate_list, y=[], fit_mode=False)
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
                             *,
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
                    # print("jj table_text_alphanum: [%s]" %
                    #       table_text_alphanum.replace('\n', '~'))
                    pos_cand_text_list.append(table_text_alphanum)
            pos_words_vectorizer.fit(pos_cand_text_list)
            # print("pos_words_vectorizer.vocab = {}".format(pos_words_vectorizer.vocabulary_))
            self.pos_word_set = set(pos_words_vectorizer.vocabulary_.keys())
        # doesn't matter if min_df is 1 or 2
        pos_word_tokenizer = self.words_vectorizer.build_tokenizer()

        numeric_matrix = np.zeros(shape=(len(span_candidate_list), 36))
        for i, span_candidate in enumerate(span_candidate_list):
            # table_text = fix_rate_table_text(span_candidate['text'])
            table_text_alphanum = span_candidate['text_alphanum']
            # print("text_alphanum: [{}]".format(table_text_alphanum))
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

        # in case these have no words in pre_table_word_list or sechead_word_list
        num_pre_table_word = 0
        for words in pre_table_words_list:
            if words:
                num_pre_table_word += len(words.split())
        num_sechead_word = 0
        for words in sechead_words_list:
            if words:
                num_sechead_word += len(words.split())
        if num_pre_table_word < 4:
            pre_table_words_list = list(words_list)  # make a shallow copy
        if num_sechead_word < 4:
            sechead_words_list = list(words_list)  # make a shallow copy

        if fit_mode:
            self.words_vectorizer.fit(words_list)
            self.pre_table_vectorizer.fit(pre_table_words_list)
            self.sechead_vectorizer.fit(sechead_words_list)
            self.row_header_vectorizer.fit(row_header_words_list)
            self.min_max_scaler.fit(numeric_matrix)

            # print("----- fit:")
            # print("pretable_vocab = {}".format(self.pre_table_vectorizer.vocabulary_))
            # print("sechead_vocab = {}".format(self.sechead_vectorizer.vocabulary_))
            # print("row_header_vocab = {}".format(self.row_header_vectorizer.vocabulary_))
            # print("pos_word_list = {}".format(self.pos_word_set))
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
        self.candidates_to_matrix(span_candidate_list, y=y, fit_mode=True)
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
        X_out = self.candidates_to_matrix(span_candidate_list, y=[], fit_mode=False)
        end_time = time.time()
        TableTextTransformer.transform_count += 1
        logger.debug("%s transform called #%d, len(span_candidate_list) = %d, took %.0f msec",
                     self.name, TableTextTransformer.transform_count, len(span_candidate_list),
                     (end_time - start_time) * 1000)
        return X_out
