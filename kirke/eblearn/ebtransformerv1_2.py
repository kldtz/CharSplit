#!/usr/bin/env python3

import logging
# pylint: disable=unused-import
from typing import Dict, List, Set

from nltk import FreqDist
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

from kirke.eblearn import ebattrvec
from kirke.eblearn import igain, bigramutils
from kirke.eblearn.ebtransformerbase import EbTransformerBase
from kirke.utils import stopwordutils, strutils


# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEBUG_MODE = False

PROVISION_ATTRLISTS_MAP = {'party': (ebattrvec.PARTY_BINARY_ATTR_LIST,
                                     ebattrvec.PARTY_NUMERIC_ATTR_LIST,
                                     ebattrvec.PARTY_CATEGORICAL_ATTR_LIST),
                           'default': (ebattrvec.DEFAULT_BINARY_ATTR_LIST,
                                       ebattrvec.DEFAULT_NUMERIC_ATTR_LIST,
                                       ebattrvec.DEFAULT_CATEGORICAL_ATTR_LIST)}

def get_transformer_attr_list_by_provision(provision: str):
    if PROVISION_ATTRLISTS_MAP.get(provision):
        return PROVISION_ATTRLISTS_MAP.get(provision)
    return PROVISION_ATTRLISTS_MAP.get('default')


# pylint: disable=C0301
# based on http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py

# this is a class specific transformer because of information gain and
# class-specific cols_to_keep array.
# pylint: disable=too-many-instance-attributes
class EbTransformerV1_2(EbTransformerBase):

    # MAX_NUM_TOP_WORDS_IN_BAG = 25000
    # MAX_NUM_TOP_WORDS_IN_BAG = 1500000
    MAX_NUM_BI_TOPGRAM_WORDS = 175

    """Transform a list ebantdoc to matrix."""
    def __init__(self, provision: str) -> None:
        # provision is needed because of infogain computation need to know the classes
        super().__init__(provision)
        self.version = '1.2'

        logger.info('EbTransformerV1_2(%s)', self.provision)
        (binary_attr_list, numeric_attr_list, categorical_attr_list) = \
                get_transformer_attr_list_by_provision(self.provision)

        # self.provision = provision
        self.binary_attr_list = binary_attr_list
        self.numeric_attr_list = numeric_attr_list
        self.categorical_attr_list = categorical_attr_list

        self.cols_to_keep = []   # used in remove_zero_columns

        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')

        self.n_top_positive_words = []  # type: List[str]
        self.vocab_id_map = {}  # type: Dict[str, int]
        self.positive_vocabs = set([])  # type: Set[str]

        # used for bi_topgram_matrix generation
        self.vocabulary = {}  # type: Dict[str, int]

        # handling sechead, with min appearance in sentence = 5
        # now changed to 2 because custom training corpus might have only 6 docs
        self.sechead_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2))

    # label_list is a list of booleans
    # pylint: disable=too-many-statements, too-many-locals, too-many-branches
    def ebantdoc_list_to_csr_matrix(self,
                                    attrvec_list,
                                    label_list,
                                    fit_mode=False):
        # prov = self.provision
        # print("attrvec_list.size = ", len(attrvec_list))
        # print("label_list.size = ", len(label_list))
        num_rows = len(attrvec_list)

        # handle numeric_matrix and categorical_matrix
        numeric_matrix = np.zeros(shape=(num_rows,
                                         len(self.binary_attr_list) +
                                         len(self.numeric_attr_list)))
        categorical_matrix = np.zeros(shape=(num_rows,
                                             len(self.categorical_attr_list)))
        for instance_i, attrvec in enumerate(attrvec_list):
            for ibin, binary_attr in enumerate(self.binary_attr_list):
                numeric_matrix[instance_i, ibin] = strutils.bool_to_int(attrvec.get_val(binary_attr))
            for inum, numeric_attr in enumerate(self.numeric_attr_list):
                numeric_matrix[instance_i, len(self.binary_attr_list) + inum] = attrvec.get_val(numeric_attr)
            for icat, cat_attr in enumerate(self.categorical_attr_list):
                categorical_matrix[instance_i, icat] = attrvec.get_val(cat_attr)
        if fit_mode:
            numeric_matrix = self.min_max_scaler.fit_transform(numeric_matrix)
            categorical_matrix = self.one_hot_encoder.fit_transform(categorical_matrix)
        else:
            numeric_matrix = self.min_max_scaler.transform(numeric_matrix)
            categorical_matrix = self.one_hot_encoder.transform(categorical_matrix)

        # handle bag of words
        sent_st_list = []
        positive_sent_st_list = []  # only populated if fit_mode
        sechead_st_list = []
        if fit_mode:  # label_list:  # for testing, there is no label_list
            for attrvec, label in zip(attrvec_list, label_list):
                sent_st = attrvec.bag_of_words

                # if label:
                #     print("++++++++++ pos sent_st: {}".format(sent_st))
                # else:
                #     print("neg sent_st: {}".format(sent_st))

                sent_st_list.append(sent_st)
                if label:
                    positive_sent_st_list.append(sent_st)
                sechead_st_list.append(attrvec.sechead)
        else:
            for attrvec in attrvec_list:
                sent_st = attrvec.bag_of_words
                sent_st_list.append(sent_st)
                sechead_st_list.append(attrvec.sechead)

        nostop_sent_st_list = stopwordutils.remove_stopwords(sent_st_list, mode=2)
        is_debug = False

        if fit_mode:
            # we are cheating here because vocab is trained from both training and testing
            # jshaw, TODO, remove
            logger.info("starting computing info_gain")
            # the tokenizer is bigramutils, not igain's
            igain_vocabs = igain.doc_label_list_to_vocab(sent_st_list,
                                                         label_list,
                                                         tokenize=bigramutils.eb_doc_to_all_ngrams,
                                                         provision=self.provision)

            logger.info("starting computing unigram and bigram")
            unused_vocabs, positive_vocabs = \
                bigramutils.doc_label_list_to_vocab(sent_st_list,
                                                    label_list,
                                                    tokenize=bigramutils.eb_doc_to_all_ngrams)
            # replace vocabs with igain.vocab
            vocab_id_map = {}
            for vid, vocab in enumerate(igain_vocabs):
                vocab_id_map[vocab] = vid
            self.vocab_id_map = vocab_id_map
            self.positive_vocabs = positive_vocabs

            if is_debug:
                with open("/tmp/{}_vocabs.tsv".format(self.provision), "wt") as fvcabout:
                    for vocab in igain_vocabs:
                        print(vocab, file=fvcabout)
                print('wrote {}'.format("/tmp/{}_vocabs.tsv".format(self.provision)))

                with open("/tmp/{}_posvocabs.tsv".format(self.provision), "wt") as fvcabout:
                    for vocab in sorted(positive_vocabs):
                        print(vocab, file=fvcabout)
                print('wrote {}'.format("/tmp/{}_posvocabs.tsv".format(self.provision)))

            # handling sechead, with min appearance in sentence = 5
            # now changed to 2 because custom training corpus might have only 6 docs
            # Before 08/15/2018, self.sechead_vectorizer is not in _init_(),
            # so do here as before, just in case
            self.sechead_vectorizer = CountVectorizer(min_df=2, ngram_range=(1, 2))
            # If there is no section head vocab due to some reason, such as a very small
            # custom training corpus, simply add a dummpy vocab.
            try:
                self.sechead_vectorizer.fit(sechead_st_list)
            except ValueError:
                self.sechead_vectorizer = CountVectorizer(vocabulary=['dummy'])

            # handling bi-topgram
            # only lower case, mode=0, label_list must not be empty
            logger.info("starting computing bi_topgram")
            nostop_positive_sent_st_list = stopwordutils.remove_stopwords(positive_sent_st_list, mode=0)
            filtered_list = []
            for nostop_positive_sent in nostop_positive_sent_st_list:
                for tmp_w in nostop_positive_sent.split():
                    if len(tmp_w) > 3:
                        filtered_list.append(tmp_w)

            # The words in FreqDist at the same frequency is unordered.
            # To make the classification result consistent, take the wanted
            # frequency and perform cut-off based on that frequency.
            MAX_NUM_BI_TOPGRAM_WORDS = EbTransformerV1_2.MAX_NUM_BI_TOPGRAM_WORDS
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

            # replace this top positive word with top most informative words
            # self.n_top_positive_words = top_igain_unigrams
            # after some thinking, those words can be either 'for' or 'against' the provision,
            # and not always better, better simply leave them as before

        # still need to go through rest of fit_mode because of more vars are setup

        # logger.info("converting into matrix")
        # bow_matrix = self.gen_top_ig_ngram_matrix(sent_st_list, tokenize=igain.eb_doc_to_all_ngrams)
        bow_matrix, perc_positive_ngrams = self.gen_top_ngram_matrix(sent_st_list,
                                                                     tokenize=bigramutils.eb_doc_to_all_ngrams)
        sechead_matrix = self.sechead_vectorizer.transform(sechead_st_list)

        # print("n_top_positive_words = {}".format(self.n_top_positive_words))
        bi_topgram_matrix = self.gen_bi_topgram_matrix(nostop_sent_st_list, fit_mode=fit_mode)

        # put together my perc_positive_matrix
        perc_pos_ngram_matrix = np.zeros(shape=(num_rows, 1))
        for instance_i, perc_pos_ngram in enumerate(perc_positive_ngrams):
            perc_pos_ngram_matrix[instance_i, 0] = perc_pos_ngram

        comb_matrix = sparse.hstack((numeric_matrix, perc_pos_ngram_matrix, categorical_matrix, bow_matrix,
                                     sechead_matrix))
        sparse_comb_matrix = sparse.csr_matrix(comb_matrix)
        nozero_sparse_comb_matrix = self.remove_zero_column(sparse_comb_matrix, fit_mode=fit_mode)
        X = sparse.hstack((nozero_sparse_comb_matrix, bi_topgram_matrix), format='csr')

        if is_debug:
            print("  shape of bow_matrix: {}".format(bow_matrix.shape))
            print("  shape of sechead_matrix: {}".format(sechead_matrix.shape))
            print("  shape of bi_topgram_matrix: {}".format(bi_topgram_matrix.shape))
            print("  shape of numeric_matrix: {}".format(numeric_matrix.shape))
            print('  shape of perc_pos_ngram_matrix: {}'.format(perc_pos_ngram_matrix.shape))
            print('  shape of categorical_matrix: {}'.format(categorical_matrix.shape))
            print('  shape of sparse_comb_matrix: {}'.format(sparse_comb_matrix.shape))
            print('  shape of nozero_sparse_comb_matrix: {}'.format(nozero_sparse_comb_matrix.shape))
            print("  shape of X: {}".format(X.shape))

        # return sparse_comb_matrix, bi_topgram_matrix, sent_st_list
        return X

    # pylint: disable=too-many-locals
    def gen_bi_topgram_matrix(self, sent_st_list, fit_mode=False):
        # print("len(sent_st_list)= {}".format(len(sent_st_list)))
        # for each sentence, find which top words it contains.  Then generate all pairs of these,
        # and generate the sparse matrix row entries for the rows it contains.
        indptr = [0]
        indices = []
        data = []
        for sent_st in sent_st_list:
            sent_words = set(sent_st.split())   # TODO, a little repetitive, split again
            found_words = [common_word for common_word
                           in self.n_top_positive_words if common_word in sent_words]

            for index_w1, tmp_w1 in enumerate(found_words):
                for index_w2 in range(index_w1 + 1, len(found_words)):
                    tmp_w2 = found_words[index_w2]
                    col_name = ",".join((tmp_w1, tmp_w2))
                    if fit_mode:
                        # print("col_name= {}".format(col_name))
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

    """
    def gen_top_ig_ngram_matrix(self, sent_st_list, tokenize):
        # for each sentence, find which top words it contains.  Then generate all pairs of these,
        # and generate the sparse matrix row entries for the rows it contains.
        indptr = [0]
        indices = []
        data = []
        for sent_st in sent_st_list:
            sent_wordset = tokenize(sent_st)

            for ngram in sent_wordset:
                index = self.vocab_id_map.get(ngram)
                if index:
                    indices.append(index)
                    data.append(1)
            indptr.append(len(indices))

        top_ig_ngram_matrix = sparse.csr_matrix((data, indices, indptr),
                                                shape=(len(sent_st_list), len(self.vocab_id_map)),
                                                dtype=int)
        return top_ig_ngram_matrix
    """

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
                if ngram in self.positive_vocabs:
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
