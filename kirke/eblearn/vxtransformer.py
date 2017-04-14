#!/usr/bin/env python

import logging

from nltk import FreqDist
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from kirke.utils import stopwordutils, strutils

from kirke.eblearn import bag_transform, bigram_transform

# pylint: disable=C0301
# based on http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py

# this is a class specific transformer because of information gain and
# class-specific cols_to_keep array.
class VxTransformer(BaseEstimator, TransformerMixin):

    # MAX_NUM_TOP_WORDS_IN_BAG = 25000
    # MAX_NUM_TOP_WORDS_IN_BAG = 1500000
    MAX_NUM_BI_TOPGRAM_WORDS = 175

    fit_count = 0
    transform_count = 0

    """Transform a list ebantdoc to matrix."""
    def __init__(self, provision):
        # provision is needed because of infogain computation need to know the classes
        self.provision = provision

        self.bag_transform = bag_transform.BagTransform()
        self.bigram_transformer = bigram_transform.BigramTransform(self.provision)

        self.cols_to_keep = []   # used in remove_zero_columns

        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')

        self.n_top_positive_words = []
        self.vocabulary = {}  # used for bi_topgram_matrix generation
        self.vocab_id_map = {}
        self.positive_vocab = {}

    def fit(self, attrvec_list, label_list=None):
        VxTransformer.fit_count += 1
        logging.info("fitting #%s called, len(attrvec_list) = %d, len(label_list) = %d",
                     VxTransformer.fit_count, len(attrvec_list), len(label_list))

        # ignore the result X.  The goal here is to set up the vars.
        self.ebantdoc_list_to_csr_matrix(attrvec_list,
                                         label_list,
                                         fit_mode=True)
        return self

    def transform(self, attrvec_list):
        VxTransformer.transform_count += 1
        logging.debug("transform called #%d, len(attrvec_list) = %d",
                      VxTransformer.transform_count, len(attrvec_list))

        # pylint: disable=C0103
        X = self.ebantdoc_list_to_csr_matrix(attrvec_list,
                                             [],
                                             fit_mode=False)
        return X

    # label_list is a list of booleans
    # pylint: disable=R0912, R0914
    def ebantdoc_list_to_csr_matrix(self,
                                    attrvec_list,
                                    label_list,
                                    fit_mode=False):
        # prov = self.provision
        # print("attrvec_list.size = ", len(attrvec_list))
        # print("label_list.size = ", len(label_list))
        num_rows = len(attrvec_list)

        print('Prepping generic train data' + self.provision)
        bag_matrix, Ys = self.bag_transform.storeBagMatrix(attrvec_list, label_list)

        bigram_matrix = self.bigram_transformer.storeBigramMatrix(attrvec_list)
        
        if fit_mode:
            # we are cheating here because vocab is trained from both training and testing
            # jshaw, TODO, remove
            logging.info("starting computing info_gain")
            # igain_vocabs = igain.doc_label_list_to_vocab(sent_st_list, label_list, tokenize=igain.eb_doc_to_all_ngrams, debug_mode=True, provision=self.provision)
            # the tokenizer is bigramutils, not igain's
            igain_vocabs = igain.doc_label_list_to_vocab(sent_st_list,
                                                         label_list,
                                                         tokenize=bigramutils.eb_doc_to_all_ngrams,
                                                         debug_mode=True,
                                                         provision=self.provision)

            logging.info("starting computing unigram and bigram")
            vocabs, positive_vocabs = bigramutils.doc_label_list_to_vocab(sent_st_list,
                                                                          label_list,
                                                                          tokenize=bigramutils.eb_doc_to_all_ngrams)
            # replace vocabs with igain.vocab
            vocab = igain_vocabs
            vocab_id_map = {}
            for vid, vocab in enumerate(vocabs):
                vocab_id_map[vocab] = vid
            self.vocab_id_map = vocab_id_map
            self.positive_vocabs = positive_vocabs

            with open("{}_vocabs.tsv".format(self.provision), "wt") as fvcabout:
                for vocab in vocabs:
                    print(vocab, file=fvcabout)

            with open("{}_posvocabs.tsv".format(self.provision), "wt") as fvcabout:
                for vocab in positive_vocabs:
                    print(vocab, file=fvcabout)                    

            # handling bi-topgram
            # only lower case, mode=0, label_list must not be empty
            logging.info("starting computing bi_topgram")
            nostop_positive_sent_st_list = stopwordutils.remove_stopwords(positive_sent_st_list, mode=0)
            filtered_list = []
            for nostop_positive_sent in nostop_positive_sent_st_list:
                for tmp_w in nostop_positive_sent.split():
                    if len(tmp_w) > 3:
                        filtered_list.append(tmp_w)
            fdistribution = FreqDist(filtered_list)
            self.n_top_positive_words = [item[0] for item in
                                         fdistribution.most_common(EbTransformer.MAX_NUM_BI_TOPGRAM_WORDS)]

            # replace this top positive word with top most informative words
            # self.n_top_positive_words = top_igain_unigrams
            # after some thinking, those words can be either 'for' or 'against' the provision,
            # and not always better, better simply leave them as before

        # still need to go through rest of fit_mode because of more vars are setup

        logging.info("converting into matrix")
        # bow_matrix = self.gen_top_ig_ngram_matrix(sent_st_list, tokenize=igain.eb_doc_to_all_ngrams)
        bow_matrix, perc_positive_ngrams = self.gen_top_ngram_matrix(sent_st_list,
                                                                     tokenize=bigramutils.eb_doc_to_all_ngrams)

        # print("n_top_positive_words = {}".format(self.n_top_positive_words))
        bi_topgram_matrix = self.gen_bi_topgram_matrix(nostop_sent_st_list, fit_mode=fit_mode)

        # put together my perc_positive_matrix
        perc_pos_ngram_matrix = np.zeros(shape=(num_rows, 1))
        for instance_i, perc_pos_ngram in enumerate(perc_positive_ngrams):
            perc_pos_ngram_matrix[instance_i, 0] = perc_pos_ngram

        comb_matrix = sparse.hstack((numeric_matrix, perc_pos_ngram_matrix, categorical_matrix, bow_matrix))
        sparse_comb_matrix = sparse.csr_matrix(comb_matrix)

        nozero_sparse_comb_matrix = self.remove_zero_column(sparse_comb_matrix, fit_mode=fit_mode)

        # print("shape of bi_topgram: ", bi_topgram_matrix.shape)
        # pylint: disable=C0103
        X = sparse.hstack((nozero_sparse_comb_matrix, bi_topgram_matrix), format='csr')
        # print("combined shape of X = {}".format(X.shape))
        # print("shape of X: {}", X)

        # return sparse_comb_matrix, bi_topgram_matrix, sent_st_list
        return X


    # pylint: disable=C0103
    def remove_zero_column(self, X, fit_mode=False):
        # print("remove_zero_column(), shape of matrix X = ", X.shape)

        if fit_mode:
            col_sum = X.sum(axis=0)
            col_sum = np.squeeze(np.asarray(col_sum))
            zerofind = list(np.where(col_sum == 0))
            all_cols = np.arange(X.shape[1])
            # print("zerofind= ", zerofind)

            # pylint: disable=E1101
            self.cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, zerofind)))[0]

        X = X[:, self.cols_to_keep] #  remove cols where sum is zero
        # print("after remove_zero_column(), shape of matrix X = ", X.shape)
        return X
