#!/usr/bin/env python

import logging

from nltk import FreqDist
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from kirke.eblearn import igain, ebattrvec, bigramutils
from kirke.utils import stopwordutils, strutils




# pylint: disable=C0301
# based on http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py

# this is a class specific transformer because of information gain and
# class-specific cols_to_keep array.
class EbTransformer(BaseEstimator, TransformerMixin):

    # MAX_NUM_TOP_WORDS_IN_BAG = 25000
    # MAX_NUM_TOP_WORDS_IN_BAG = 1500000
    MAX_NUM_BI_TOPGRAM_WORDS = 175

    fit_count = 0
    transform_count = 0

    """Transform a list ebantdoc to matrix."""
    def __init__(self, provision):
        # provision is needed because of infogain computation need to know the classes
        self.provision = provision
        self.cols_to_keep = []   # used in remove_zero_columns

        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')

        self.n_top_positive_words = []
        self.vocabulary = {}  # used for bi_topgram_matrix generation
        self.vocab_id_map = {}
        self.positive_vocab = {}

    def fit(self, attrvec_list, label_list=None):
        EbTransformer.fit_count += 1
        num_pos_inst = 0
        for label in label_list:
            if label:
                num_pos_inst += 1
        logging.info("fitting #%s called, len(attrvec_list) = %d, len(label_list) = %d, num_pos = %d",
                     EbTransformer.fit_count, len(attrvec_list), len(label_list), num_pos_inst)
        

        # ignore the result X.  The goal here is to set up the vars.
        self.ebantdoc_list_to_csr_matrix(attrvec_list,
                                         label_list,
                                         fit_mode=True)
        return self

    def transform(self, attrvec_list):
        EbTransformer.transform_count += 1
        logging.debug("transform called #%d, len(attrvec_list) = %d",
                      EbTransformer.transform_count, len(attrvec_list))

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

        # handle numeric_matrix and categorical_matrix
        binary_indices = ebattrvec.BINARY_INDICES
        numeric_indices = ebattrvec.NUMERIC_INDICES
        categorical_indices = ebattrvec.CATEGORICAL_INDICES
        numeric_matrix = np.zeros(shape=(num_rows, len(binary_indices) + len(numeric_indices)))
        categorical_matrix = np.zeros(shape=(num_rows, len(categorical_indices)))
        for instance_i, attrvec in enumerate(attrvec_list):
            for ibin, binary_index in enumerate(binary_indices):
                numeric_matrix[instance_i, ibin] = strutils.bool_to_int(attrvec[binary_index])
            for inum, numeric_index in enumerate(numeric_indices):
                numeric_matrix[instance_i, len(binary_indices) + inum] = attrvec[numeric_index]
            for icat, cat_index in enumerate(categorical_indices):
                categorical_matrix[instance_i, icat] = attrvec[cat_index]
        if fit_mode:
            numeric_matrix = self.min_max_scaler.fit_transform(numeric_matrix)
            categorical_matrix = self.one_hot_encoder.fit_transform(categorical_matrix)
        else:
            numeric_matrix = self.min_max_scaler.transform(numeric_matrix)
            categorical_matrix = self.one_hot_encoder.transform(categorical_matrix)

        # handle bag of words
        sent_st_list = []
        positive_sent_st_list = []  # only populated if fit_mode
        if fit_mode:  # label_list:  # for testing, there is no label_list
            for attrvec, label in zip(attrvec_list, label_list):
                sent_st = attrvec[ebattrvec.TOKENS_TEXT_INDEX]
                sent_st_list.append(sent_st)
                if label:
                    positive_sent_st_list.append(sent_st)
        else:
            for attrvec in attrvec_list:
                sent_st = attrvec[ebattrvec.TOKENS_TEXT_INDEX]
                sent_st_list.append(sent_st)

        nostop_sent_st_list = stopwordutils.remove_stopwords(sent_st_list, mode=2)

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

        # logging.info("converting into matrix")
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

    def gen_bi_topgram_matrix(self, sent_st_list, fit_mode=False):
        # print("len(sent_st_list)= {}".format(len(sent_st_list)))
        # for each sentence, find which top words it contains.  Then generate all pairs of these,
        # and generate the sparse matrix row entries for the rows it contains.
        indptr = [0]
        indices = []
        data = []
        for sent_st in sent_st_list:
            sent_words = set(sent_st.split())   # TODO, a little repetitive, split again
            found_words = [common_word for common_word in self.n_top_positive_words if common_word in sent_words]

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