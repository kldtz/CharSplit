#!/usr/bin/env python

import logging

import numpy as np
from nltk import FreqDist
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing

from eblearn import igain
from eblearn import ebattr
from utils import strutils, stopwordutils

# based on http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py

# this is a class specific transformer because of information gain and
# class-specific cols_to_keep array.
class EbTransformer(BaseEstimator, TransformerMixin):

    # MAX_NUM_TOP_WORDS_IN_BAG = 25000
    MAX_NUM_TOP_WORDS_IN_BAG = 1500000
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

    def fit(self, attrvec_ebsent_list, label_list=None):
        EbTransformer.fit_count += 1
        logging.info("fitting #{} called, len(attrvec_ebsent_list) = {}, len(label_list) = {}".format(EbTransformer.fit_count,
                                                                                               len(attrvec_ebsent_list),
                                                                                               len(label_list)))
        attrvec_list = []
        ebsent_list = []
        for attrvec, ebsent in attrvec_ebsent_list:
            attrvec_list.append(attrvec)
            ebsent_list.append(ebsent)

        # ignore the result X.  The goal here is to set up the vars.
        self.ebantdoc_list_to_csr_matrix(attrvec_list,
                                         ebsent_list,
                                         label_list,
                                         fit_mode=True)
        return self

    def transform(self, attrvec_ebsent_list):
        EbTransformer.transform_count += 1
        logging.debug("transform called #{}, len(attrvec_ebsent_list) = {}".format(EbTransformer.transform_count,
                                                                           len(attrvec_ebsent_list)))

        attrvec_list = []
        ebsent_list = []
        for attrvec, ebsent in attrvec_ebsent_list:
            attrvec_list.append(attrvec)
            ebsent_list.append(ebsent)
        
        X = self.ebantdoc_list_to_csr_matrix(attrvec_list,
                                             ebsent_list,
                                             [],
                                             fit_mode=False)
        return X

    # label_list is a list of booleans
    def ebantdoc_list_to_csr_matrix(self,
                                    attrvec_list,
                                    ebsent_list,
                                    label_list,
                                    fit_mode=False):
        # prov = self.provision
        # print("attrvec_list.size = ", len(attrvec_list))
        # print("ebsent_list.size = ", len(ebsent_list))
        # print("label_list.size = ", len(label_list))
        num_rows = len(attrvec_list)

        # handle numeric_matrix and categorical_matrix
        binary_indices = ebattr.binary_indices
        numeric_indices = ebattr.numeric_indices
        categorical_indices = ebattr.categorical_indices
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
            for ebsent, label in zip(ebsent_list, label_list):
                sent_st = ebsent.get_tokens_text()
                sent_st_list.append(sent_st)
                if label:
                    positive_sent_st_list.append(sent_st)
        else:
            for ebsent in ebsent_list:
                sent_st = ebsent.get_tokens_text()
                sent_st_list.append(sent_st)
        
        nostop_sent_st_list = stopwordutils.remove_stopwords(sent_st_list, mode=2)

        if fit_mode:
            # we are cheating here because vocab is trained from both training and testing
            vocabs = igain.doc_label_list_to_vocab(sent_st_list, label_list, tokenize=igain.eb_doc_to_all_ngrams)
            vocab_id_map = {}
            for vid, vocab in enumerate(vocabs):
                vocab_id_map[vocab] = vid
            self.vocab_id_map = vocab_id_map
            
            # handling bi-topgram
            # only lower case, mode=0, label_list must not be empty
            nostop_positive_sent_st_list = stopwordutils.remove_stopwords(positive_sent_st_list, mode=0)
            filtered_list = []
            for nostop_positive_sent in nostop_positive_sent_st_list:
                for w in nostop_positive_sent.split():
                    if len(w) > 3:
                        filtered_list.append(w)
            fdistribution = FreqDist(filtered_list)
            self.n_top_positive_words = [item[0] for item in
                                         fdistribution.most_common(EbTransformer.MAX_NUM_BI_TOPGRAM_WORDS)]

        # still need to go through rest of fit_mode because of more vars are setup

        bow_matrix = self.gen_top_ig_ngram_matrix(sent_st_list, self.vocab_id_map, tokenize=igain.eb_doc_to_all_ngrams)

        # print("n_top_positive_words = {}".format(self.n_top_positive_words))
        bi_topgram_matrix = self.gen_bi_topgram_matrix(nostop_sent_st_list, fit_mode=fit_mode)

        comb_matrix = sparse.hstack((numeric_matrix, categorical_matrix, bow_matrix))
        sparse_comb_matrix = sparse.csr_matrix(comb_matrix)

        nozero_column_train_csr_matrix_part1 = self.remove_zero_column(sparse_comb_matrix, fit_mode=fit_mode)

        # print("shape of bi_topgram: ", bi_topgram_matrix.shape)
        X = sparse.hstack((nozero_column_train_csr_matrix_part1, bi_topgram_matrix), format='csr')
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

            for iw1, w1 in enumerate(found_words):
                for iw2 in range(iw1 + 1, len(found_words)):
                    w2 = found_words[iw2]
                    col_name = ",".join((w1, w2))
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
    

    def gen_top_ig_ngram_matrix(self, sent_st_list, vocab_id_map, tokenize):
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

    def remove_zero_column(self, X, fit_mode=False):
        # print("remove_zero_column(), shape of matrix X = ", X.shape)

        if fit_mode:
            col_sum = X.sum(axis=0)
            col_sum = np.squeeze(np.asarray(col_sum))
            zerofind = list(np.where(col_sum==0))
            all_cols = np.arange(X.shape[1])
            # print("zerofind= ", zerofind)

            self.cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, zerofind)))[0]

        X = X[:, self.cols_to_keep] #  remove cols where sum is zero
        # print("after remove_zero_column(), shape of matrix X = ", X.shape)
        return X
    
