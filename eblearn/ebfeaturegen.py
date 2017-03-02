import logging

from nltk import FreqDist

import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

from eblearn import igain
from eblearn import ebattr
from utils import strutils, stopwordutils

# Because each classifier might have different columns being zero, there is a need to keep track of
# which column is removed and pass that information out.
def remove_zero_column(X, cols_to_keep, train_mode=False):
    # print("remove_zero_column(), shape of matrix X = ", X.shape)

    if train_mode:
        col_sum = X.sum(axis=0)
        col_sum = np.squeeze(np.asarray(col_sum))
        zerofind = list(np.where(col_sum==0))
        all_cols = np.arange(X.shape[1])
        # print("zerofind= ", zerofind)

        cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, zerofind)))[0]

    X = X[:, cols_to_keep] #  remove cols where sum is zero
    # print("after remove_zero_column(), shape of matrix X = ", X.shape)
    return X, cols_to_keep



class EbFeatureGenerator:

    # MAX_NUM_TOP_WORDS_IN_BAG = 25000
    MAX_NUM_TOP_WORDS_IN_BAG = 1500000
    MAX_NUM_BI_TOPGRAM_WORDS = 175

    def __init__(self):
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
        self.vectorizer = CountVectorizer(max_features=EbFeatureGenerator.MAX_NUM_TOP_WORDS_IN_BAG,
                                          # stop_words='english',
                                          stop_words=None,
                                          lowercase=False,
                                          ngram_range=(1, 3),
                                          # ngram_range=(1, 2),
                                          token_pattern=r'\b\w+\b')
        self.stopwords = self.vectorizer.get_stop_words()
        self.n_top_positive_words = []
        self.vocabulary = {}  # used for bi_topgram_matrix generation
        self.vocab_id_map = {}

    # label_list is a list of booleans
    def attrvecs_to_csr_matrix(self, attrvec_list, ebsent_list,
                               prov, label_list, classifier_cols_to_keep, feature_dir, train_mode=False):
        num_rows = len(attrvec_list)
        trtest_mode_st = 'train' if train_mode else 'test'

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
        if train_mode:
            numeric_matrix = self.min_max_scaler.fit_transform(numeric_matrix)
            categorical_matrix = self.one_hot_encoder.fit_transform(categorical_matrix)
        else:
            numeric_matrix = self.min_max_scaler.transform(numeric_matrix)
            categorical_matrix = self.one_hot_encoder.transform(categorical_matrix)

        # handle bag of words
        sent_st_list = []
        positive_sent_st_list = []
        if label_list:  # for testing, there is no label_list
            for ebsent, label in zip(ebsent_list, label_list):
                sent_st = ebsent.get_tokens_text()
                sent_st_list.append(sent_st)
                if label:
                    positive_sent_st_list.append(sent_st)
        else:
            for ebsent in ebsent_list:
                sent_st = ebsent.get_tokens_text()
                sent_st_list.append(sent_st)
        
        is_save_sent = True
        if is_save_sent:
            strutils.save_str_list(sent_st_list,
                                   '{}/{}.sent.{}.v1.txt'.format(feature_dir,
                                                                 prov,
                                                                 trtest_mode_st))
            if label_list:  # test mode, no positive sentences needed
                strutils.save_str_list(positive_sent_st_list,
                                       '{}/{}.sent.pos.v1.txt'.format(feature_dir,
                                                                      prov))

        nostop_sent_st_list = stopwordutils.remove_stopwords(sent_st_list, mode=2)
        strutils.save_str_list(nostop_sent_st_list,
                               '{}/{}.sent.nostop.{}.x1.txt'.format(feature_dir,
                                                                    prov,
                                                                    trtest_mode_st))

        if train_mode:
            # we are cheating here because vocab is trained from both training and testing
            vocabs = igain.doc_label_list_to_vocab(sent_st_list, label_list, tokenize=igain.eb_doc_to_all_ngrams)
            vocab_id_map = {}
            for vid, vocab in enumerate(vocabs):
                vocab_id_map[vocab] = vid
            self.vocab_id_map = vocab_id_map
            
        bow_matrix = self.gen_top_ig_ngram_matrix(sent_st_list, self.vocab_id_map, tokenize=igain.eb_doc_to_all_ngrams)

        # handling bi-topgram
        # only lower case, mode=0
        if label_list:  # training mode
            nostop_positive_sent_st_list = stopwordutils.remove_stopwords(positive_sent_st_list, mode=0)
            filtered_list = []
            for nostop_positive_sent in nostop_positive_sent_st_list:
                for w in nostop_positive_sent.split():
                    if len(w) > 3:
                        filtered_list.append(w)
            fdistribution = FreqDist(filtered_list)
            self.n_top_positive_words = [item[0] for item in
                                         fdistribution.most_common(EbFeatureGenerator.MAX_NUM_BI_TOPGRAM_WORDS)]

        # print("n_top_positive_words = {}".format(self.n_top_positive_words))
        bi_topgram_matrix = self.gen_bi_topgram_matrix(nostop_sent_st_list, train_mode=train_mode)

        comb_matrix = sparse.hstack((numeric_matrix, categorical_matrix, bow_matrix))
        sparse_comb_matrix = sparse.csr_matrix(comb_matrix)

        if train_mode:
            logging.info("{} top ig ngram size= {}".format(prov, len(self.vocab_id_map)))
            with open(feature_dir + "/{}.top_ig_ngram.txt".format(prov), 'wt') as vocab_out:
                for vid, vocab in sorted([(vid, ngram) for (ngram, vid) in self.vocab_id_map.items()]):
                    print(vocab, file=vocab_out)

            with open(feature_dir + "/{}.train_label_sents.txt".format(prov), "wt") as train_txt_out:
                for sent_label, sent_st in zip(label_list, sent_st_list):
                    print(sent_label, sent_st, sep='\t', file=train_txt_out)

        nozero_column_train_csr_matrix_part1, cols_to_keep  = remove_zero_column(sparse_comb_matrix,
                                                                                 classifier_cols_to_keep,
                                                                                 train_mode=train_mode)
        # print("shape of bi_topgram: ", bi_topgram_matrix.shape)
        X = sparse.hstack((nozero_column_train_csr_matrix_part1, bi_topgram_matrix), format='csr')
        # print("combined shape of X = {}".format(X.shape))

        # return sparse_comb_matrix, bi_topgram_matrix, sent_st_list
        return X, cols_to_keep

    
    def gen_bi_topgram_matrix(self, sent_st_list, train_mode=False):
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
                    if train_mode:
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

        if train_mode:
            bi_topgram_matrix = sparse.csr_matrix((data, indices, indptr), dtype=int)
        else:
            bi_topgram_matrix = sparse.csr_matrix((data, indices, indptr),
                                                  shape=(len(sent_st_list), len(self.vocabulary)),
                                                  dtype=int)            
        return bi_topgram_matrix


    def gen_top_ig_ngram_matrix(self, sent_st_list, vocab_id_map, tokenize, train_mode=False):
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
    
