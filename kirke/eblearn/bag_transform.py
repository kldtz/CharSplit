from scipy import sparse
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from kirke.eblearn import ebattrvec
from kirke.utils import strutils

# import ebrevia.learn.util as util


# truncate the following features to avoid outlier issues
ENT_START_MAX = 50000
NTH_CANDIDATE_MAX = 500
NUM_CHARS_MAX = 300
NUM_WORDS_MAX = 50


class BagTransform:

    def __init__(self):
        self.cols_to_keep = []   # used in remove_zero_columns
        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.one_hot_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
        self.vectorizer = None

    def storeBagMatrix(self, attrvec_list):
        return self._storeBagMatrix(attrvec_list, testmode=False)

    def storeBagMatrixTest(self, attrvec_list):
        return self._storeBagMatrix(attrvec_list, testmode=True)

    def _storeBagMatrix(self, attrvec_list, testmode):
        print("start store bag matrix")
        number_of_top_words = 25000

        # these needs to be adjusted
        # set_col = [0, 1, 2, 3, 8]
        # categorical_col = list(range(9, 17))
        # numeric_col = [5, 6, 7]
        # numeric_col.extend(list(range(17, 23)))
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
        if not testmode:
            numeric_matrix = self.min_max_scaler.fit_transform(numeric_matrix)
            categorical_matrix = self.one_hot_encoder.fit_transform(categorical_matrix)
        else:
            numeric_matrix = self.min_max_scaler.transform(numeric_matrix)
            categorical_matrix = self.one_hot_encoder.transform(categorical_matrix)

        # count = len(attrvec_list)

        # cols = len(set_col) + len(numeric_col)
        # yes_no_matrix = numpy.zeros(shape=(count, cols))  # turn yes and no in (1,0) numpy mat
        # categorical_matrix = numpy.zeros(shape=(count, len(categorical_col)))
        # jshaw, yes_no_matrix == numeric_matrix??

        sentences = []
        for attrvec in attrvec_list:
            sentence = attrvec[ebattrvec.TOKENS_TEXT_INDEX].replace('\u201c', '"')
            sentence = sentence.replace('\u201d', '"')
            sentence = sentence.replace('\u2019', "'")
            sentences.append(sentence)

        if (not testmode):
            # self.vectorizer = CountVectorizer(max_features=number_of_top_words,
            #                                   stop_words='english',lowercase=False)
            self.vectorizer = CountVectorizer(max_features=number_of_top_words,
                                              stop_words='english',
                                              lowercase=False,
                                              ngram_range=(1, 2),
                                              token_pattern=r'\b\w+\b')

            bow_matrix = self.vectorizer.fit_transform(sentences)
            # make pickling smaller (this attr is only needed for introspection)
            self.vectorizer.stop_words_ = None
        else:
            bow_matrix = self.vectorizer.transform(sentences)

        bag_matrix = sparse.hstack((numeric_matrix, categorical_matrix, bow_matrix))

        ##############################
        ##########  write matrix,lists
        ##############################
        bag_matrix = sparse.csr_matrix(bag_matrix)

        return bag_matrix


