#!/usr/bin/env python3

import logging
import time

from nltk import FreqDist
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

from kirke.eblearn import igain, bigramutils
from kirke.utils import stopwordutils, strutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEBUG_MODE = False

class EbTransformerBase(BaseEstimator, TransformerMixin):

    fit_count = 0
    transform_count = 0

    def __init__(self, provision):
        self.provision = provision
        self.num_instances = 0
        self.num_pos_instances = 0
        self.num_neg_instances = 0
        

    def fit(self, attrvec_list, label_list=None):
        self.num_pos_instances = 0
        for label in label_list:
            if label:
                self.num_pos_instances += 1
        logger.info("fitting #%s called, len(attrvec_list) = %d, len(label_list) = %d, num_pos = %d",
                    EbTransformerBase.fit_count,
                    len(attrvec_list),
                    len(label_list),
                    self.num_pos_instances)
        

        start_time = time.time()        
        # ignore the result X.  The goal here is to set up the vars.
        self.ebantdoc_list_to_csr_matrix(attrvec_list,
                                         label_list,
                                         fit_mode=True)
        end_time = time.time()
        logger.debug("%s fit called #%d, len(attrvec_list) = %d, took %.0f msec",
                     self.provision,
                     EbTransformerBase.fit_count,
                     len(attrvec_list),
                     (end_time - start_time) * 1000)
        EbTransformerBase.fit_count += 1
        return self

    
    def transform(self, attrvec_list):
        # pylint: disable=C0103
        start_time = time.time()
        X = self.ebantdoc_list_to_csr_matrix(attrvec_list,
                                             [],
                                             fit_mode=False)
        end_time = time.time()
        EbTransformerBase.transform_count += 1
        logger.debug("%s transform called #%d, len(attrvec_list) = %d, took %.0f msec",
                     self.provision,
                     EbTransformerBase.transform_count,
                     len(attrvec_list),
                     (end_time - start_time) * 1000)

        return X

    # label_list is a list of booleans
    # pylint: disable=R0912, R0914
    def ebantdoc_list_to_csr_matrix(self,
                                    attrvec_list,
                                    label_list,
                                    fit_mode=False):
        pass

    
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
    
