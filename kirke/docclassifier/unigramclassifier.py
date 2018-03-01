
import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing

from kirke.docclassifier import doccatutils


class DocClassifier(ABC):

    def __init__(self):
        self.pred_status = {}

    def get_pred_status(self):
        return self.pred_status

    def save(self, model_file_name):
        logging.info("saving model file: %s", model_file_name)
        joblib.dump(self, model_file_name)

    @abstractmethod
    def train(self, txt_fn_list_fn, model_file_name, catnames=None):
        pass

    @abstractmethod
    def predict(self, doc_text, catnames=None):
        pass

    @abstractmethod
    def train_and_evaluate(self, txt_fn_list_fn, catnames=None):
        pass


class UnigramDocClassifier(DocClassifier):

    def __init__(self):
        DocClassifier.__init__(self)
        self.classifier = None
        self.transformer = None

        self.catname_list = []
        self.catname_catid_map = {}
        # self.catid_catname_map = {}
        self.valid_tags = set([])

    def train(self, txt_fn_list_fn, model_file_name):

        logging.info('start training unigram document classifier: [%s]', txt_fn_list_fn)

        # instead of using all valid tags, we just want tags with f1 >= 0.75
        # doccatutils.load_doccat_maps('dict/doccat.valid.count.tsv')
        self.catname_list, self.catname_catid_map = \
            doccatutils.load_doccat_prod_maps('dict/doccat.prod.tsv')

        self.valid_tags = set(self.catname_list)

        # each y is a list of catid
        # pylint: disable=invalid-name
        X_both, y_both = doccatutils.load_data(txt_fn_list_fn, self.catname_catid_map, self.valid_tags)

        X_both = np.asarray(X_both)
        y_both = np.asarray(y_both)

        mlbinarizer = preprocessing.MultiLabelBinarizer()
        bin_y_both = mlbinarizer.fit_transform(y_both)

        # use all data for training
        X_train = X_both
        y_train = bin_y_both

        tf_vectorizer = TfidfVectorizer(min_df=2)
        self.transformer = tf_vectorizer.fit(X_train)

        train_ngram = tf_vectorizer.transform(X_train)
        # loss='hinge', but prodct_prob not happy with it
        # Tried penalty='l2', but result worse 88 -> 85
        sgd = OneVsRestClassifier(SGDClassifier(loss='log', penalty='l1'))
        sgd.fit(train_ngram, y_train)

        self.classifier = sgd
        self.save(model_file_name)
        print("wrote '{}'".format(model_file_name))
        return sgd

    # pylint: disable=too-many-locals
    def train_and_evaluate(self, txt_fn_list_fn, prod_status_fname=None):
        logging.info('start training and evaluate unigram document classifier: [%s]',
                     txt_fn_list_fn)

        # pylint: disable=invalid-name
        EB_DOC_KFOLD = 3

        self.catname_list, self.catname_catid_map = \
           doccatutils.load_doccat_maps('dict/doccat.valid.count.tsv')
        self.valid_tags = set(self.catname_list)

        # each y is a list of catid
        # pylint: disable=invalid-name
        X_both, y_both = doccatutils.load_data(txt_fn_list_fn, self.catname_catid_map, self.valid_tags)

        X_both = np.asarray(X_both)
        y_both = np.asarray(y_both)

        mlbinarizer = preprocessing.MultiLabelBinarizer()
        bin_y_both = mlbinarizer.fit_transform(y_both)

        kfolds = KFold(n_splits=EB_DOC_KFOLD, shuffle=True)

        prec_list, recall_list, f1_list = [], [], []
        report_list = []  # to be combined
        for train_index, test_index in kfolds.split(X_both, bin_y_both):

            X_train = X_both[train_index]
            y_train = bin_y_both[train_index]

            X_test = X_both[test_index]
            y_test = bin_y_both[test_index]

            tf_vectorizer = TfidfVectorizer(min_df=2)
            tf_vectorizer.fit(X_train)

            train_ngram = tf_vectorizer.transform(X_train)
            # sgd = SGDClassifier(loss="hinge", penalty='l1')
            sgd = OneVsRestClassifier(SGDClassifier(loss='log', penalty='l1'))
            sgd.fit(train_ngram, y_train)

            test_ngram = tf_vectorizer.transform(X_test)
            preds = sgd.predict(test_ngram)

            #for pred, yval in zip(preds, y_test):
            #    print("pred= {}, yval= {}".format(pred, yval))
            result = classification_report(y_test, preds, target_names=self.catname_list)
            # print('result = [{}]'.format(result))
            print(result)
            report_list.append(result)

            prec, recall, f1 = doccatutils.report_to_eval_scores(result)
            prec_list.append(prec)
            recall_list.append(recall)
            f1_list.append(f1)

        sum_prec = sum(prec_list)
        sum_recall = sum(recall_list)
        sum_f1 = sum(f1_list)
        sum_count = len(f1_list)

        doccatutils.print_combined_reports(report_list, self.valid_tags)
        print("\nreported avg precision= {:.2f}, recall= {:.2f}, f1= {:.2f}".format(sum_prec / sum_count,
                                                                                    sum_recall / sum_count,
                                                                                    sum_f1 / sum_count))

        print("\nwith filtering of threshold 0.75")
        prod_tag_list = doccatutils.print_combined_reports(report_list, self.valid_tags, threshold=0.75,
                                                           prod_status_fname=prod_status_fname)

        # save the valid tags to limit the production document categories
        prod_tags_fname = 'dict/doccat.prod.tsv'
        with open(prod_tags_fname, 'wt') as fout:
            for tag_id, tag in enumerate(prod_tag_list):
                print("{}\t{}".format(tag, tag_id), file=fout)
        print("wrote {}".format(prod_tags_fname))


    def predict(self, doc_text: str, catnames=None) -> List[str]:
        doc_feats = [doccatutils.doc_text_to_docfeats(doc_text)]
        # pylint: disable=invalid-name
        X_test = self.transformer.transform(doc_feats)

        # probs = self.classifier.predict_proba(X_test)
        # we only have 1 document
        preds = self.classifier.predict(X_test)[0]
        # print('preds: {}'.format(preds))

        result = []
        for i, pred in enumerate(preds):
            if pred == 1:
                result.append(self.catname_list[i])

        return result
