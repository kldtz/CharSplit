
import logging
from abc import ABC, abstractmethod

import numpy as np

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing

from kirke.docclassifier import doccategory, doccatutils


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

    def train(self, txt_fn_list_fn, model_file_name):

        logging.info('start training unigram document classifier: [%s]', txt_fn_list_fn)

        # each y is a list of catid
        # pylint: disable=invalid-name
        X_both, y_both, catname_list = doccatutils.load_data(txt_fn_list_fn, is_step1=False)

        # TODO, in future, this should be influnced by catnames
        self.catname_list = catname_list

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
    def train_and_evaluate(self, txt_fn_list_fn, is_step1=False):
        logging.info('start training and evaluate unigram document classifier: [%s]',
                     txt_fn_list_fn)

        # pylint: disable=invalid-name
        EB_DOC_KFOLD = 5

        # each y is a list of catid
        X_both, y_both, catname_list = doccatutils.load_data(txt_fn_list_fn, is_step1)

        X_both = np.asarray(X_both)
        y_both = np.asarray(y_both)

        mlbinarizer = preprocessing.MultiLabelBinarizer()
        bin_y_both = mlbinarizer.fit_transform(y_both)

        kfolds = KFold(n_splits=EB_DOC_KFOLD, shuffle=True)

        prec_list, recall_list, f1_list = [], [], []
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
            result = classification_report(y_test, preds, target_names=catname_list)
            print(result)

            prec, recall, f1 = doccatutils.report_to_eval_scores(result)
            prec_list.append(prec)
            recall_list.append(recall)
            f1_list.append(f1)

        sum_prec = sum(prec_list)
        sum_recall = sum(recall_list)
        sum_f1 = sum(f1_list)
        sum_count = len(f1_list)
        print("avg precision= {:.2f}, recall= {:.2f}, f1= {:.2f}".format(sum_prec / sum_count,
                                                                         sum_recall / sum_count,
                                                                         sum_f1 / sum_count))


    def predict(self, doc_text, catnames=None):
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