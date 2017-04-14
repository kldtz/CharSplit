# take a provision and a CSV file, and train the classifier, saving it to disk

# TODO should we be able to train individual provisions separately

import os
import os.path
import tempfile
import concurrent.futures
import ebrevia.learn.bag_transform as bag_transform
import ebrevia.learn.bigram_transform as bigram_transform
import ebrevia.learn.sgd as sgd
from sklearn.externals import joblib

# Whether to store the intermediate feature matrices in files.
# This can speed up experimentation if trying different classification algorithms,
# but means you have to remember to delete the matrices when training other provisions,
# or else they'll be incorrectly used for those provisions.
CACHE_MATRICES = False


def predictProvision(test_file, bag_matrix, Y, clf):
    return clf.predict(test_file, bag_matrix, Y)


def trainProvision(provision, train_file, test_file, bag_matrix, bag_matrix_te, Y, Y_te, prefix):
    print('STARTING TRAINING FOR ' + provision)
    clf = EBClassifier(prefix, provision)
    clf.train(train_file, test_file, bag_matrix, bag_matrix_te, Y, Y_te)
    return clf


class EBClassifier:
    def __init__(self, prefix, provision):
        self.prefix = prefix
        self.provision = provision

    def train(self, train_file, test_file, bag_matrix, bag_matrix_te, Y, Y_te):
        provision_file = self.prefix + self.provision + '_matrix.pkl'

        if CACHE_MATRICES and os.path.isfile(provision_file):
            [self.bigram_transformer, bigram_matrix, bigram_matrix_te, overrides] = joblib.load(provision_file)
        else:
            print('Prepping train data for ' + self.provision)
            self.bigram_transformer = bigram_transform.BigramTransform(self.provision)
            bigram_matrix = self.bigram_transformer.storeBigramMatrix(train_file)

            print('Prepping test data for ' + self.provision)
            bigram_matrix_te, overrides = self.bigram_transformer.storeBigramMatrixTest(test_file)
            if CACHE_MATRICES:
                joblib.dump([self.bigram_transformer, bigram_matrix, bigram_matrix_te, overrides], provision_file)

        print('training and testing for ' + self.provision)
        self.sgdClassifier = sgd.Sgd()
        self.sgdClassifier.trainAndTest(train_file, bigram_matrix, bag_matrix, Y,
                                        bigram_matrix_te, bag_matrix_te, Y_te, overrides)
        print('DONE TRAINING FOR ' + self.provision)

    def predict(self, test_file, bag_matrix, Y):
        print('STARTING PREDICTING FOR ' + self.provision)
        bigram_matrix, overrides = self.bigram_transformer.storeBigramMatrixTest(test_file)
        # load and classify the matrices
        predictions = list(self.sgdClassifier.predict(bigram_matrix, bag_matrix, Y, overrides))
        print('DONE PREDICTING FOR ' + self.provision)
        return predictions


class Learner:
    def __init__(self, prefix, newschool=False):
        self.prefix = prefix

        # one to share
        self.bag_transform = bag_transform.BagTransform(newschool)

        # one per provision
        self.clfs = {}

    # file is the csv file to train on
    # prefix is the directory where various files will be stored
    # provision is the provision that should be trained on
    def train(self, train_file, test_file, provisions=None):
        print("prefix is: " + self.prefix)
        mkdirIfNecessary(self.prefix)

        general_file = self.prefix + 'general_matrix.pkl'

        if CACHE_MATRICES and os.path.isfile(general_file):
            [self.bag_transform, bag_matrix, Ys, bag_matrix_te, Ys_te] = joblib.load(general_file)
        else:
            print('Prepping generic train data' + train_file)
            bag_matrix, Ys = self.bag_transform.storeBagMatrix(train_file)

            print('Prepping generic test data ' + test_file)
            bag_matrix_te, Ys_te = self.bag_transform.storeBagMatrixTest(test_file)
            if CACHE_MATRICES:
                joblib.dump([self.bag_transform, bag_matrix, Ys, bag_matrix_te, Ys_te], general_file)

        if (provisions == None):
            provisions = Ys.keys()
        self.provisions = list(provisions)

        print('Provisions are ', provisions)

        with concurrent.futures.ProcessPoolExecutor(1) as executor:
            future_to_provision = {
                executor.submit(trainProvision, provision,
                                train_file, test_file,
                                bag_matrix, bag_matrix_te,
                                Ys[provision], Ys_te[provision], self.prefix):
                    provision for provision in provisions}
            for future in concurrent.futures.as_completed(future_to_provision):
                provision = future_to_provision[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%s generated an exception: %s' % (provision, exc))
                else:
                    self.clfs[provision] = data

    # predict on a file which has never been processed
    def predict(self, test_file, provisions=None):
        if (provisions == None):
            provisions = self.provisions

        if (len(provisions) == 0):
            return {}

        # should be temp space
        test_prefix = tempfile.mkdtemp(prefix='learner')

        # process the incoming file into a matrix
        bag_matrix, Ys = self.bag_transform.storeBagMatrixTest(test_file)
        predictions = {}
        with concurrent.futures.ThreadPoolExecutor(8) as executor:
            future_to_provision = {executor.submit(predictProvision, test_file, bag_matrix, Ys[provision],
                                                   self.clfs[provision]):
                                       provision for provision in provisions}
            for future in concurrent.futures.as_completed(future_to_provision):
                provision = future_to_provision[future]
                data = future.result()
                predictions[provision] = data

        return predictions

    def save(self, fname):
        joblib.dump(self, fname)


def load(prefix):
    return joblib.load(prefix + 'learner.pkl')


def mkdirIfNecessary(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
