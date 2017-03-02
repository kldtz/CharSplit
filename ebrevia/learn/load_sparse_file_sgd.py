import csv
import numpy 
import sklearn
import nltk,re,pprint
import pylab as pl
import math
import scipy.sparse 
import util
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import recall_score
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy import sparse
from scipy.sparse import lil_matrix,csr_matrix,hstack
from sklearn.preprocessing import normalize
from sklearn.linear_model import *
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import *
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.feature_selection import RFE



import time


prefix=util.get_commandline_prefix()

start_time = time.time()

filename = prefix+'csr_file.npz'
fileY    =  prefix+'Y_file.npy'
header_file = prefix+'header_file'

print("start loading bag_matrix")
X = util.load_sparse_csr(filename)
print("end bag matrix ")
print("start loading Y")
Y = numpy.load(fileY)
print("end of loading Y")
header_list=[]
f = open(header_file,'rt',encoding = 'utf-8') 
line = f.read()
header_list=line.split('\n')
header_list = header_list[:(len(header_list)-1)]
#
print("start log regression")
#
print("Start SGD")

svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, Y)
ranking = rfe.ranking_.reshape(digits.images[0].shape)


'''
sgd = SGDClassifier( loss='log',alpha=10**-6, n_iter=200, penalty='l2', shuffle=True,verbose=1,class_weight='auto')
sgd.fit(X, Y)
y_sgd_pred = sgd.predict(X)
recall_rate = sklearn.metrics.recall_score(Y,y_sgd_pred)
print("recall_rate ",recall_rate)
precision_rate = sklearn.metrics.precision_score(Y,y_sgd_pred)
print("precision_rate ",precision_rate)

'''
