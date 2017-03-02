
# 
# 30% test set 
# 5 x positive samples in testing
# recall =0.9 precision = 0.65
import sys
import csv
import numpy 
import sklearn
import nltk,re,pprint
import math
import scipy.sparse 
import util
from fractions import Fraction
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
from scipy.sparse import lil_matrix,csr_matrix,hstack,vstack
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
from sklearn.feature_selection import RFE,VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.cross_validation  import train_test_split
from sklearn.grid_search import  GridSearchCV
from fractions import Fraction
import time


start_time = time.time()

prefix = util.get_commandline_prefix()

filename = prefix+'csr_file.npz'
fileY    =  prefix+'Y_file.npy'
header_file = prefix+'header_file'

print("start loading bag_matrix")
X = util.load_sparse_csr(filename)
print("end bag matrix ")
print("start loading Y")
print("original shape of X = ",X.shape)
Y = numpy.load(fileY)
z=numpy.where(Y==1)
X_old = X
Y_old = Y
# find positive rows
l = list(z[0])
Y_pos = Y[l]
X_pos = X[l,:]
#
multiplier = 5
print("multiplier = ",multiplier)
for item in range(multiplier):
  X = vstack((X,X_pos),format='csr')
  Y = numpy.concatenate((Y,Y_pos),axis=0)
#
print("end of loading Y")
header_list=[]
f = open(header_file,'rt',encoding = 'utf-8') 
line = f.read()
header_list=line.split('\n')
header_list = header_list[:(len(header_list)-1)]
#
print("start log regression")
#
# Create TRAIN and TEST Datasets
X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,test_size = 0.3,random_state=42)
X_tr1,X_te1,Y_tr1,Y_te1 = train_test_split(X_old,Y_old,test_size = 0.3,random_state=42)
print("Start SGD")
C_param = [10**(-7),10**(-6), 10**(-5)]
best= numpy.zeros((len(C_param ),4))
header_result=["C","threshold 90","recall 90","precision 90"]
for iC, C in enumerate(C_param):
    
        
        print("@@@@@@@@@@@@@@@@@@@@")
        print("running without feature selection")
        print("C= ",C)
        print("shape of X_new = ",X.shape)
        #sgd= LogisticRegression(penalty='l2',C=C)
        sgd= SGDClassifier( loss='log',alpha=C, n_iter=300, penalty='l2', shuffle=True)
       
        sgd.fit(X_tr, Y_tr)
        
        # PREDICTION STARTS
        sgd_pred = sgd.predict(X_te1)
        recall_rate = sklearn.metrics.recall_score(Y_te1,sgd_pred)
        print("recall_rate no threshold ",recall_rate)
        precision_rate = sklearn.metrics.precision_score(Y_te1,sgd_pred)
        print("precision_rate no threshold ",precision_rate)
        T = sgd.predict_proba(X_te1)
        V = T[:,0] - T[:,1]
        start, stop, n = 0.0,1.0, 200
        grid=[float(Fraction(start) + i * (Fraction(stop) - Fraction(start)) / n) for i in range(n+1)]
        recall_90 =0
        precision_90 = 0
        threshold_90 = 0
               # measure precision when recall is equal or close to 0.90
        for threshold in grid: 
                sgd_pred = ~(V > threshold)
                recall_rate = sklearn.metrics.recall_score(Y_te1,sgd_pred)
                #print("threshold = ",threshold)
                #print("C = ",C)
                #print("recall_rate ",recall_rate)
                #precision_rate = sklearn.metrics.precision_score(Y_te1,sgd_pred)
                #print("precision_rate ",precision_rate)
                if abs(recall_rate-0.90) < abs(recall_90-0.90):
                     recall_90 = recall_rate
                     precision_90 = precision_rate
                     threshold_90 = threshold
        best[iC,0] = C
        best[iC,1] = threshold_90
        best[iC,2] = recall_90
        best[iC,3] = precision_90
        print(header_result)
        print(best)

