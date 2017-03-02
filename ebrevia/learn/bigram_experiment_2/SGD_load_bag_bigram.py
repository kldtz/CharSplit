
# 
# 
import os
import glob
import csv
import numpy 
import sklearn
import nltk,re,pprint
import math
import scipy.sparse 
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
from sklearn.feature_selection import SelectKBest,chi2
import time
import util



def load_sparse_csr(filename):
    loader = numpy.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])





prefix = util.get_commandline_prefix()
boost_flag = 0 # if boost_flag >0 then bag_matrix positive features are increased by booster amount
#
LR_flag = 0 # run LR vs SGD
multiplier = 0
iterations = 500
#
if boost_flag: # in case of boosting
    print("boosting is on")
    filename =prefix + 'csr_file_new_boosted.npz'
    filebigram = prefix +'file_bigram_new.npz'
    fileY    = prefix + 'Y_file_new_boosted.npy'
    header_file_bag =prefix + 'header_file_bag_new_boosted.txt'
    header_file_bigram = prefix +'header_file_bigram_new_boosted.txt'
    header_file = prefix +'header_bag_&_bigram_boosted.txt'
else:
    filename =prefix + 'csr_file_new.npz'
    filebigram = prefix +'file_bigram_new.npz'
    fileY    = prefix + 'Y_file_new.npy'
    header_file_bag =prefix + 'header_file_bag_new.txt'
    header_file_bigram = prefix +'header_file_bigram_new.txt'
    header_file = prefix +'header_bag_&_bigram.txt'




header_file_list = [header_file_bag,header_file_bigram]
print("new code")
print("start loading bag_matrix")
X = load_sparse_csr(filename)
print("end bag matrix ")
print("start loading Y")
print("original shape of X = ",X.shape)
colSum = X.sum(axis=0)
colSum = numpy.squeeze(numpy.asarray(colSum))
zerofind = list(numpy.where(colSum==0))
all_cols = numpy.arange(X.shape[1])
cols_to_keep = numpy.where(numpy.logical_not(numpy.in1d(all_cols, zerofind)))[0]
X = X[:, cols_to_keep] #  remove cols where sum is zero
print("zerofind= ",zerofind)
Xbigram = load_sparse_csr(filebigram)
print("shape of Xbigram",Xbigram.shape)
X = sparse.hstack((X,Xbigram))
print("new shape of X = ",X.shape)

##########################
# read header referring to bag of words
header_list_bag=[]
f = open(header_file_bag,'rt',encoding = 'utf-8') 
line = f.read()
header_list_bag=line.split('\n')
header_list_bag = header_list_bag[:(len(header_list_bag)-1)]
header_list_bag = [ item for i,item in enumerate(header_list_bag) if i not in zerofind ]
#########################
header_list_bigram=[]
f = open(header_file_bigram,'rt',encoding = 'utf-8') 
line = f.read()
header_list_bigram=line.split('\n')
header_list_bigram = header_list_bigram[:(len(header_list_bigram)-1)]
header_list_bigram = [ item for i,item in enumerate(header_list_bigram)]
########################
final_header= header_list_bag + header_list_bigram
#        # write final header
with open(header_file,'w',encoding='utf-8',errors='ignore') as f:
   for irow,row in enumerate(final_header):
         f.write((row) + '\n' )


# end of removal zero columns in both X and header_list
Y = numpy.load(fileY)
# lets do feature selection
#b = SelectKBest(score_func=chi2,k=5000)
#b.fit_transform(X,Y)
#  separate positives from negatives

X = X.tolil()

# find positive rows

#

print("multiplier = ",multiplier)
print("number of iterations = ",iterations)

print("end of loading Y")

#
print("start log regression")
#
#mask  = b.get_support()
# Create TRAIN and TEST Datasets
X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,test_size = 0.3,random_state=42)
z=numpy.where(Y_tr==1)
ls = list(z[0])
Y_pos = Y_tr[ls]
X_pos = X_tr[ls,:]
# add the positives to the training set
for item in range(multiplier):
   X_tr = vstack((X_tr,X_pos),format='csr')
   Y_tr = numpy.concatenate((Y_tr,Y_pos),axis=0)
#  Y = numpy.concatenate((Y,Y_pos),axis=0)
#X_tr1 = X_tr1[:, mask]
#X_te1 = X_te1[:, mask]
#
#mask header_list
#
#header_revised=[]
#for ih in range(len(header_list)):
#  if mask[ih]:
#      header_revised.append(header_list[ih])

print("Start SGD/LR")
if LR_flag:
     print("run LR")
     C_param_best= [0.01,1,10,100] # if you use LR instead of SGD
else: # run SGD
     print("run SGD")
     C_param_best = [10**(-7),10**(-6),10**(-5)] # if you use SGD


best= numpy.zeros((len(C_param_best ),4))
header_result=["C","threshold 90","recall 90","precision 90"]
for iC, C in enumerate(C_param_best): 
    
        
        print("@@@@@@@@@@@@@@@@@@@@")
        print("running without feature selection")
        print("C= ",C)
        print("shape of X_train = ",X_tr.shape)
        print("picks classifier")
        if LR_flag:
           sgd= LogisticRegression(penalty='l2',C=C)
        else:
           sgd= SGDClassifier( loss='log',alpha=C, n_iter=iterations, penalty='l2', shuffle=True)
        #
        sgd.fit(X_tr, Y_tr)
        
        # PREDICTION STARTS
        sgd_pred = sgd.predict(X_te)
        recall_rate = sklearn.metrics.recall_score(Y_te,sgd_pred)
        print("recall_rate no threshold ",recall_rate)
        precision_rate = sklearn.metrics.precision_score(Y_te,sgd_pred)
        print("precision_rate no threshold ",precision_rate)
        T = sgd.predict_proba(X_te)
        V = T[:,0] - T[:,1]
        start, stop, n = 0.0,1.0, 200
        grid=[float(Fraction(start) + i * (Fraction(stop) - Fraction(start)) / n) for i in range(n+1)]
        recall_90 =0
        precision_90 = 0
        threshold_90 = 0
               # measure precision when recall is equal or close to 0.90
        for threshold in grid: 
                sgd_pred = ~(V > threshold)
                recall_rate = sklearn.metrics.recall_score(Y_te,sgd_pred)
                precision_rate = sklearn.metrics.precision_score(Y_te,sgd_pred)
                
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
##################################################
pos_pred = sgd.predict(X_pos)
error_loc = numpy.where(pos_pred==0)
ls1 = numpy.array(ls)
er_loc = list( ls1[error_loc])
##########################################################
bag_of_words=[]
bag_of_words_column=22
input_file = 'C:\\Users\\LouVacca\\Desktop\\ebrevia4\\new.csv'
output_file = 'C:\\Users\\LouVacca\\Desktop\\ebrevia4\\change_of_control.'
with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for  row in reader:
              row_bag = row[bag_of_words_column].replace('\u201c', '"')
              row_bag = row_bag.replace('\u201d', '"')
              row_bag = row_bag.replace('\u2019', "'")
              bag_of_words.append(row_bag)
f.close()
f=open(output_file, 'wt', encoding='utf-8') 
for loc in er_loc:
     f.write(bag_of_words[loc])
     f.write('#########################################################################' +'\n')
f.close()
