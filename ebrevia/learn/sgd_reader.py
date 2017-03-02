import csv
import numpy 
import sklearn
import nltk,re,pprint
import pylab as pl
import math
import util
import scipy.sparse 
from alpha_func import alpha_func
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import recall_score
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy import sparse
from scipy.sparse import lil_matrix,csr_matrix,hstack,vstack
import time

prefix = util.get_commandline_prefix()
fileY    =  prefix+'Y_file'
header_file = prefix+'header_file'
file_yes_no = prefix+'yes_no'
bag_file = prefix+'bag_file'

Y = numpy.load(fileY+'.npy')
yes_no_matrix = numpy.load(file_yes_no+'.npy')


header = []
with open(header_file,mode = 'r',encoding = 'utf-8') as f:
     reader = csv.reader(f,delimiter=',')
     for row in reader:
              header.append(row)
#            
# only bag matrix remains
#
yes_no_size = yes_no_matrix.shape[1]
n_features = len(header)
n_rows = 3
count=-1

temp_matrix =numpy.zeros((n_rows,n_features))
with open(bag_file,mode = 'r') as f:
    for iline, line in enumerate(f):
          print("iline = ",iline)
          
          count = count + 1
          st1  = numpy.array(re.findall(r'\d+', line))
          
          if ( (count%n_rows)==0 and count > 0 ):
              
              count = 0
              print("count = ",count)
                    

              #### starts over again
              temp_matrix = numpy.zeros((n_rows,n_features))
              #### fills up first row of temp matrix
              temp_matrix[0,0:yes_no_size] += yes_no_matrix[iline,0:yes_no_size]
              for item in st1:
                 temp_matrix[0, int(item)+yes_no_size]=1
        
             
          else:
              print("count = ",count)
              temp_matrix[count,0:yes_no_size] += yes_no_matrix[iline,0:yes_no_size]
              for item in st1:
                 temp_matrix[count, int(item)+yes_no_size]=1
          
       
