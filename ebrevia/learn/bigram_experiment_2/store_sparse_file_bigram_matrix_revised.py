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
from sklearn.metrics import recall_score
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy import sparse
from scipy.sparse import lil_matrix,csr_matrix,coo_matrix,hstack
import time

prefix=util.get_commandline_prefix()
header_file = prefix+'header_file_bigram_new.txt'
input_file = prefix+'new.csv'
file_bigram = prefix+'file_bigram_new'

n_bigram_words = 150
number_of_top_words=175 # !!!!!!! make sure that n_bigram_words < number_of_top_words
Y_column=26
bag_of_words_column=22
header_list=[]
bag_of_words = []
set_col = [0,1,2,3,7]
numeric_col =[4,5,6]
numeric_col.extend(list(range(8,22)))
Y=[]

count=0
with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
              count = count+1
              if(count==1):
                 for i in range(len(row)):
                     header_list.append(row[i])
              
f.close()




with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for  row in reader:
              row_bag = row[bag_of_words_column].replace('\u201c', '"')
              row_bag = row_bag.replace('\u201d', '"')
              row_bag = row_bag.replace('\u2019', "'")
              bag_of_words.append(row_bag)
             
              
              
f.close()


bag_of_words = bag_of_words[1:]  #eliminate header

print("done with normalization")
#
######
joined_bag = " ".join(bag_of_words)
print("start of tokenization")
tokenized_bag = word_tokenize(joined_bag)
print("end of tokenization")
stops = set(stopwords.words('english'))
print("start filtering out stopwords")
filtered_list = [w for w in tokenized_bag if not w.lower() in stops and len(w)>1]
print("end of filtering")
print("Find list of top words")
fdistribution= FreqDist(filtered_list)
most_common_list = fdistribution.most_common(number_of_top_words)
top_words = fdistribution.most_common(n_bigram_words)
common_words_list=[]
top_100_words  = [item[0] for item in top_words]
del fdistribution
common_words_list = [ item[0] for item in most_common_list]
n_features = number_of_top_words

# this sparse matrix will contain everything 
bag_matrix = numpy.zeros((len(bag_of_words),n_features))
# lets copy yes_no_matrix to bag_matrix



print("start filling bag_of_words part")
for ind2,common_word in  enumerate(common_words_list):
         indices = [i for i, s in enumerate(bag_of_words) if common_word in s]
         bag_matrix[indices,ind2]+= 1


print("done allocating bag_matrix")


bigram_header=[]

print("start creating bigrams")

matrix_rows=[]
matrix_cols=[]
matrix_values=[]
n_columns = -1
for iw,w in enumerate(top_100_words):
    print("iw =",iw)
    for iw2 in range(iw+1,len(top_100_words)):    #len(top_100_words)):
        #print("iw2 =",iw2)
        #print("create colsum")
        col_sum = (bag_matrix[:,iw] + bag_matrix[:,iw2]) 
        zero_index =  numpy.where(col_sum == 2) # match is where value = 0
        n_zero_values = zero_index[0].shape[0]
        if (n_zero_values > 0):
           bigram_header.append( ','.join( ( top_100_words[iw],top_100_words[iw2]) ) )
           print(n_columns)
           n_columns +=1
           #print("n_zero_values",n_zero_values)
           zero_indices = list(numpy.array(zero_index[0]).reshape(-1,))
           matrix_rows.extend(zero_indices)
           matrix_cols.extend([n_columns]*n_zero_values)
matrix_values=[1]*len(matrix_rows)    
bigram_matrix = sparse.csr_matrix((matrix_values,(matrix_rows,matrix_cols)))
        # reshape bigram_matrix only number of rows
if (bag_matrix.shape[0] > bigram_matrix.shape[0]):
    row_diff = bag_matrix.shape[0] - bigram_matrix.shape[0]
    r_shape = (row_diff,bigram_matrix.shape[1])
    filler_matrix = coo_matrix(r_shape)
    bigram_matrix = scipy.sparse.vstack((bigram_matrix,filler_matrix))

#############################################################
##########  write BIGRAM MATRIX ONLY and CORRESPONDING HEADER
#############################################################
print("write bigrams to file")
util.save_sparse_csr(file_bigram,bigram_matrix.tocsr())

with open(header_file,'w',encoding='utf-8') as f:
   for row in bigram_header:
      f.write(((row)) + '\n')
f.close()


