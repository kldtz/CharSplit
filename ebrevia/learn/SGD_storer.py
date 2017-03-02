import csv
import numpy 
import sklearn
import nltk,re,pprint
import pylab as pl
import math
import scipy.sparse 
import util
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
from scipy.sparse import lil_matrix,csr_matrix,hstack
import time

def save_sparse_csr(filename,array):
    numpy.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

prefix=util.get_commandline_prefix()

fileY    =  prefix+'Y_file'
header_file = prefix+'header_file'
file_yes_no = prefix+'yes_no'
bag_file = prefix+'bag_file'

start_time = time.time()

number_of_top_words=10000
Y_column=26
bag_of_words_column=22
header_list=[]
bag_of_words = []
set_col = [0,1,2,3,4]
numeric_col = list(range(5,22))
Y=[]

count=0
with open(prefix+'notsparse.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
              count = count+1
              if(count==1):
                 for i in range(len(row)):
                     header_list.append(row[i])
              
f.close()


cols = len(set_col)+len(numeric_col)
yes_no_matrix = numpy.zeros(shape=(count,cols)) # turn yes and no in (1,0) numpy mat

count=0
with open(prefix+'notsparse.csv', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for  row in reader:
              bag_of_words.append(row[bag_of_words_column]) 
              if ('yes' in row[Y_column]):
                    Y.append(1)
              else:
                    Y.append(0)
              for ibinary,binary in enumerate(set_col):
                  if ('yes' in row[binary]):
                       yes_no_matrix[count,ibinary]=1
              if (count > 0): # avoiding header line
                  for inumer,numer in enumerate(numeric_col):
                       yes_no_matrix[count,(inumer+len(set_col))]=row[numer]
              count =count+1
              
f.close()


###########################
# save Y response file
###########################
Y=Y[1:] #eliminate header
Y = numpy.array(Y)
numpy.save(fileY,Y)



#####################################
### preprocess and save yes_no_matrix
#####################################
yes_no_matrix=yes_no_matrix[1:,] #eliminate header
min_max_scaler = preprocessing.MinMaxScaler()
yes_no_matrix= min_max_scaler.fit_transform(yes_no_matrix)
numpy.save(file_yes_no,yes_no_matrix)
###############################
# build and print final_header
###############################
yn_header=[header_list[i] for i in set_col]
num_header = [header_list[i] for i in numeric_col]
#final_header = yn_header+num_header+common_words_list
#with open(header_file,'w',encoding='utf-8') as f:
#    for row in final_header:
#        f.write((row) + '\n')



bag_of_words = bag_of_words[1:]  #eliminate header

#
######
joined_bag = " ".join(bag_of_words)
print("start of tokenization")
tokenized_bag = word_tokenize(joined_bag)
print("end of tokenization")
stops = set(stopwords.words('english'))
print("start filtering out stopwords")
filtered_list = [w for w in tokenized_bag if not w in stops and len(w)>1]
print("end of filtering")
print("Find list of top words")
fdistribution= FreqDist(filtered_list)
most_common_list = fdistribution.most_common(number_of_top_words)
#del fdistribution

common_words_list=[]
for item in most_common_list:
    common_words_list.append(item[0])
n_features =  number_of_top_words

# this sparse matrix will contain everything 


#with open(bag_file,'w',encoding='utf-8') as f:
#


print("starting saving bag_matrix in a csv file")



with open(bag_file,'a') as  f:
  for ind1,word in enumerate(bag_of_words):
       bag_vector = []
       for ind2,common_word in  enumerate(common_words_list):
           if( common_word in word):
                  bag_vector.append(ind2)
       
       f.writelines( str(bag_vector))
       f.writelines("\n")
f.close()
print("done saving bag_matrix")


yn_header=[header_list[i] for i in set_col]
num_header = [header_list[i] for i in numeric_col]
final_header = yn_header+num_header+common_words_list
##############################
##########  write header_file
##############################

with open(header_file, "w",encoding = 'utf-8') as f:
    for val in final_header:
        f.write(val)
        f.write('\n')



