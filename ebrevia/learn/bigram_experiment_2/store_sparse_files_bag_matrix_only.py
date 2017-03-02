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
from scipy.sparse import lil_matrix,csr_matrix,hstack
import time

prefix = util.get_commandline_prefix()
boost_flag = 1 # if boost_flag >0 then bag_matrix positive features are increased by booster amount
booster = 5
input_file = prefix+'new.csv'
distinct_positives_file = prefix + 'distinct_positives.txt'
#
if boost_flag: # in case of boosting
    print("boosting is on")
    filename = prefix+'csr_file_new_boosted'
    fileY    =  prefix+'Y_file_new_boosted'
    header_file = prefix+'header_file_bag_new_boosted.txt'
else:
    filename = prefix+'csr_file_new'
    fileY    =  prefix+'Y_file_new'
    header_file = prefix+'header_file_bag_new.txt'

#
number_of_top_words=15*(10**3)
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


cols = len(set_col)+len(numeric_col)
yes_no_matrix = numpy.zeros(shape=(count,cols)) # turn yes and no in (1,0) numpy mat

count=0
with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for  row in reader:
              row_bag = row[bag_of_words_column].replace('\u201c', '"')
              row_bag = row_bag.replace('\u201d', '"')
              row_bag = row_bag.replace('\u2019', "'")
              bag_of_words.append(row_bag)
              if ('yes' in row[Y_column]):
                    Y.append(1)
              else:
                    Y.append(0)
              for ibinary,binary in enumerate(set_col):
                  if ('yes' in row[binary]):
                       yes_no_matrix[count,ibinary]=1
              if (count >0): # avoiding header line
                  for inumer,numer in enumerate(numeric_col):
                       yes_no_matrix[count,(inumer+len(set_col))]=row[numer]
              count =count+1
              
f.close()

Y=Y[1:] #eliminate header
Y=numpy.array(Y,dtype=bool)
pos = numpy.where(Y>0)
ls_pos = list(pos[0])
neg = numpy.where(Y==0)
ls_neg = list(neg[0])
######
#
yes_no_matrix=yes_no_matrix[1:,] #eliminate header
bag_of_words = bag_of_words[1:]  #eliminate header
min_max_scaler = preprocessing.MinMaxScaler()
yes_no_matrix= min_max_scaler.fit_transform(yes_no_matrix)
bag_of_words_pos=[]
bag_of_words_neg=[]
for p in ls_pos:
    bag_of_words_pos.append(bag_of_words[p])
for p in ls_neg:
    bag_of_words_neg.append(bag_of_words[p])
print("length of positives = ", len(bag_of_words_pos))
print("done with normalization")
#
######
#joined_bag = " ".join(bag_of_words)
joined_bag_pos = " ".join(bag_of_words_pos)
joined_bag_neg = " ".join(bag_of_words_neg)
print("start of tokenization")
#tokenized_bag = word_tokenize(joined_bag)
tokenized_bag_pos = word_tokenize(joined_bag_pos)
tokenized_bag_neg = word_tokenize(joined_bag_neg)
print("end of tokenization")
stops = set(stopwords.words('english'))
print("start filtering out stopwords")
#filtered_list = [w for w in tokenized_bag if not w.lower() in stops and len(w)>1]
filtered_list_pos = [w for w in tokenized_bag_pos if not w.lower() in stops and len(w)>1]
filtered_list_neg = [w for w in tokenized_bag_neg if not w.lower() in stops and len(w)>1]
print("end of filtering")
print("Find list of top words")
#most_common_list = FreqDist(filtered_list).most_common(number_of_top_words)
most_common_list_pos = FreqDist(filtered_list_pos).most_common(number_of_top_words)
most_common_list_neg = FreqDist(filtered_list_neg).most_common(number_of_top_words)
common_words_list_pos = [ item[0] for item in most_common_list_pos]
common_words_list_neg = [ item[0] for item in most_common_list_neg]
#common_words_list = [ item[0] for item in most_common_list]
print("length of common positives = ", len(common_words_list_pos))
print("length of common negatives = ", len(common_words_list_neg))
unique_pos = list( set(common_words_list_pos)-set(common_words_list_neg))
with open(distinct_positives_file,'w',encoding='utf-8') as f:
   for row in unique_pos:
        f.write((row) + '\n')
pos_neg_intersect = set.intersection(set(common_words_list_pos),set(common_words_list_neg))
pos_neg_union = set.union(set(common_words_list_pos),set(common_words_list_neg))
common_words_list = list(pos_neg_union)
print("length of intersection",len(pos_neg_intersect))
print("length of union = ",len(common_words_list))
index_unique_pos = []
for i,ic in enumerate(common_words_list):
      if(ic in unique_pos):
            index_unique_pos.append(i)
print("length of booster list = ",len(index_unique_pos))

# change number of top words to reflect combo of positives and negatives

number_of_top_words = len(common_words_list)
n_features = cols + number_of_top_words

# this sparse matrix will contain everything 
bag_matrix = sparse.lil_matrix((len(bag_of_words),n_features))
# lets copy yes_no_matrix to bag_matrix

for irow,row in enumerate(yes_no_matrix):
     bag_matrix[irow,0:len(row)]=row

print("start filling bag_of_words part")
if boost_flag: # boost bag_matrix
       for ind2,common_word in  enumerate(common_words_list):
          indices = [i for i, s in enumerate(bag_of_words) if common_word in s]
          if ( ind2 in index_unique_pos): # case where word match includes unique positive features
                          bag_matrix[indices,(ind2+cols)]+= booster*scipy.ones((len(indices),1))
          else:          # case where word match is not unique positive feature
                          bag_matrix[indices,(ind2+cols)]+= scipy.ones((len(indices),1))
else: # do not boost: normal case
       for ind2,common_word in  enumerate(common_words_list):
          indices = [i for i, s in enumerate(bag_of_words) if common_word in s]
          bag_matrix[indices,(ind2+cols)]+= scipy.ones((len(indices),1))
    

print("done allocating bag_matrix")

# combine the partial headers into a final header list
yn_header=[header_list[i] for i in set_col]
num_header = [header_list[i] for i in numeric_col]
final_header = yn_header+num_header+common_words_list

##############################
##########  write matrix,lists
##############################
bag_matrix = sparse.csr_matrix(bag_matrix)
util.save_sparse_csr(filename,bag_matrix)
del bag_matrix
numpy.save(fileY,Y)

with open(header_file,'w',encoding='utf-8') as f:
   for row in final_header:
        f.write((row) + '\n')


