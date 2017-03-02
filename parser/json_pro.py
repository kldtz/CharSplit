import os
import glob
import csv
import numpy 
import nltk,re
import math
import scipy.sparse 
from fractions import Fraction
from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from fractions import Fraction
import time
import random
import itertools
import pickle
import json
import codecs
from collections import Counter
from pprint import pprint
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import shlex
from nltk.corpus import ieer
from nltk.corpus import wordnet
import collections
import util_parse
from statistics import median
from nltk.corpus import words

character_list=[]
character_list.append('(')
character_list.append(')')
character_list.append(' ')

number_most_popular = 50
median_multiplier = 2
n_docs = 100
prefix = util_parse.get_commandline_prefix()

law_file = 'law_dict.txt'
law_dict = util_parse.get_law_dict(law_file)


text_file_list,ant_file_list = util_parse.get_file_lists(prefix)
len_train = int(len(text_file_list)*0.75) # create a training set and a test set
train_text = text_file_list[:len_train]
test_text = text_file_list[len_train:]
train_ant = ant_file_list[:len_train]
test_ant = ant_file_list[len_train:]
average_pct_pred =0

for iant,ant in enumerate(train_ant): # DOCUMENT LEVEL
   final_list = []
   list_party = util_parse.return_party_list(ant,prefix)
   new_trimmed = util_parse.ant_trimmer(list_party)
   print("trimmed = ",new_trimmed,'\n')
   if not new_trimmed :
      new_trimmed = list_party
   #print("####################################",'\n')
   txt_file = os.path.splitext(ant)
   txt_file = txt_file[0]
   input_file = prefix+'//'+txt_file+'.txt'
   print("text file = ",input_file)
   print("document number = ",iant)
   (word_cnt,Named_Ent,sentences) = util_parse.Count_Popular(input_file)
   #print("word_cnt",word_cnt)
   upper_list = util_parse.extract_upper_singles(word_cnt)
   new_upper = []
   for up in upper_list:
      up_toks = nltk.word_tokenize(up)
      if str(up_toks).lower() not in words.words():
         new_upper.append(up)
         break
      
   if upper_list:
      print("upper_list = ",new_upper)
   median_cnt = median(word_cnt.values())
   threshold_cnt = min( max(word_cnt.values()),median_multiplier * median_cnt)
   most_popular = word_cnt.most_common(number_most_popular)
   capital_popular = util_parse.trim_popular(most_popular,threshold_cnt)
   earliest=[]
   for icap,cap in enumerate(capital_popular):
        for i,sen in enumerate(sentences):
             if cap[0] in sen:
                  earliest.append( i)
                  break
  
   duplicates =  [x for x, y in collections.Counter(earliest).items() if y > 1]
   if not duplicates: # fill duplicates with earliest if there are no duplicates
        duplicates = earliest

   if 0 not in duplicates:
      duplicates.append(0)
   duplicates = sorted(duplicates)
   
   capital_popular =  [pop[0] for pop in capital_popular]

   
   
   new_popular =  util_parse.eliminate_geo_words(capital_popular,prefix)
   
   #new_popular =  util_parse.eliminate_not_english_words(new_popular)
   combined_indices = []
   tot_list_pop = []
   len_dupl= len(duplicates)
   len_dupl = min(len_dupl,2)
   for iduplic,duplic_sen_number in enumerate(duplicates[0:len_dupl]):
            print("sentence number = ",duplic_sen_number)
            sen = sentences[duplic_sen_number]
            list_pop = util_parse.sen_pop_strings(sen,new_popular)
            
            stop = stopwords.words('english')
            sen =  [i for i in sen.split() if i not in stop]
            sen = ' '.join(sen)
            sen.replace('\n',' ')
            sen.replace('\"',' ')
            (name_list,namedEnt) = util_parse.extract_entities(sen)
          
            sen_leaves =  namedEnt.leaves()
           
            len_sen_leaves = len(sen_leaves)
            
            
            unique_indices = []
            
            for pop in list_pop:
     
               index_list,NNP_list = util_parse.search_for_NNPS_simple(sen_leaves,new_popular,law_dict)
              
     
               if NNP_list:
                   combined_indices = combined_indices + NNP_list   
               else:
                   print("problem: string not found = ",pop)
              
            if combined_indices:
              [unique_indices.append(item) for item in combined_indices if item not in unique_indices]
              #unique_indices = util_parse.eliminate_geo_words(unique_indices,prefix)
  
            tot_list_pop = tot_list_pop+list_pop
            print("tot = ",tot_list_pop)
             
   tot_list_pop = list(set(tot_list_pop)) 
   final_list = tot_list_pop+unique_indices
   final_list = final_list + new_upper
   print("final_list = ",final_list)
   
   pred_pct = util_parse.overlap_func(final_list,new_trimmed)
   average_pct_pred += pred_pct
   print("predicted pct = ",pred_pct)
   print("####################################",'\n')
   if iant > n_docs:
        print("average pred pct = ",average_pct_pred/(n_docs+2))
        break
  

