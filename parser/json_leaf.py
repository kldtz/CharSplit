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

from nltk.corpus import ieer
from nltk.corpus import wordnet
import collections
import util_parse
from statistics import median
character_list=[]
character_list.append('(')
character_list.append(')')
character_list.append(' ')

number_most_popular = 50
median_multiplier = 7
prefix = util_parse.get_commandline_prefix()

law_file = 'law_dict.txt'
law_dict = util_parse.get_law_dict(law_file)


text_file_list,ant_file_list = util_parse.get_file_lists(prefix)
len_train = int(len(text_file_list)*0.75) # create a training set and a test set
train_text = text_file_list[:len_train]
test_text = text_file_list[len_train:]
train_ant = ant_file_list[:len_train]
test_ant = ant_file_list[len_train:]


for iant,ant in enumerate(train_ant):
   list_party = util_parse.return_party_list(ant,prefix)
   print(list_party,'\n')
   #print("####################################",'\n')
   txt_file = os.path.splitext(ant)
   txt_file = txt_file[0]
   input_file = prefix+'//'+txt_file+'.txt'
   print("text file = ",input_file)
   (word_cnt,Named_Ent,sentences) = util_parse.Count_Popular(input_file)
   median_cnt = median(word_cnt.values())
   threshold_cnt = min( max(word_cnt.values()),median_multiplier * median_cnt)
   print("threshold = ",threshold_cnt)
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
   print("duplicates = ",duplicates,'\n')
   duplicates = sorted(duplicates)
   capital_popular =  [pop[0] for pop in capital_popular]
   print("cap_popular = ",capital_popular,'\n')
   new_popular =  util_parse.eliminate_geo_words(capital_popular,prefix)
   print("geo_pop = ",new_popular,'\n')
   new_popular =  util_parse.eliminate_not_english_words(new_popular)
   
  
   for iduplic,duplic_sen_number in enumerate(duplicates):
            print("sentence number = ",duplic_sen_number)
            sen1 = sentences[duplic_sen_number]
            
            stop = stopwords.words('english')
            sen =  [i for i in sen1.split() if i not in stop]
            sen = ' '.join(sen)
            sen.replace('\n',' ')
            (name_list,namedEnt) = util_parse.extract_entities(sen)
            print("sen = ",sen)
            #print("new pop =",new_popular)
            
            list_pop = util_parse.sen_pop_strings(sen,new_popular)
            #print("pop_list = ",list_pop)
            #print("named_Ent",namedEnt)
            
      
            for pop in list_pop:
               tok_pop = nltk.word_tokenize(pop)
               #print("tok pop= ",tok_pop[0])
               word_list,node_loc = util_parse.getlabels(namedEnt,tok_pop[0])
               print(" pop = ",pop,'\n')
               print("word_list =",word_list)
               #print("node_loc =",node_loc)
               if node_loc>-1:
                  NN_list = util_parse.extract_NNP(word_list,pop)  
                  print("NN list = ",NN_list)
        
            if iduplic>-1:
               break
    
   
   print("####################################",'\n')
  
   if iant > 5:
        break
  

