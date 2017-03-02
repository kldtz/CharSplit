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
from nltk.tokenize import SpaceTokenizer

import shlex
from nltk.corpus import ieer
from nltk.corpus import wordnet
import collections
import util_parse
from statistics import median
from nltk.corpus import words
import sen_picker 

character_list=[]
character_list.append('(')
character_list.append(')')
character_list.append(' ')

number_most_popular = 50
median_multiplier = 2
n_docs = -1
bottom_doc = 0.95
top_nsentences_search  = 10
max_sentences = 3
prefix = util_parse.get_commandline_prefix()

law_file = 'law_dict.txt'
law_dict = util_parse.get_law_dict(law_file)
cachedStopWords = stopwords.words("english")

text_file_list,ant_file_list = util_parse.get_file_lists(prefix)
len_train = int(len(text_file_list)*0.75) # create a training set and a test set
train_text = text_file_list[:len_train]
test_text = text_file_list[len_train:]
train_ant = ant_file_list[:len_train]
test_ant = ant_file_list[len_train:]
#
false_pos =  pickle.load( open( "total_negative.p", "rb" ) )

# 0.9449 average pred without subtractions
# 
average_pct_pred = 0
min_cnt = 1

dummy,train_ant = util_parse.popular_parties(prefix,train_ant,min_cnt)
word_counts = util_parse.popular_parties_inc(prefix,train_ant,min_cnt)
common_parties = word_counts.most_common(200)
common_parties = [item[0].lower() for item in common_parties ]
print("common parties",common_parties)

party_counts,test_ant = util_parse.popular_parties(prefix,test_ant,min_cnt)
law_dict = list( set(law_dict) - set(party_counts))

counter = 0
for iant,ant in enumerate(test_ant): # DOCUMENT LEVEL
   final_list = []
   #list_party = util_parse.return_party_list(ant,prefix)
   list_party,party_pos = util_parse.return_party_list_and_pos(ant,prefix)
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
   with open(input_file, newline='', encoding='utf-8') as f:
        tot1 = f.read()
   
   #tot = ' '.join([word for word in tot1.split() if word not in cachedStopWords ])
   sent_tokenize_list = sent_tokenize(tot1)

   sentence_number,by_flag = util_parse.search_bybetween(sent_tokenize_list)
   dummy,start_end = util_parse.start_sen_tokenizer(tot1)
   tokenizer = SpaceTokenizer()
   nes = []
   nes_sen = []
   dupli  = sen_picker.sen_picker(ant,median_multiplier,number_most_popular,prefix)
   #  
   top_nsentences_search = min(top_nsentences_search,(len(sent_tokenize_list)-1))
   sen_dupli  = util_parse.pick_top(sent_tokenize_list,top_nsentences_search,max_sentences)
   sen_dupli = list(set(sen_dupli+dupli))
   if not dupli:
      dupli = list([0,1])
   sen_dupli.sort()
   print("dupli = ",sen_dupli)
   len_sen_dupli = len(sen_dupli)
   for isen  in sen_dupli:
       
         sentence = sent_tokenize_list[isen]
         #print(sentence)
         index_list = []
         sen_index = []
         
         toks = nltk.word_tokenize(sentence)
         #inter = list(set(toks) & set(common_parties))
         inter = util_parse.word_intersect_lower(toks,common_parties)
         if inter :
            print("sentence # ",isen)
            for it,t in enumerate(toks):
               if (t.istitle() or t.isupper()):
                  #if t not in false_pos:
                     index_list.append(it)
                     sen_index.append(isen)
                     t = t.replace('\n','')
                     t = re.sub(r'(_)', '', t)
                     t = " ".join(re.findall("[a-zA-Z]+", t))
                     nes.append(t.lower())
         
   print(nes)

     
   #pred_pct = util_parse.overlap_func_lower(nes,new_trimmed)
   if nes:
      counter += 1
      pred_pct = util_parse.overlap_func_lower(nes,new_trimmed)
      average_pct_pred += pred_pct
      print("predicted pct = ",pred_pct)
   print("####################################################")
   #if iant > n_docs:
   #     print("average pred pct = ",average_pct_pred/(n_docs+2))
   #    break

print("average pred pct = ",average_pct_pred/counter)
