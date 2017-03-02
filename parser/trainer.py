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
from nltk.tokenize.punkt import PunktSentenceTokenizer
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
average_pct_pred = 0
min_cnt = 1
party_counts,train_ant = util_parse.popular_parties(prefix,train_ant,min_cnt)
law_dict = list( set(law_dict) - set(party_counts))

total_negative = []
for iant,ant in enumerate(train_ant): # DOCUMENT LEVEL
   final_list = []
   list_party,party_pos = util_parse.return_party_list_and_pos(ant,prefix)
   new_trimmed = util_parse.ant_trimmer(list_party)
   print("trimmed = ",new_trimmed,'\n')
   if not new_trimmed :
      new_trimmed = list_party
   new_trimmed = ' '.join(new_trimmed)
   new_trimmed  = nltk.word_tokenize(new_trimmed)
   set_trimmed = set(new_trimmed)
   #print("####################################",'\n')
   txt_file = os.path.splitext(ant)
   txt_file = txt_file[0]
   input_file = prefix+'//'+txt_file+'.txt'
   print("text file = ",input_file)
   print("document number = ",iant)
   with open(input_file, newline='', encoding='utf-8') as f:
        tot1 = f.read()
   
   tot = ' '.join([word for word in tot1.split() if word not in cachedStopWords ])
   sent_tokenize_list,dummy = util_parse.start_sen_tokenizer(tot)
   raw_sentence_list,start_end = util_parse.start_sen_tokenizer(tot1)
   dupli  = sen_picker.sen_picker(ant,median_multiplier,number_most_popular,prefix)
   sen_dupli  = list([0,1])
   
   sen_dupli = sorted(list(set(sen_dupli + dupli)))
   if len(sen_dupli) > 2:
      sen_dupli = sen_dupli[0:2]
   tagged_list = []  
   for isen  in sen_dupli:
         
         raw_sentence = raw_sentence_list[isen]
         raw_sentence = " ".join(re.findall("[a-zA-Z]+", raw_sentence))
         raw_tagged = nltk.pos_tag(nltk.word_tokenize(raw_sentence))
         tagged_list = tagged_list + [item[0] for item in raw_tagged]
        
            
   negative_list = list( set(tagged_list) - set_trimmed)
   total_negative = total_negative + negative_list
   
   print("####################################################")
   
total_negative = set(total_negative)

total_negative = total_negative - set(party_counts)
