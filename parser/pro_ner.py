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
   with open(input_file, newline='', encoding='utf-8') as f:
        tot = f.read()
   
   tot = ' '.join([word for word in tot.split() if word not in cachedStopWords ])
   sent_tokenize_list = sent_tokenize(tot)
   tokenizer = SpaceTokenizer()
   nes = []
   nes_sen = []
   dupli  = sen_picker.sen_picker(ant,median_multiplier,number_most_popular,prefix)
   sen_dupli  = list([0,1,2])
   sen_dupli = list(set(sen_dupli+dupli))
   sen_dupli.sort()
   for isen  in sen_dupli:
         sentence = sent_tokenize_list[isen]
         print("sentence # ",isen)
         toks = tokenizer.tokenize(sentence)
         for t in toks:
            if (t.istitle() or t.isupper()):
               t = t.replace('\n','')
               t= re.sub(r'(_)', '', t)
               t= " ".join(re.findall("[a-zA-Z]+", t))
               nes.append(t)
               nes_sen.append(isen)
   #nes =  util_parse.eliminate_geo_words(nes,prefix)
   
   for ne in nes:
        if ne.lower() in law_dict:
            #print("Found",pop,'\n')
            nes.remove(ne)
   
   print(nes)
   pred_pct = util_parse.overlap_func(nes,new_trimmed)
   average_pct_pred += pred_pct
   print("predicted pct = ",pred_pct)
   print("####################################################")
   if iant > n_docs:
        print("average pred pct = ",average_pct_pred/(n_docs+2))
        break

