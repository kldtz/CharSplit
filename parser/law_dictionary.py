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
import stan_tagger 
from nltk.corpus import ieer
from nltk.corpus import wordnet
import collections
import util_parse
from statistics import median
character_list=[]
character_list.append('(')
character_list.append(')')
character_list.append(' ')

number_most_popular = 15000

prefix = util_parse.get_commandline_prefix()


text_file_list,ant_file_list = util_parse.get_file_lists(prefix)
len_train = int(len(text_file_list)*0.75) # create a training set and a test set
train_text = text_file_list[:len_train]


total_file = []
for t_text in train_text:
    input_file = prefix+'//'+t_text 
    with open(input_file, newline='', encoding='utf-8') as f:
        doc = f.read()
        total_file.append(doc)
total_file = ' '.join(total_file)
print("done with joining")
#sent_tokenize_list = sent_tokenize(total_file)          
tokenized = nltk.word_tokenize(total_file)
cnt = Counter()
for word in tokenized:
     cnt[word] += 1
most_popular = cnt.most_common(number_most_popular) 
items = dict(most_popular).keys()
output_file = "law_dict.txt" 
with open(output_file,'w',encoding ='utf-8')as f:
   for item in items:
    f.write("%s\n" % item.lower())
