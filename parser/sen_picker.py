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

def sen_picker(ant,median_multiplier,number_most_popular,prefix):

  
     
      txt_file = os.path.splitext(ant)
      txt_file = txt_file[0]
      input_file = prefix+'//'+txt_file+'.txt'
      (word_cnt,Named_Ent,sentences) = util_parse.Count_Popular(input_file)
      #print("word_cnt",word_cnt)
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
    
      return duplicates
