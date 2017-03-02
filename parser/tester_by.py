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

<<<<<<< HEAD
legal_stop = list(['WITNESSETH','WHEREAS','Table','Effective','Agreement','effective','shall','This'])
=======
legal_stop = list(['WITNESSETH','WHEREAS','Table','TABLE','Effective','Agreement','effective','shall','This'])
>>>>>>> origin
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
# 
average_pct_pred = 0
min_cnt = 1

dummy,train_ant = util_parse.popular_parties(prefix,train_ant,min_cnt)
word_counts = util_parse.popular_parties_inc(prefix,train_ant,min_cnt)

party_counts,test_ant = util_parse.popular_parties(prefix,test_ant,min_cnt)
law_dict = list( set(law_dict) - set(party_counts))

counter = 0
yet_counter = 0
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
   sent_tokenize_list = [item.replace('\n',' ') for item in sent_tokenize_list]
   sent_tokenize_list = [item.replace('.',' ') for item in sent_tokenize_list]
<<<<<<< HEAD
   sent_tokenize_list_lower = [item.lower() for item in sent_tokenize_list]
=======
   sent_tokenize_list = [item.replace('-',' ') for item in sent_tokenize_list]
   #sent_tokenize_list_lower = [item.lower() for item in sent_tokenize_list]
>>>>>>> origin
   
   depth = 4
   sentence_numbers=[]
   
   by_flag,sentence_number1 = util_parse.search_bybetween(sent_tokenize_list,depth)
<<<<<<< HEAD
   by_flag_low,sentence_number1_low = util_parse.search_bybetween(sent_tokenize_list_lower,depth)
=======
   bycolon_flag,sentence_number_colon = util_parse.search_bybetweencolon(sent_tokenize_list,depth)
   by_flag_low,sentence_number1_low = util_parse.search_bybetween(sent_tokenize_list,depth)
>>>>>>> origin
   between_flag,sentence_number_bet = util_parse.search_between_only(sent_tokenize_list,depth)
   among_flag,sentence_number2 = util_parse.search_byamong(sent_tokenize_list,depth)
   byismade_flag,sentence_number3 = util_parse.search_by_ismade(sent_tokenize_list,depth)
   byisentered_flag,sentence_number4 = util_parse.search_by_isentered(sent_tokenize_list,depth)
<<<<<<< HEAD
   execamong_flag,sentence_number_exec = util_parse.search_execamong(sent_tokenize_list_lower,depth)
   landlordintro_flag,sentence_number_landintro = util_parse.search_landintro(sent_tokenize_list_lower,depth)
=======
   execamong_flag,sentence_number_exec = util_parse.search_execamong(sent_tokenize_list,depth)
   landlordintro_flag,sentence_number_landintro = util_parse.search_landintro(sent_tokenize_list,depth)
   amongand_flag,sentence_number_amongand = util_parse.search_amongand(sent_tokenize_list,depth)
   landlord_flag,sentence_number_landlord = util_parse.search_landlord(sent_tokenize_list,depth)
   lessor_flag,sentence_number_lessor= util_parse.search_lessor(sent_tokenize_list,depth)

>>>>>>> origin
   # ORDER IN THIS LIST COUNTS!MAKE SURE CORRESPONDS TO FUNCTIONS
   sentence_numbers.append( (by_flag,sentence_number1))
   sentence_numbers.append( (by_flag_low,sentence_number1_low))
   sentence_numbers.append( (among_flag,sentence_number2))
   sentence_numbers.append( (between_flag,sentence_number_bet))
<<<<<<< HEAD
=======
   sentence_numbers.append( (bycolon_flag,sentence_number_colon))
>>>>>>> origin
   sentence_numbers.append( (byismade_flag,sentence_number3))
   sentence_numbers.append( (byisentered_flag,sentence_number4))
   sentence_numbers.append( (execamong_flag,sentence_number_exec))
   sentence_numbers.append( (landlordintro_flag,sentence_number_landintro))
<<<<<<< HEAD
=======
   sentence_numbers.append( (amongand_flag,sentence_number_amongand))
   sentence_numbers.append( (landlord_flag,sentence_number_landlord))
   sentence_numbers.append( (lessor_flag,sentence_number_lessor))
>>>>>>> origin
   #
   min_value = depth
   z=numpy.array(sentence_numbers).sum(axis=0)
   if z[0]<1:
      pointer = -1
      print("no function triggered")
   else:
      for inum,num in enumerate(sentence_numbers):
         if num[0]==1 :
<<<<<<< HEAD
            if num[1]<min_value:
               min_value = num[1]
               pointer = inum
   
                   
   
   if   pointer==0:
        print("by between func")
        util_parse.find_byb(sent_tokenize_list,sentence_number1,legal_stop)
   elif pointer==1:
        print("by between func lower")
        util_parse.find_byb(sent_tokenize_list_lower,sentence_number1_low,legal_stop)
=======
             pointer = inum
             break

  
   if   pointer==0:
        print("by between func")
        util_parse.find_byb(sent_tokenize_list,sentence_number1,legal_stop)
        
   elif pointer==1:
        print("by between func lower")
        util_parse.find_byb(sent_tokenize_list,sentence_number1_low,legal_stop)
>>>>>>> origin
   elif pointer==2:    
        print("by among func")
        util_parse.find_among(sent_tokenize_list,sentence_number2,legal_stop)
   elif pointer==3:
        print(" between only func")
        util_parse.find_between(sent_tokenize_list,sentence_number_bet,legal_stop)
   elif pointer==4:
<<<<<<< HEAD
        print("by is made func")
        util_parse.find_byismade(sent_tokenize_list,sentence_number3,legal_stop)
   elif pointer==5:
        print("by is entered func")
        util_parse.find_byisentered(sent_tokenize_list,sentence_number4,legal_stop)
   elif pointer==6:    
        print("executed among func ")
        util_parse.find_execamong(sent_tokenize_list,sentence_number_exec,legal_stop)
   elif pointer==7:    
        print("landlord: func ")
        util_parse.find_landintro(sent_tokenize_list,sentence_number_landintro,legal_stop)
        
   else :
        sen  = sent_tokenize_list[0]
        if 'landlord' in sen.lower():
           print("sen  = ",sen)
           input("Enter")
        yet_counter += 1
        
         
=======
        print("by between func colon")
        util_parse.find_byb_colon(sent_tokenize_list,sentence_number_colon,legal_stop)
        
   elif pointer==5:
        print("by is made func")
        util_parse.find_byismade(sent_tokenize_list,sentence_number3,legal_stop)
   elif pointer==6:
        print("by is entered func")
        util_parse.find_byisentered(sent_tokenize_list,sentence_number4,legal_stop)
   elif pointer==7:    
        print("executed among func ")
        util_parse.find_execamong(sent_tokenize_list,sentence_number_exec,legal_stop)
   elif pointer==8:    
        print("landlord: func ")
        util_parse.find_landintro(sent_tokenize_list,sentence_number_landintro,legal_stop)
   elif pointer==9:    
        print("among and func ")
        util_parse.find_amongand(sent_tokenize_list,sentence_number_amongand,legal_stop)
   elif pointer==10:    
        print("landlord/tenant func ")
        #util_parse.find_landlord(sent_tokenize_list,sentence_number_landlord,legal_stop)
        input("LANDLORD STUFF")
   elif pointer==11:    
        print("lessor/lessee func ")
        #util_parse.find_lessor(sent_tokenize_list,sentence_number_lessor,legal_stop)
        
   else :
           sen  = sent_tokenize_list[0]
           print("sen[0] = ",sen)
           yet_counter += 1
  
>>>>>>> origin
   print("#############################################################")  

   
