import sys
import numpy
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
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.corpus import wordnet
import collections
import util_parse
from nltk.sem import chat80
import shlex
from nltk.tokenize import SpaceTokenizer
cachedStopWords = stopwords.words("english")
import calendar

from itertools import chain

def partition(alist, indices):
    pairs = zip(chain([0], indices), chain(indices, [None]))
    return (alist[i:j] for i, j in pairs)


def one_cap_least(word):
     bool_res = any(x.isupper() for x in word)
     return bool_res

def get_law_dict(law_file):
     new_dict=[]
     with open(law_file,encoding='utf-8') as f:
        law_dict = f.readlines()
     law_dict = [ idict.replace('\n','') for idict in law_dict ]
     for il,law in enumerate(law_dict):
         law = " ".join(re.findall("[a-zA-Z]+", law))
         if wordnet.synsets(law):
            new_dict.append(law)
     new_dict = list(set(new_dict))
     return new_dict

def get_commandline_prefix():
    prefix = prefix = "C://Users//LouVacca//Desktop//parser///export-train"
    if(len(sys.argv) > 1):
        prefix = sys.argv[-1]
    return prefix
'''
def find_entities(sen,word,pop):
    
    if quotd:
        print("quoted= ",quotd,'\n')
        print("new_popular = ",pop,'\n')
        print("len of quoted = ",len(quotd),'\n')
        for quot  in  quotd:
            print("quot = ",quot,'\n')
            
            if  quot in pop:
                quotd.remove(quot)
                #begin_str = sen.find(word)
                #end_str = begin_str+len(word)
    print("quoted after = ",quotd,'\n')
    
    return quotd
'''

def get_file_lists(prefix):
    text_file_list = []
    ant_file_list = []
    for file in os.listdir(prefix):
      if file.endswith(".txt"):
         text_file_list.append(file)
      elif file.endswith("ant"):
        ant_file_list.append(file)
    return (text_file_list,ant_file_list)
def eliminate_geo_words(popular,prefix):
    
    
    locations = []
    location_file = 'Locations.csv'
    location_file = prefix + '//' + location_file
    with open(location_file, 'rt',encoding='utf-8') as f:
       reader = csv.reader(f, delimiter=',') 
       for row in reader:
          locations.extend(row)
    #print("first location = ",locations[0])
    for pop in popular:
        if pop in locations:
            #print("Found",pop,'\n')
            popular.remove(pop)
    return popular

def eliminate_not_english_words(popular):
    
       
    for pop in popular:
        #print("pop 0  = ",pop[0],'\n')
        word_list = nltk.word_tokenize(pop)
        #print("word_list =",word_list,'\n')
        for word in word_list:
            if not  wordnet.synsets(word):
                #print("Found Not English",pop,'\n')
                popular.remove(pop)
                break
    return popular

'''
def find_bracketed(sentence,popular,character_list):
    
    
        if tk in chat80.words:
            print("geographical",'\n')
   
    tagged = nltk.pos_tag(tokenized)
    sentence_tree = nltk.ne_chunk(tagged)
    print("sen tree=",type(sentence_tree))
    #print(sentence_tree)
    GPE_chunks=[]
    for chunk in sentence_tree:
             if hasattr(chunk, 'label') :
                 b = chunk.label()
                 if b == 'GPE':
                     GPE_chunks.append(chunk)
                 #print("type =",type(chunk.label))
                 #print("chunk.label=",chunk.label(),"\n")
                 #print (chunk.label, ' '.join(c[0] for c in chunk.leaves()) )
    print(GPE_chunks)
   
    #for i,pop in enumerate(popular):
    #     pop=pop[0]
    #     if pop in sentence:
           
            begin_str = sentence.find(pop)
            end_str = begin_str+len(pop)
            #
            if begin_str-1 > -1:
              beg_char = sentence[begin_str-1]
            else:
              beg_char ='beg'
            if end_str+1 < len(sentence):
              end_char = sentence[begin_str-1]
            else:
              end_char ='end'
            print("begin character = ",)
            print("end character = ",sentence[end_str+1])
            if beg_char in character_list or  end_char in character_list :
               new_popular.append(pop)
           
    return popular
'''
def extract_entities(text):
     tokenized = nltk.word_tokenize(text)
     tagged = nltk.pos_tag(tokenized)
     namedEnt = nltk.ne_chunk(tagged, binary = True)
     #print(tagged )
     #print (namedEnt)
     name_list = [' '.join([y[0] for y in x.leaves()]) for x in namedEnt.subtrees() if x.label() == "NE"]
     return (name_list,namedEnt)

def return_party_list(ant_file,prefix):
      ant_file = prefix+'//'+ant_file
      list_party =[]
      party_pos =[]
      with open(ant_file,encoding = 'utf-8') as data_file:    
         data = json.load(data_file)
         for dat in data:
             if dat.get('type')== 'party':
                newdat = dat.get('text')
                list_party.append(newdat)
                start = dat.get('start')
                end = dat.get('end')
                #print("start = ",start)
                #party_pos.append(start)
                #party_pos.append(end)
      return list_party

def return_party_list_and_pos(ant_file,prefix):
      ant_file = prefix+'//'+ant_file
      list_party =[]
      party_pos =[]
      with open(ant_file,encoding = 'utf-8') as data_file:    
         data = json.load(data_file)
         for dat in data:
             if dat.get('type')== 'party':
                newdat = dat.get('text')
                list_party.append(newdat)
                start = dat.get('start')
                end = dat.get('end')
                party_pos.append((start,end))
                #party_pos.append(end)
      return (list_party,party_pos)


def Count_Popular(input_file):
    with open(input_file, newline='', encoding='utf-8') as f:
        tot = f.read()
    sent_tokenize_list = sent_tokenize(tot)          
    (name_list,Named_Ent) = extract_entities(tot)
    cnt = Counter()
    for word in name_list:
     cnt[word] += 1
    return (cnt,Named_Ent,sent_tokenize_list)

def Count_Popular_Trimmed(input_file):
    with open(input_file, newline='', encoding='utf-8') as f:
        tot = f.read()
    tot = ' '.join([word for word in tot.split() if word not in cachedStopWords ])
    sent_tokenize_list = sent_tokenize(tot)          
    (name_list,Named_Ent) = extract_entities(tot)
    cnt = Counter()
    for word in name_list:
     cnt[word] += 1
    return (cnt,Named_Ent,sent_tokenize_list)

def trim_popular(most_popular,threshold_cnt):
     
     trimmed = []
     for mst in most_popular:
          word = mst[0]
          ct = mst[1]
          if (word.istitle() and ct>threshold_cnt) :
          #if (ct>1):
               #print(word,'\n')
               trimmed.append(mst)
   
     return trimmed
    
def locate_pop(pop,sen_leaves):
            beg_ind = -2
            end_ind = -2
            untagged = [ p[0]  for ip,p in enumerate(sen_leaves)]
            #print(untagged,'\n')
            pop_split = pop.split(" ")
            len_pop = len(pop_split)
            if len_pop > 1: # multiple words
                
                 #print(pop_split)
                 #beg_ind = pop_finder(pop_split[0], untagged)
                 #end_ind = pop_finder(pop_split[-1], untagged)
                 #if pop_split[1] in untagged[(beg_ind+1)]:
                        #end_ind = pop_finder(pop_split[-1], untagged)
                 beg_ind,end_ind = many_pop_finder(pop, untagged,len_pop)
                     
      
            else: # only 1 word
              
               #if (pop,'NNP') in sen_leaves:
                 beg_ind = pop_finder(pop, untagged)
                 end_ind = beg_ind
                 #print("index= ",beg_ind)
            return (beg_ind,end_ind)
                 
def  pop_finder(pop0, untagged):
    
    for iunt,unt in enumerate(untagged):
        if pop0 in unt:
            ind = iunt
            break
    return(ind)

def  many_pop_finder(pop, untagged,len_pop):
    
    for iunt,unt in enumerate(untagged):
        if pop[0] in unt and pop[-1] in untagged[iunt+len_pop-1]:
            beg_ind = iunt
            end_ind = iunt +len_pop-1
            break
    return (beg_ind,end_ind)

def  sen_pop_strings(sen,new_popular):
    list_pop = []
    for new_pop in new_popular:
        '''
        pop_tokens = nltk.word_tokenize(new_pop)
        if len(pop_tokens)>1:
           flag  = 1
           for tok in pop_tokens:
               if tok not in sen:
                   flag = 0
                   break
           if flag == 1:
              list_pop.append(new_pop)
        else:
        '''
        if new_pop in sen:
              list_pop.append(new_pop)
    return list_pop


def  find_indices_sen(sen_leaves,list_pop):
    index_list =[]
    
    for list_p in list_pop:
        for isen,sen in enumerate(sen_leaves):
            if list_p in sen[0]:
                index_list.append(isen)
    return index_list

def search_for_left_NNPS(beg_ind,sen_leaves,capital_popular,law_dict):
  index_list = []
  string_list = []
  beg_left_ind = -2
  end_left_ind = -2
  punkt_flag=0
  cap_joined = (' ').join(capital_popular)
  for i in range((beg_ind-1),-1,-1):
     #print("i loop")
     if (sen_leaves[i][1] == ',' or sen_leaves[i][1] == '.'):
         punkt_ind = i
         punkt_flag = 1
         #print("punkt_ind",punkt_ind)
     break
  #
  punkt_flag = 0
  if punkt_flag :
      if (punkt_ind) < (beg_ind-1):
         for i in range((punkt_ind+1),(beg_ind)):
             #
             word = sen_leaves[i][0]
             if ( word.istitle() or word.isupper() ) and 'NNP' in sen_leaves[i][1] and word not in cap_joined  :
                 index_list.append(i)
                 string_list.append(sen_leaves[i][0])
          
      if index_list:
          beg_left_ind = index_list[0]
          end_left_ind = index_list[-1]

  else: # no punkt found between the word and beginning of document
      for i in range(0,(beg_ind)):
            word = sen_leaves[i][0]
            if (word.istitle() or word.isupper()) and 'NNP' in sen_leaves[i][1] and word not in cap_joined :
                 index_list.append(i)
                 string_list.append(sen_leaves[i][0])
      
  return (beg_left_ind,end_left_ind,index_list,string_list)

def search_for_right_NNPS(end_ind,sen_leaves,capital_popular,law_dict):
  index_list = []
  string_list = []
  beg_left_ind = -2
  end_left_ind = -2
  punkt_flag=0
  cap_joined = (' ').join(capital_popular)
  for i in range((end_ind+1),len(sen_leaves)):
     #print("i loop")
     if (sen_leaves[i][1] == ',' or sen_leaves[i][1] == '.'):
         punkt_ind = i
         punkt_flag = 1
         #print("punkt_ind",punkt_ind)
     break
  #
  punkt_flag = 0
  if punkt_flag :
      if (punkt_ind) > (end_ind+1):
         for i in range((end_ind+1),(punkt_ind)):
             #
             word = sen_leaves[i][0]
             if word.istitle() and 'NNP' in sen_leaves[i][1] and word not in cap_joined :
                 index_list.append(i)
                 string_list.append(sen_leaves[i][0])
          
      if index_list:
          beg_left_ind = index_list[0]
          end_left_ind = index_list[-1]

  else: # no punkt found between the word and beginning of document
      for i in range(end_ind+1,len(sen_leaves)):
            word = sen_leaves[i][0]
            if word.istitle() and 'NNP' in sen_leaves[i][1] and word not in cap_joined :
                 index_list.append(i)
                 string_list.append(sen_leaves[i][0])
      
  return (beg_left_ind,end_left_ind,index_list,string_list)    

def find_consecutives(int_list):
   diff_ls=[]
   ls_index = []
   if len(int_list)>1:
       for i in range(0,len(int_list)-1):
           diff = int_list[i+1]-int_list[i]
           
           if diff == 1:
              diff_ls.append(i)
              diff_ls.append(i+1)

   set_diff = set(diff_ls)
   diff_ls = sorted(list(set_diff))
   return diff_ls

def find_consecutives_list(int_list):
   diff_ls=[]
   ls_index = []
   if len(int_list)>1:
       for i in range(0,len(int_list)-1):
           diff = int_list[i+1]-int_list[i]
           
           if diff == 1:
              diff_ls.append(i)
              diff_ls.append(i+1)

   set_diff = set(diff_ls)
   diff_ls = sorted(list(set_diff))
   for dif in diff_ls:
        ls_index.append(int_list[dif])
   return ls_index

def getlabels(parent,word_match):
    word_list = []
    word_flag = 0
    iloc = -1
    for inode,node in enumerate(parent):
        if type(node) is nltk.Tree:
            if word_flag == 0:
                word_list[:]=[]
            if word_flag == 1:
                break
            #if node.label() == 'ROOT':
                #print("======== Sentence =========")
                #print("Sentence:", " ".join(node.leaves()))
            #else:
                # print("Label:", node.label())
                # print("Leaves:", node.leaves())

            getlabels(node,word_match)
            
        else:
            
            word_list.append(node)
            if  (word_match in node[0] and 'NN' in node[1]) :
           
                #print("Word:", node)
                iloc = inode
                word_flag = 1
    if word_flag == 1:            
        return (word_list,iloc)
    else:
        return ([],iloc)

def extract_NNP(word_list,pop):
    new_list=[]
    final_list=[]
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    #for iw in  range(len(word_list)):
    #    if word_list[iw][0] not in punctuation:
    #        new_list.append(word_list[iw])
    len_pop = len(nltk.word_tokenize(pop)) 
    counter = 0
    index_list = []
    for iw in  range(len(word_list)):
        #print("inside word = ",word_list[iw][0])
        if  ('NN' in  word_list[iw][1] and word_list[iw][0].istitle() ) :
            counter=counter+1
            new_list.append(word_list[iw][0])
            index_list.append(iw)
    if len(index_list) >0 :
       ind = find_consecutives(index_list)
       if len(ind)>0:
           for index in ind:
               final_list.append(new_list[index])
    
    if counter>len_pop:
        return final_list
    else:
        return []
        


 

def search_for_NNPS_simple(sen_leaves,capital_popular,law_dict):
  index_list = []
  string_list = []
  cap_joined = (' ').join(capital_popular)
  

  
  for i in range(len(sen_leaves)):
            word = sen_leaves[i][0]
            if word.istitle() and 'NNP' in sen_leaves[i][1] and word not in cap_joined :
            
                 index_list.append(i)
                 string_list.append(sen_leaves[i][0])
      
  return (index_list,string_list)    

def overlap_func(final_list,new_trimmed):
    new_trimmed =  ' '.join(new_trimmed)
    len_trimmed  = len(new_trimmed) - new_trimmed.count(' ')
    #final_list.sort(key=len,reverse=False)

    for final in final_list:
        while final in new_trimmed:
            new_trimmed = new_trimmed.replace(final,'',1)
    after_len_trimmed = len(new_trimmed) - new_trimmed.count(' ')
    if len_trimmed == 0:
        pct = []
    else:
        pct = 1 - after_len_trimmed/len_trimmed 
    return pct

def overlap_func_lower(final_list,new_trimmed):
    new_trimmed =  ' '.join(new_trimmed)
    new_trimmed = new_trimmed.lower()
    len_trimmed  = len(new_trimmed) - new_trimmed.count(' ')
    #final_list.sort(key=len,reverse=False)

    for final in final_list:
        while final in new_trimmed:
            new_trimmed = new_trimmed.replace(final,'',1)
    after_len_trimmed = len(new_trimmed) - new_trimmed.count(' ')
    if len_trimmed == 0:
        pct = []
    else:
        pct = 1 - after_len_trimmed/len_trimmed 
    return pct
            
def extract_upper_singles(word_cnt):
    big_rare_words = []
    words = list(word_cnt.keys())
    cnt = list(word_cnt.values())
    
    for iw,word in enumerate(words):
        flag  = 1
        sub_word_list = nltk.word_tokenize(word)
        
        for sub_word in sub_word_list:
          #print("word synset",wordnet.synsets(word.lower()),"word = ",sub_word)
          if cnt[iw]==1 and word.isupper() :
              flag  = flag*1
          else:
              flag  = 0
        if flag:
            big_rare_words.append(word)
    return big_rare_words      
def ant_trimmer(list_party):
   
   trimmed_list_party = [' '.join(w for w in a.split() if ( w.isupper() or w.istitle() ))  for a in list_party ]
   new_trimmed = []
   for trim in trimmed_list_party:
      trim = " ".join(re.findall("[a-zA-Z]+", trim))
      new_trimmed.append(trim)
   return new_trimmed 

def popular_parties(prefix,train_ant,min_cnt):
   party_best = []
   new_train_ant = []
   for iant,ant in enumerate(train_ant): # DOCUMENT LEVEL
       final_list = []
       list_party = util_parse.return_party_list(ant,prefix)
       new_trimmed = util_parse.ant_trimmer(list_party)
       #print("trimmed = ",new_trimmed,'\n')
       if not new_trimmed :
           new_trimmed = list_party
       else:
           new_train_ant.append(ant)
       party_best = party_best + new_trimmed
   party_best = ' '.join(party_best)
   words = re.findall(r'\w+', party_best)
   cap_words = [word.lower() for word in words] #capitalizes all the words
   #
   word_counts = Counter(cap_words) #counts the number each time a word appears
   z = list(word_counts.keys())
   v = list(word_counts.values())
   common_party_words=[]
   for iv,val in enumerate(v):
       if val > min_cnt:
           if wordnet.synsets(z[iv]):
               common_party_words.append(z[iv])
   return (common_party_words,new_train_ant)
def popular_parties_inc(prefix,train_ant,min_cnt):
   party_best = []
   new_train_ant = []
   for iant,ant in enumerate(train_ant): # DOCUMENT LEVEL
       final_list = []
       list_party = util_parse.return_party_list(ant,prefix)
       new_trimmed = util_parse.ant_trimmer(list_party)
       #print("trimmed = ",new_trimmed,'\n')
       if not new_trimmed :
           new_trimmed = list_party
       else:
           new_train_ant.append(ant)
       party_best = party_best + new_trimmed
   party_best = ' '.join(party_best)
   words = re.findall(r'\w+', party_best)
   cap_words = [word.lower() for word in words] #capitalizes all the words
   #
   word_counts = Counter(cap_words) #counts the number each time a word appears
  
   return word_counts
################################################################
def create_predictive_matrix(sen,new_trimmed):
     #sen = " ".join(re.findall("[a-zA-Z]+", sen))
     tagged_sen  = nltk.pos_tag(nltk.word_tokenize(sen))
     print(tagged_sen)
     Y  = [0] *len(tagged_sen)
     tag_list = []
    
     b = [('__','E')]*2
     tagged = b + tagged_sen + b
    
     for itag in range(len(tagged)):
          if itag>1 and itag < (len(tagged)-2) :
              
                                        
               ind = list([itag-2,itag-1,itag,itag+1,itag+2])
               
               for i in ind:
                  tag_list.append(tagged[i][1])


     for itag,tag in enumerate(tagged_sen):
          if tagged_sen[itag][0] in new_trimmed:
                    new_trimmed = new_trimmed.replace(tagged_sen[itag][0],' ')
                    Y[itag]=1
                   
               
     if len(tag_list) % 5 == 0 and len(Y)*5 == len(tag_list):
          size_Y  = len(tag_list) / 5
          
     else:
          print("Error; number of tags is not multiple of 5")
     return (tag_list,Y,new_trimmed)       
               

def start_sen_tokenizer(text):
     sen_list = []
     start_end = []
     for start, end in PunktSentenceTokenizer().span_tokenize(text):          
         sen_list.append(text[start:end])
         start_end.append((start,end))
     return (sen_list,start_end)
   
def better_overlap(sen_dupli,start_end,party_pos):
     dupli_list = []
     party_list = []
     start_list=[]
     for du in sen_dupli:
         start_list.append(start_end[du])
     for du in start_list:
        for i in range(du[0],(du[1]+1)):
            dupli_list.append(i)

     for p in party_pos:
          for i in range(p[0],(p[1]+1)):
            party_list.append(i)
          
     diff_set = set(party_list)-set(dupli_list)
     
     pred = 1 - len(diff_set)/len(party_list)

     return pred

def count_cap_words(text):
     words = nltk.word_tokenize(text)
     count = 0
     for w in words:
          if w.istitle() or w.isupper():
               count+=1
     return count

def pick_top(sent_tokenize,top_nsentences_search,max_sentences):
     sen_list = []
     top = min((top_nsentences_search),len(sent_tokenize))
     for i in range(top):
          if count_cap_words(sent_tokenize[i])>1:
                 sen_list.append(i)
     if len(sen_list)>max_sentences:
       sen_list = sen_list[0:max_sentences]
     return sen_list

def get_cap_words(sent_tokenize_list,bottom_nphrases):
     tot_list= []
     for i in range(bottom_nphrases,len(sent_tokenize_list)):
          tot_list = tot_list + word_tokenize(sent_tokenize_list[i])
     
     cap_list = []
     for t in tot_list:
          if t.istitle() or t.isupper():
               cap_list.append(t)
     return cap_list
def get_cap_words_text(text):
     tot_list= []
     words = nltk.word_tokenize(text)
     for word in words:
          if word.istitle() or word.isupper() or word[0].isupper():
               tot_list.append(word)
          else:
               print("not capitalized")
               break
     return ' '.join(tot_list)  

def word_intersect(nes,cap_list):
     remain_bottom = []
     for ne in nes:
          for cap in cap_list:
               if cap == ne:
                    remain_bottom.append(cap)
     return list(set(remain_bottom))

def word_intersect_lower(nes,cap_list):
     remain_bottom = []
     for ne in nes:
          for cap in cap_list:
               if cap.lower() == ne.lower():
                    remain_bottom.append(cap)
     return list(set(remain_bottom))
 

def search_bybetween(sent_tokenize_list,max_n_sentence_search):
     flag = 0
     
     
     for isen,sen in enumerate(sent_tokenize_list):
          if 'by and between' in sen.lower()  and isen < max_n_sentence_search :
             by_match = re.search(r"[^a-zA-Z](by and between)[^a-zA-Z]", sen.lower())
             if by_match:
                  sen_remain  = sen[by_match.end(1):]
                  #print("sen remain = ",sen_remain)
                  if 'and ' in sen_remain.lower() or 'and,' in sen_remain.lower():
                       words  = nltk.word_tokenize(sen_remain)
                       
                       if words[0].istitle() or words[0].isupper() or words[0][0].isupper():
                           flag = 1
                           break
               
          
     return (flag,isen)

def search_bybetweencolon(sent_tokenize_list,max_n_sentence_search):
     flag = 0
     
     
     for isen,sen in enumerate(sent_tokenize_list):
          if 'by and between' in sen.lower()  and isen < max_n_sentence_search :
             by_match = re.search(r"[^a-zA-Z](by and between:)[^a-zA-Z]", sen.lower())
             if by_match:
                  sen_remain  = sen[by_match.end(1):]
                  #print("sen remain = ",sen_remain)
                  if ' and ' in sen_remain.lower() or 'and,' in sen_remain.lower():
                       words  = nltk.word_tokenize(sen_remain)
                       
                       if words[0].istitle() or words[0].isupper() or words[0][0].isupper():
                           flag = 1
                           break
               
          
     return (flag,isen)



def search_landintro(sent_tokenize_list,max_n_sentence_search):
     flag = 0
     
     
     for isen,sen in enumerate(sent_tokenize_list):
          if 'landlord:' in sen.lower() and 'tenant:' in sen.lower() and isen < max_n_sentence_search :
                           flag = 1
                           break
               
          
     return (flag,isen)

def search_landlord(sent_tokenize_list,max_n_sentence_search):
     flag = 0
     
     
     for isen,sen in enumerate(sent_tokenize_list):
          if 'landlord' in sen.lower() and 'tenant' in sen.lower() and isen < max_n_sentence_search :
                           flag = 1
                           break
               
          
     return (flag,isen)

def search_lessor(sent_tokenize_list,max_n_sentence_search):
     flag = 0
     
     
     for isen,sen in enumerate(sent_tokenize_list):
          if 'lessee' in sen.lower() and 'lessor' in sen.lower() and isen < max_n_sentence_search :
                           flag = 1
                           break
               
          
     return (flag,isen)

def search_byamong(sent_tokenize_list,max_n_sentence_search):
     flag = 0
     
     
     for isen,sen in enumerate(sent_tokenize_list):
          if 'by and among' in sen.lower()  and isen < max_n_sentence_search :
             by_match = re.search(r"(by and among)", sen.lower())
             if by_match:
                  sen_remain  = sen[by_match.end(1):]
                  if 'and ' in sen_remain.lower() :
                       words  = nltk.word_tokenize(sen_remain)
                       if words[0].istitle() or words[0].isupper() or words[0][0].isupper():
                           flag = 1
                           break
               
          
     return (flag,isen)

def search_amongand(sent_tokenize_list,max_n_sentence_search):
     flag = 0
     
     
     for isen,sen in enumerate(sent_tokenize_list):
          if 'among' in sen.lower()  and isen < max_n_sentence_search :
             by_match = re.search(r"(among)", sen.lower())
             if by_match:
                  sen_remain  = sen[by_match.end(1):]
                  if 'and ' in sen_remain.lower() or 'and,' in sen_remain.lower():
                       words  = nltk.word_tokenize(sen_remain)
                       if words[0].istitle() or words[0].isupper() or words[0][0].isupper():
                           flag = 1
                           break
               
          
     return (flag,isen)


def search_execamong(sent_tokenize_list,max_n_sentence_search):
     flag = 0
     
     
     for isen,sen in enumerate(sent_tokenize_list):
          if 'executed' in sen.lower()  and isen < max_n_sentence_search :
             by_match = re.search(r"(executed)", sen.lower())
             if by_match:
                  sen_remain  = sen[by_match.end(1):]
                  if 'among ' in sen_remain.lower():
                       words  = nltk.word_tokenize(sen_remain)
                       if words[0].istitle() or words[0].isupper() or words[0][0].isupper():
                           flag = 1
                           break
               
          
     return (flag,isen)

def search_by_ismade(sent_tokenize_list,max_n_sentence_search):
     flag = 0
     
     
     for isen,sen in enumerate(sent_tokenize_list):
          if 'is made' in sen.lower()  and isen < max_n_sentence_search :
             #print("sentence n = ",isen,"has is made")
             by_match = re.search(r"(is made )", sen.lower())
             if by_match:
                  sen_remain  = sen[by_match.end(1):]
                  if 'by ' in sen_remain.lower():
                      by_match1 = re.search(r"(by )", sen_remain.lower())
                      if by_match1:
                           sen_remain1 = sen_remain[by_match1.end(1):]
                           if 'and ' in sen_remain1.lower():
                             flag = 1
                             break
               
          
     return (flag,isen)

def search_by_isentered(sent_tokenize_list,max_n_sentence_search):
     flag = 0
     
     
     for isen,sen in enumerate(sent_tokenize_list):
          if 'is entered' in sen.lower()  and isen < max_n_sentence_search :
             
             by_match = re.search(r"(is entered )", sen.lower())
             if by_match:
                  sen_remain  = sen[by_match.end(1):]
                  if 'by ' in sen_remain.lower():
                           flag = 1
                           break
               
          
     return (flag,isen)
def search_between_only(sent_tokenize_list,max_n_sentence_search):
     flag = 0
     for isen,sen in enumerate(sent_tokenize_list):
         
          if 'between' in sen.lower()  and isen < max_n_sentence_search :
          
             by_match = re.search(r"(between)", sen.lower())
             if by_match:
                  sen_remain  = sen[by_match.end(1):]
                  if 'and ' in sen_remain.lower():
                      
                       words  = nltk.word_tokenize(sen_remain)
                      
                       if (words[0].istitle() or words[0].isupper()) or words[0][0].isupper() or words[0].isdigit():
                           flag = 1
                           break
               
          
     return (flag,isen)


          
     
def find_byb(sent_tokenize_list,sentence_number,legal_words):
    
      sen = sent_tokenize_list[sentence_number]
      #print(sen)
      by_match = re.search(r"[^a-zA-Z](by and between)[^a-zA-Z]", sen.lower())
      end_by = by_match.end(1)
      and_match = re.search(r"[^a-zA-Z](and)[^a-zA-Z]",sen[end_by:].lower() )
      and_start = and_match.start(1)
      min_ind = and_start
      comma_match  = re.search(r"[^a-zA-Z](,)[^a-zA-Z]", sen[end_by:].lower())
      if comma_match:
          min_ind = min(min_ind,comma_match.start(1))
      entity1 = sen[end_by:end_by+min_ind]
      
      remaining_str = sen[end_by+min_ind:len(sen)-1]
      #print("remaining = ",remaining_str)
     
      and_count = remaining_str.lower().count('and')
      
      if and_count ==1:
         print("and count = 1")
         and_match = re.search(r"(and)",remaining_str.lower() )
         remaining_str = remaining_str[and_match.end(1):]
         
         comma_match = re.search(r"(,)",remaining_str )
         if comma_match:
           entity2 = remaining_str[0:comma_match.start(1)]
           print("entity = ",entity1)
           print("entity2 = ",entity2)
         else:
           entity2 = remaining_str
           print("entity = ",entity1)
           print("entity2 = ",entity2)
      elif and_count > 1 :
         words = nltk.word_tokenize(remaining_str)
         entity3 =[]
         #print("words = ",words)
         inde = []
         for iw,w in enumerate(words):
            if 'and' == w.lower():
               inde.append(iw)
         
         for ind in inde :
            and_index = -10000000
            if words[ind+1].istitle() or words[ind+1].isupper() or one_cap_least(words[ind+1]):
               and_index = ind
              
         #
            if and_index>-1:
                for i in range(ind+1,len(words)):
                   if ( words[i].istitle() or words[i].isupper() or one_cap_least(words[i]) ) and words[i] not in calendar.month_name and words[i] not in stopwords.words("english"):
                      entity3.append(words[i])
                   else:
                        break
         print("entity = ",entity1)
         print("next entity = ",entity3)
       

def find_byb_colon(sent_tokenize_list,sentence_number,legal_words):
    
      sen = sent_tokenize_list[sentence_number]
      #print(sen)
      by_match = re.search(r"[^a-zA-Z](by and between:)[^a-zA-Z]", sen.lower())
      end_by = by_match.end(1)
      and_match = re.search(r"(and)",sen[end_by:].lower() )
      and_start = and_match.start(1)
      min_ind = and_start
      comma_match  = re.search(r"[^a-zA-Z](,)[^a-zA-Z]", sen[end_by:].lower())
      if comma_match:
          min_ind = min(min_ind,comma_match.start(1))
      entity1 = sen[end_by:end_by+min_ind]
      
      remaining_str = sen[end_by+min_ind:len(sen)-1]
      #print("remaining = ",remaining_str)
     
      and_count = remaining_str.lower().count('and')
      print("and_count = ",and_count)
      if and_count ==1:
         #print("and count = 1")
         and_match = re.search(r"(and)",remaining_str.lower() )
         remaining_str = remaining_str[and_match.end(1):]
         
         comma_match = re.search(r"(,)",remaining_str )
         if comma_match:
           entity2 = remaining_str[0:comma_match.start(1)]
           print("entity = ",entity1)
           print("entity2 = ",entity2)
         else:
           entity2 = remaining_str
           print("entity = ",entity1)
           print("entity2 = ",entity2)
      elif and_count > 1 :
         
         words = nltk.word_tokenize(remaining_str)
         entity3 =[]
         #print("words = ",words)
         inde = []
         for iw,w in enumerate(words):
            if 'and' == w.lower():
               inde.append(iw)
         #print("inde = ",inde)
         for ind in inde :
            and_index = -10000000
            if words[ind+1].istitle() or words[ind+1].isupper() or one_cap_least(words[ind+1]):
               and_index = ind
               
         #
            if and_index>-1:
               for i in range(ind+1,len(words)):
                   if ( words[i].istitle() or words[i].isupper() or one_cap_least(words[i]) ) and words[i] not in calendar.month_name:
                       entity3.append(words[i])
               #else:
               #   break
         print("entity = ",entity1)
         print("next entity = ",entity3)
       

def find_between(sent_tokenize_list,sentence_number,legal_words):
    
      sen = sent_tokenize_list[sentence_number]
      #print("sen = ",sen)
      by_match = re.search(r"[^a-zA-Z](between)[^a-zA-Z]", sen.lower())
      end_by = by_match.end(1)
      and_match = re.search(r"[^a-zA-Z](and)[^a-zA-Z]",sen[end_by:].lower() )
      and_start = and_match.start(1)
      min_ind = and_start
      comma_match  = re.search(r"[^a-zA-Z](,)[^a-zA-Z]", sen[end_by:].lower())
      if comma_match:
          min_ind = min(min_ind,comma_match.start(1))
      entity1 = sen[end_by:end_by+min_ind]
      #print("entity1 =",entity1)
      remaining_str = sen[end_by+min_ind:len(sen)-1]
      #print("remaining = ",remaining_str)
     
      and_count = remaining_str.lower().count('and ')
      
      if and_count == 1:
         #print("and count = 1")
         and_match = re.search(r"(and )",remaining_str.lower() )
         remaining_str = remaining_str[and_match.end(1):]
         #print("remaining more = ",remaining_str)
         comma_match = re.search(r"(,)",remaining_str )
         if comma_match:
           entity2 = remaining_str[0:comma_match.start(1)]
           entity2 = get_cap_words_text(entity2)
           print("entity = ",entity1)
           print("entity2 = ",entity2)
         else:
           entity2 = get_cap_words_text(remaining_str)
           print("entity = ",entity1)
           print("entity2 = ",entity2)
      elif and_count > 1 :
         and_index = -10000000
         words = nltk.word_tokenize(remaining_str)
         entity3 =[]
         #print("words = ",words)
         inde = []
         for iw,w in enumerate(words):
            if 'and' == w.lower():
               inde.append(iw)
         
         
         for ind in inde :
            bool_t = words[ind+1].istitle()
            bool_u = words[ind+1].isupper()
            bool_c = one_cap_least(words[ind+1])
            bool_s =  words[ind+1] not in stopwords.words("english")
            bool_l = words[ind+1] not in legal_words
            bool_ca = words[ind+1] not in calendar.month_name
            if (bool_t or  bool_u or bool_c) and bool_s and bool_l and bool_ca:
               and_index = ind
               break
         #
         if and_index>-1:
            for i in range(ind+1,len(words)):
               bool_t = words[i].istitle()
               bool_u = words[i].isupper()
               bool_c = one_cap_least(words[i])
               bool_s =  words[i] not in stopwords.words("english")
               bool_l = words[i] not in legal_words
               bool_ca = words[i] not in calendar.month_name
               if (bool_t or bool_u or bool_c   ) and bool_s and bool_l and bool_ca:
                  entity3.append(words[i])
               else:
                  break
         print("entity = ",entity1)
         print("next entity = ",entity3)
       



                
def find_among(sent_tokenize_list,sentence_number,legal_words):
      entities = []
      
      sen = sent_tokenize_list[sentence_number]
      #print(sen)
      by_match = re.search(r"[^a-zA-Z](by and among)[^a-zA-Z]", sen.lower())
      end_by = by_match.end(1)
      and_match = re.search(r"[^a-zA-Z](and)[^a-zA-Z]",sen[end_by:].lower() )
      and_start = and_match.start(1)
      min_ind = and_start
      #comma_match  = re.search(r"[^a-zA-Z](,)[^a-zA-Z]", sen[end_by:])
      #if comma_match:
      #    min_ind = min(min_ind,comma_match.start(1))
      entity1 = sen[end_by:end_by+min_ind]
      entities.append(entity1)
      
      remaining_str = sen[end_by+min_ind:len(sen)-1]
      #print("remaining = ",remaining_str)
     
      and_count = remaining_str.lower().count('and ')
      #print("and count",and_count)
      word_flag = 0
      if (and_count)>0:
          and_list=[]
          for m in re.finditer('and', remaining_str.lower()):
             and_list.append((m.start(),m.end())) 
          for ian,an in enumerate(and_list):
               if word_flag == 1:
                    break
               if ian ==0 and and_count>1:
                    
                    words = nltk.word_tokenize(remaining_str[0:an[0]])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if (word.istitle() or word.isupper() or  one_cap_least(word)) and word not in legal_words:
                               if word.title() not in calendar.month_name :
                                  rem.append(word)
                               else:
                                    word_flag = 1
                                    break
               elif ian == 0 and and_count == 0:
                    
                    words = nltk.word_tokenize(remaining_str[an[1]:])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if (word.istitle() or word.isupper() or  one_cap_least(word)) and word not in legal_words :
                               if word.title() not in calendar.month_name :
                                  rem.append(word)
                               else:
                                    word_flag = 1
                                    break
               elif ian == (len(and_list)-1):
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] :    ])
                    #print("words 1 =",words)
                    rem = []
                    for word in words:
                             if (word.istitle() or word.isupper() or  one_cap_least(word)) and word not in legal_words :
                               if word.title() not in calendar.month_name :
                                  rem.append(word)
                               else:
                                    word_flag = 1
                                    break
               else:
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] : and_list[ian][0]   ])
                    #print("words 2 =",words)
                    rem = []
                    for word in words:
                             if (word.istitle() or word.isupper() or  one_cap_least(word)) and word not in legal_words:
                               if word.title() not in calendar.month_name :
                                  rem.append(word)
                               else:
                                    word_flag = 1
                                    break
                                   
               #print("rem = ",' '.join(rem))
               entities.append(' '.join(rem))
      print("entities = ",entities)

def find_between2(sent_tokenize_list,sentence_number,legal_words):
      entities = []
      sen = sent_tokenize_list[sentence_number]
      #print(sen)
      by_match = re.search(r"[^a-zA-Z](between)[^a-zA-Z]", sen.lower())
      end_by = by_match.end(1)
      and_match = re.search(r"[^a-zA-Z](and)[^a-zA-Z]",sen[end_by:].lower() )
      and_start = and_match.start(1)
      min_ind = and_start
      #comma_match  = re.search(r"[^a-zA-Z](,)[^a-zA-Z]", sen[end_by:])
      #if comma_match:
      #    min_ind = min(min_ind,comma_match.start(1))
      entity1 = sen[end_by:end_by+min_ind]
      entities.append(entity1)
      
      remaining_str = sen[end_by+min_ind:len(sen)-1]
      #print("remaining = ",remaining_str)
     
      and_count = remaining_str.lower().count('and ')
      #print("and count",and_count)
      if (and_count)>0:
          and_list=[]
          for m in re.finditer('and', remaining_str.lower()):
             and_list.append((m.start(),m.end())) 
          for ian,an in enumerate(and_list):
               if ian ==0 and and_count>1:
                    
                    words = nltk.word_tokenize(remaining_str[0:an[0]])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               elif ian == 0 and and_count == 0:
                    
                    words = nltk.word_tokenize(remaining_str[an[1]:])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               elif ian == (len(and_list)-1):
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] :    ])
                    #print("words 1 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               else:
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] : and_list[ian][0]   ])
                    #print("words 2 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
                                   
               #print("rem = ",' '.join(rem))
               entities.append(' '.join(rem))
      print("entities = ",entities)
     
     

def find_byismade(sent_tokenize_list,sentence_number,legal_words):
      entities = []
      sen = sent_tokenize_list[sentence_number]
      #print(sen)
      by_match = re.search(r"[^a-zA-Z](by)[^a-zA-Z]", sen)
      end_by = by_match.end(1)
      and_match = re.search(r"[^a-zA-Z](and)[^a-zA-Z]",sen[end_by:] )
      and_start = and_match.start(1)
      min_ind = and_start
      #comma_match  = re.search(r"[^a-zA-Z](,)[^a-zA-Z]", sen[end_by:])
      #if comma_match:
      #    min_ind = min(min_ind,comma_match.start(1))
      entity1 = sen[end_by:end_by+min_ind]
      entities.append(entity1)
      
      remaining_str = sen[end_by+min_ind:len(sen)-1]
      #print("remaining = ",remaining_str)
     
      and_count = remaining_str.count('and ')
      #print("and count",and_count)
      if (and_count)>0:
          and_list=[]
          for m in re.finditer('and', remaining_str):
             and_list.append((m.start(),m.end())) 
          for ian,an in enumerate(and_list):
               if ian ==0 and and_count>1:
                    
                    words = nltk.word_tokenize(remaining_str[0:an[0]])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               elif ian ==0 and and_count == 0:
                    
                    words = nltk.word_tokenize(remaining_str[an[1]:])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               elif ian == (len(and_list)-1):
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] :    ])
                    #print("words 1 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               else:
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] : and_list[ian][0]   ])
                    #print("words 2 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
                                   
               #print("rem = ",' '.join(rem))
               entities.append(' '.join(rem))
      print("entities = ",entities)
     

def find_byisentered(sent_tokenize_list,sentence_number,legal_words):
      entities = []
      
      sen = sent_tokenize_list[sentence_number]
      #print(sen)
      by_match = re.search(r"[^a-zA-Z](by)[^a-zA-Z]", sen)
      end_by = by_match.end(1)
      and_match = re.search(r"[^a-zA-Z](and)[^a-zA-Z]",sen[end_by:] )
      and_start = and_match.start(1)
      min_ind = and_start
      #comma_match  = re.search(r"[^a-zA-Z](,)[^a-zA-Z]", sen[end_by:])
      #if comma_match:
      #    min_ind = min(min_ind,comma_match.start(1))
      entity1 = sen[end_by:end_by+min_ind]
      entities.append(entity1)
      
      remaining_str = sen[end_by+min_ind:len(sen)-1]
      #print("remaining = ",remaining_str)
     
      and_count = remaining_str.count('and ')
      #print("and count",and_count)
      if (and_count)>0:
          and_list=[]
          for m in re.finditer('and', remaining_str):
             and_list.append((m.start(),m.end())) 
          for ian,an in enumerate(and_list):
               if ian ==0 and and_count>1:
                    
                    words = nltk.word_tokenize(remaining_str[0:an[0]])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               elif ian ==0 and and_count == 0:
                    
                    words = nltk.word_tokenize(remaining_str[an[1]:])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               elif ian == (len(and_list)-1):
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] :    ])
                    #print("words 1 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               else:
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] : and_list[ian][0]   ])
                    #print("words 2 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
                                   
               #print("rem = ",' '.join(rem))
               entities.append(' '.join(rem))
      print("entities = ",entities)
     

def find_execamong(sent_tokenize_list,sentence_number,legal_words):
      entities = []
      
      sen = sent_tokenize_list[sentence_number]
      #print(sen)
      by_match = re.search(r"[^a-zA-Z](among)[^a-zA-Z]", sen)
      end_by = by_match.end(1)
      and_match = re.search(r"[^a-zA-Z](and)[^a-zA-Z]",sen[end_by:] )
      and_start = and_match.start(1)
      min_ind = and_start
      #comma_match  = re.search(r"[^a-zA-Z](,)[^a-zA-Z]", sen[end_by:])
      #if comma_match:
      #    min_ind = min(min_ind,comma_match.start(1))
      entity1 = sen[end_by:end_by+min_ind]
      entities.append(entity1)
      
      remaining_str = sen[end_by+min_ind:len(sen)-1]
      #print("remaining = ",remaining_str)
     
      and_count = remaining_str.count('and ')
      #print("and count",and_count)
      if (and_count)>0:
          and_list=[]
          for m in re.finditer('and', remaining_str):
             and_list.append((m.start(),m.end())) 
          for ian,an in enumerate(and_list):
               if ian ==0 and and_count>1:
                    
                    words = nltk.word_tokenize(remaining_str[0:an[0]])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               elif ian ==0 and and_count == 0:
                    
                    words = nltk.word_tokenize(remaining_str[an[1]:])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               elif ian == (len(and_list)-1):
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] :    ])
                    #print("words 1 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
               else:
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] : and_list[ian][0]   ])
                    #print("words 2 =",words)
                    rem = []
                    for word in words:
                             if word.istitle() or word.isupper() or  one_cap_least(word):
                               if word.title() not in calendar.month_name and word not in legal_words:
                                  rem.append(word)
                               else:
                                    break
                                   
               #print("rem = ",' '.join(rem))
               entities.append(' '.join(rem))
      print("entities = ",entities)
     
                                                                                              
                
        
def find_landintro(sent_tokenize_list,sentence_number,legal_words):
      entities = []
      
      sen = sent_tokenize_list[sentence_number]
      #print(sen)
      landlord_match = re.search(r"(landlord:)", sen.lower())
      end_by = landlord_match.end(1)
      tenant_match = re.search(r"(tenant:)",sen[end_by:].lower() )
      tenant_start = tenant_match.start(1)
      tenant_end = tenant_match.end(1)
      entity1 = sen[end_by:end_by+tenant_start]
      entities.append(entity1)
      remaining_str = sen[end_by+tenant_end:len(sen)-1]
     

      words = nltk.word_tokenize(remaining_str)
     
      for word in words:
          if ( word.istitle() or word.isupper() or word[0].isupper() ) and word.lower() not in stopwords.words("english"):
             entities.append(word)
          else:
               break
      
      
      print("entities = ",entities)
     

def find_amongand(sent_tokenize_list,sentence_number,legal_words):
      entities = []
      
      sen = sent_tokenize_list[sentence_number]
      #print(sen)
      by_match = re.search(r"[^a-zA-Z](among)[^a-zA-Z]", sen.lower())
      end_by = by_match.end(1)
      and_match = re.search(r"[^a-zA-Z](and)[^a-zA-Z]",sen[end_by:].lower() )
      and_start = and_match.start(1)
      min_ind = and_start
      #comma_match  = re.search(r"[^a-zA-Z](,)[^a-zA-Z]", sen[end_by:])
      #if comma_match:
      #    min_ind = min(min_ind,comma_match.start(1))
      entity1 = sen[end_by:end_by+min_ind]
      entities.append(entity1)
      
      remaining_str = sen[end_by+min_ind:len(sen)-1]
      #print("remaining = ",remaining_str)
     
      and_count = remaining_str.lower().count('and ')
      #print("and count",and_count)
      word_flag = 0
      if (and_count)>0:
          and_list=[]
          for m in re.finditer('and', remaining_str.lower()):
             and_list.append((m.start(),m.end())) 
          for ian,an in enumerate(and_list):
               if word_flag == 1:
                    break
               if ian ==0 and and_count>1:
                    
                    words = nltk.word_tokenize(remaining_str[0:an[0]])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if (word.istitle() or word.isupper() or  one_cap_least(word)) and word not in legal_words:
                               if word.title() not in calendar.month_name :
                                  rem.append(word)
                               else:
                                    word_flag = 1
                                    break
               elif ian == 0 and and_count == 0:
                    
                    words = nltk.word_tokenize(remaining_str[an[1]:])
                    #print("words 0 =",words)
                    rem = []
                    for word in words:
                             if (word.istitle() or word.isupper() or  one_cap_least(word)) and word not in legal_words :
                               if word.title() not in calendar.month_name :
                                  rem.append(word)
                               else:
                                    word_flag = 1
                                    break
               elif ian == (len(and_list)-1):
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] :    ])
                    #print("words 1 =",words)
                    rem = []
                    for word in words:
                             if (word.istitle() or word.isupper() or  one_cap_least(word)) and word not in legal_words :
                               if word.title() not in calendar.month_name :
                                  rem.append(word)
                               else:
                                    word_flag = 1
                                    break
               else:
                    words = nltk.word_tokenize(remaining_str[and_list[ian-1][1] : and_list[ian][0]   ])
                    #print("words 2 =",words)
                    rem = []
                    for word in words:
                             if (word.istitle() or word.isupper() or  one_cap_least(word)) and word not in legal_words:
                               if word.title() not in calendar.month_name :
                                  rem.append(word)
                               else:
                                    word_flag = 1
                                    break
                                   
               #print("rem = ",' '.join(rem))
               entities.append(' '.join(rem))
      print("entities = ",entities)

          
'''
def consec_trimmer(nes,index_list):
     if len(nes) != len(index_list):
          print("error")
     sen_ids = sorted(list(set([item[1] for item in index_list])))
     for i,ind in enumerate(index_list):
'''
