#!/usr/bin/env python3

# for python3 needs mysqlclient
# on mac this requires
# export DYLD_LIBRARY_PATH=/usr/local/mysql/lib:$DYLD_LIBRARY_PATH

import MySQLdb
import sys
import os
import config
import nltk
import re
import json
from pprint import pprint
from nltk.tokenize import PunktSentenceTokenizer

testdocs_query = "select s.id, s.title, s.doc_id from dreports d, dreports_summaries dr, summaries s where d.id = dr.dreport_id and dr.summary_id = s.id and s.status = 'COMPLETE' and d.type = 1 and instr(s.ant_types,'sublicense')"
print(testdocs_query)

DOC_LIMIT = 20
SENT_LIMIT = 5
def main(eb_files):
  print("eb_files is: ",eb_files)
  sum_dir = eb_files + 'summaries/'
  connection = MySQLdb.connect (
    host = config.dbinfo['host'],
    user = config.dbinfo['user'],
    passwd = config.dbinfo['passwd'],
    db =config.dbinfo['db'])
  cursor = connection.cursor()
  cursor.execute(testdocs_query)
  test_docs = cursor.fetchall()
  cursor.close()
  sent_tokenizer = PunktSentenceTokenizer()
  for i,doc in enumerate(test_docs):


    print('sumid: %d, title: %s, docid: %d' % (doc[0],doc[1],doc[2]))
    (sumid,title,doc_id) = doc

    text_file = sum_dir+str(sumid)+'.txt'
    file = open(text_file)
    text = file.read()

    ant_fn = sum_dir+str(sumid)+'.ant'
    with open(ant_fn) as ant_file:
      ants = json.load(ant_file)

    for ant in ants:
      if ant['type'] == 'change_control':
        print("CHANGE CONTROL: ",stripSingleLine(ant['text']))

    sent_spans = sent_tokenizer.span_tokenize(text)

    # for each sentence, if any annotations overlap it, print the ones that overlap
    # and the sentence
    for span in sent_spans:
      types = []
      for ant in ants:
        a_s = ant['start']
        a_e = ant['end']
        s_s = span[0]
        s_e = span[1]
        # if they overlap, add the type
        if a_s >= s_s and a_s < s_e or s_s >= a_s and s_s < a_e:
          types.append(ant['type'])
#          pprint(ant)
      if(len(types) > 0):
        print("SENTENCE: " + stripSingleLine(text[s_s:s_e]) + "\n")
        print("TYPES: ",types,"\n\n\n")
      
      
    if i > DOC_LIMIT:
      break

def stripSingleLine(string):
  return re.sub('\s+',' ',string)


if __name__ == '__main__':
    main(sys.argv[1])
