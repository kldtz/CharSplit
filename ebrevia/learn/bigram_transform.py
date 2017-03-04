import re

from nltk import FreqDist
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy import sparse

import ebrevia.learn.readarff as readarff


# This is a customized version of "bigrams" that Luigi came up with.
# Normal bigrams just find 2-word sequences within a document.  This
# follows a different path:
#   1. Find all the words in the positive sentences, filtering stopwords and short words
#   2. Find the n_bigram_words most common words in this set
#   3. Finds all bigrams of only these common words, where a bigram means that both words
#      are substrings of the given sentence (they might be part of a bigger word)


class BigramTransform:
  def __init__(self,provision):    
    self.n_bigram_words = 175
    self.provision = provision
    
  def storeBigramMatrix(self,input_file):  
    sentences,sentences_positive = self.loadSentences(input_file)
    
    #
    ######    
    joined_sentences = " ".join(sentences_positive)
    positive_words = word_tokenize(joined_sentences)
    stops = set(stopwords.words('english'))
    filtered_list = [w for w in positive_words if not w.lower() in stops and len(w)>3]
    fdistribution= FreqDist(filtered_list)
    top_words = fdistribution.most_common(self.n_bigram_words)
    self.top_100_words  = [item[0] for item in top_words]    
    
    print("start filling bigram part")
    # for each sentence, find which top words it contains.  Then generate all pairs of these, 
    # and generate the sparse matrix row entries for the rows it contains.
    indptr = [0]
    indices = []
    data = []
    self.vocabulary = {}
    
    for sentIdx,sent in enumerate(sentences):
      found_words = []
      for ind2,common_word in enumerate(self.top_100_words):
        if(common_word in sent):
          found_words.append(common_word)  
      for iw1,w1 in enumerate(found_words):
        for iw2 in range(iw1+1,len(found_words)):
          w2 = found_words[iw2]
          col_name = ",".join((w1,w2))                 
          index = self.vocabulary.setdefault(col_name,len(self.vocabulary))                    
          indices.append(index)
          data.append(1) 
      indptr.append(len(indices))         
    bigram_matrix = sparse.csr_matrix((data, indices, indptr), dtype=int)    

    print("done creating bigrams")
    
    return bigram_matrix

  def storeBigramMatrixTest(self,input_file):
    # if they're looking for the test matrix, just produce it here    
    sentences, sentences_positive = self.loadSentences(input_file)

    print("start filling bigrams test")
    # for each sentence, find which top words it contains.  Then generate all pairs of these, 
    # and generate the sparse matrix row entries for the rows it contains.
    indptr = [0]
    indices = []
    data = []
    overrides = [None for i in range(len(sentences))]

    num_p = re.compile(r'\d+(\.\d+)*')
    patterns = {'change_control':r'change\s+(of|in)\s+control',
                'confidentiality':r'(information.*confidential|confidential.*information)',
                'limliability':r'((is|are)\s+not\s+(liable|responsible)|' +
                'will\s+not\s+be\s+(held\s+)?(liable|responsible)|' +
                'no\s+(\S+\s+){1,5}(is|will\s+be)\s+responsible\s+for|' +
                'not\s+(be\s+)?required\s+to\s+make\s+(\S+\s+){1,3}payment|' +
                'need\s+not\s+make\s(\S+\s+){1,3}payment)',
                'term':r'[“"]Termination\s+Date[”"]'}
    global_min_length = 6
    min_pattern_override_length = 8
    if(self.provision == 'term'):
      min_pattern_override_length = 0
      
    pattern = None
    if self.provision in patterns:
      pattern = re.compile(patterns[self.provision],re.IGNORECASE | re.DOTALL)
    for sentIdx,sent in enumerate(sentences):
      found_words = []
      for ind2,common_word in enumerate(self.top_100_words):
        if(common_word in sent):
          found_words.append(common_word)  
      for iw1,w1 in enumerate(found_words):
        for iw2 in range(iw1+1,len(found_words)):
          w2 = found_words[iw2]
          col_name = ",".join((w1,w2))
          if col_name in self.vocabulary:       
            index = self.vocabulary[col_name]
            indices.append(index)
            data.append(1) 
      indptr.append(len(indices))
      toks = sent.split()
      num_words = len(toks)
               
      num_numeric = 0
      for tok in toks:
        if num_p.match(tok):
          num_numeric += 1              
      
      is_toc = num_words > 60 and num_numeric / num_words > .2
      if pattern and pattern.search(sent) and num_words > min_pattern_override_length and not is_toc:
        overrides[sentIdx] = 1
      if num_words < global_min_length:
        overrides[sentIdx] = 0
        
    bigram_matrix = sparse.csr_matrix((data, indices, indptr),shape=(len(sentences),len(self.vocabulary)), dtype=int)   
       
    print("done creating bigrams test")
    return bigram_matrix,overrides
    
  def loadSentences(self,input_file):    

    sentences = []
    sentences_positive = []
    with open(input_file, newline='', encoding='utf-8') as f:
      reader,headers = readarff.getArffReader(f)
      sentence_column=headers.index('bag_of_words')
      Y_column=headers.index('ebrevia_entity_'+self.provision)
      for  row in reader:
        sentence = row[sentence_column].replace('\u201c', '"')
        sentence = sentence.replace('\u201d', '"')
        sentence = sentence.replace('\u2019', "'")
        sentences.append(sentence)
        if ('yes' in row[Y_column]):
          sentences_positive.append(sentence)
        
    f.close()
    return sentences,sentences_positive
    
if __name__ == "__main__":
  input_file = output_prefix+'new.csv'
  bt = BigramTransform();
  bt.storeBigramMatrix(input_file)
