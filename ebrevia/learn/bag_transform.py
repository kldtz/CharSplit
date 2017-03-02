from scipy import sparse
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
import numpy 

import ebrevia.learn.util as util
import ebrevia.learn.readarff as readarff

class BagTransform:

  def __init__(self,newschool):
    self.newschool = newschool

  def storeBagMatrix(self,input_file):
    return self._storeBagMatrix(input_file,False)
    
  def storeBagMatrixTest(self,input_file):
    return self._storeBagMatrix(input_file,True)
    
  def _storeBagMatrix(self,input_file,testmode):
    print("start store bag matrix")  
    number_of_top_words=25000
    header_list=[]
    sentences = []
    set_col = [0,1,2,3,8]
    categorical_col = list(range(9,17))
    numeric_col =[5,6,7]
    numeric_col.extend(list(range(17,23)))

    if(self.newschool):
      set_col = []
      categorical_col = []
      numeric_col = []
    Ys={}

    count=0
    with open(input_file, newline='', encoding='utf-8') as f:
      reader,header_list = readarff.getArffReader(f)
      for row in reader:
        count = count+1            
    f.close()
    
    
    cols = len(set_col)+len(numeric_col)
    yes_no_matrix = numpy.zeros(shape=(count,cols)) # turn yes and no in (1,0) numpy mat
    categorical_matrix = numpy.zeros(shape=(count,len(categorical_col)))
    numrows = count
    count=0
    
    with open(input_file, newline='', encoding='utf-8') as f:
      reader,headers = readarff.getArffReader(f)
      sentence_column=headers.index('bag_of_words')
      
      # find the Y's      
      for header in headers:
        if(header.startswith('ebrevia_entity_')):
          provision_name = header[len('ebrevia_entity_'):]            
          Ys[provision_name] = numpy.zeros(numrows,dtype=bool)
          # actually should initialize this to an appropriate numpy array
            
      for row in reader:
        sentence = row[sentence_column].replace('\u201c', '"')
        sentence = sentence.replace('\u201d', '"')
        sentence = sentence.replace('\u2019', "'")
        sentences.append(sentence)
        for i,header in enumerate(headers):
          if(header.startswith('ebrevia_entity_')):
            provision_name = header[len('ebrevia_entity_'):]            
            if ('yes' in row[i]):
              Ys[provision_name][count] = True              
        for ibinary,binary in enumerate(set_col):
          if ('yes' in row[binary]):
            yes_no_matrix[count,ibinary]=1
        for inumer,numer in enumerate(numeric_col):
          yes_no_matrix[count,(inumer+len(set_col))]=row[numer]
        for icat,cat in enumerate(categorical_col):
          categorical_matrix[count,icat] = row[cat]
        count =count+1
            
    f.close()

    if(testmode):
      if(len(set_col) > 0):
        yes_no_matrix = self.min_max_scaler.transform(yes_no_matrix)
      if(len(categorical_col) > 0):        
        categorical_matrix = self.enc.transform(categorical_matrix)
    else:
      self.min_max_scaler = preprocessing.MinMaxScaler()
      if(len(set_col) > 0):
        yes_no_matrix= self.min_max_scaler.fit_transform(yes_no_matrix)
      if(len(categorical_col) > 0):
        self.enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
        categorical_matrix = self.enc.fit_transform(categorical_matrix)
    
    if(not testmode):
      #self.vectorizer = CountVectorizer(max_features=number_of_top_words,stop_words='english',lowercase=False)

      self.vectorizer = CountVectorizer(max_features=number_of_top_words,stop_words='english',lowercase=False,ngram_range=(1,2),token_pattern=r'\b\w+\b')

      bow_matrix = self.vectorizer.fit_transform(sentences)
      # make pickling smaller (this attr is only needed for introspection)
      self.vectorizer.stop_words_ = None
    else:
      bow_matrix = self.vectorizer.transform(sentences) 
    
    # concatenate yes_no_matrix + bow_matrix
    if(len(set_col) > 0):
      bag_matrix = sparse.hstack((yes_no_matrix,categorical_matrix,bow_matrix))
    else:
      bag_matrix = bow_matrix
    
    
    del sentences    
    del yes_no_matrix
    del categorical_matrix
    
        
    ##############################
    ##########  write matrix,lists
    ##############################
    bag_matrix = sparse.csr_matrix(bag_matrix)

    return bag_matrix,Ys

      
if __name__ == "__main__":  
  input_file = output_prefix+'new.csv'
  store_bag_matrix(input_file)  
