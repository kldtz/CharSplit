import ebrevia.learn.bag_transform as bag_transform
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
import ebrevia.learn.readarff as readarff
import numpy
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import *
from sklearn import svm

class LemmaTokenizer(object):
  def __init__(self):
    self.wnl = WordNetLemmatizer()
  def __call__(self, doc):
    return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
  
def readfile(input_file,provision):
  count=0
  with open(input_file, newline='', encoding='utf-8') as f:
    reader,header_list = readarff.getArffReader(f)
    for row in reader:
      count = count+1            
  f.close()
  numrows = count
  count = 0

  sentences = []
  with open(input_file, newline='', encoding='utf-8') as f:
    reader,headers = readarff.getArffReader(f)
    sentence_column=headers.index('bag_of_words')
      
    # find the Y's      
    for header in headers:
      if(header.startswith('ebrevia_entity_'+provision)):
        Y = numpy.zeros(numrows,dtype=bool)
        # actually should initialize this to an appropriate numpy array


    for row in reader:
      sentence = row[sentence_column].replace('\u201c', '"')
      sentence = sentence.replace('\u201d', '"')
      sentence = sentence.replace('\u2019', "'")
      sentences.append(sentence)
      for i,header in enumerate(headers):
        if(header.startswith('ebrevia_entity_'+provision)):
          if ('yes' in row[i]):
            Y[count] = True
      count += 1

  return (sentences,Y)

  

provision = 'change_control'

tr_file = '/Users/jakem/eb-files/models/sentencesV1_l_address_only.arff'
te_file = '/Users/jakem/eb-files/models/test-data/sentencesV1_l_address_only.arff'
tr_file = '/Users/jakem/eb-files/models/sentencesV1-nonsparse.arff'
te_file = '/Users/jakem/eb-files/models/test-data/sentencesV1-nonsparse.arff'
X_tr,Y_tr = readfile(tr_file,provision)


X_te,Y_te = readfile(te_file,provision)

#pipeline = Pipeline([
#    ('vect', CountVectorizer()),
#    ('tfidf', TfidfTransformer()),
#    ('clf', SGDClassifier()),
#])

# cv = CountVectorizer(max_features=25000,
#                              stop_words='english',
#                              lowercase=True,
#                              ngram_range=(1,1),
#                              token_pattern=r'\b\w+\b',
#                              binary=True,
#                              vocabulary=['roof','roofs','rooftop','antenna','antennas','antennae','tenant','install','installed','installation','installing','right','use'])

cv = CountVectorizer(max_features=25000,
                             stop_words='english',
                             lowercase=False,
                             ngram_range=(1,2),
                             token_pattern=r'(?u)\b\w+\b',
                             binary=False)

#cv.fit(X_tr)
#trans = cv.transform(X_te)
#for i,sent in enumerate(X_te):
#  if Y_te[i]:
#    print ("POSTIIVE: ",sent,"\n")
#    print ("tranformed: ",trans[i])

clf = LogisticRegression(class_weight='auto')
#clf = SGDClassifier( loss='log',alpha=10**(-5), n_iter=50, penalty='l2',
#                   shuffle=True,class_weight='auto')
#clf = MultinomialNB()

pipeline = Pipeline([
    ('vect', cv),
    ('clf', clf)
])
#pipeline = Pipeline([
#    ('vect', cv),
#    ('clf', DecisionTreeClassifier(class_weight='auto'))
#])
#pipeline = Pipeline([
#    ('vect', cv),
#    ('clf', svm.SVC(class_weight='auto'))
#])


#pipeline = Pipeline([
#    ('vect', CountVectorizer(max_features=number_of_top_words,stop_words='english',lowercase=True,ngram_range=(1,1),token_pattern=r'\b\w+\b',vocabulary=['roof','roofs','antenna','antennas','tenant','install','installed','right','use'])),
#    ('clf', LogisticRegression(class_weight='auto')),
#])


#for model in [
#    LogisticRegression(class_weight='auto'),
#    RandomForestClassifier(),
#    DecisionTreeClassifier(class_weight='auto'),
#    SGDClassifier( loss='log',alpha=10**(-5), n_iter=100, penalty='elasticnet',
#                   shuffle=True,class_weight='auto')]:
#print (Y_tr)                     
pipeline.fit(X_tr,Y_tr)

pred = pipeline.predict(X_te)

recall_rate = sklearn.metrics.recall_score(Y_te,pred)
print("recall ",recall_rate)
precision_rate = sklearn.metrics.precision_score(Y_te,pred)
print("precision ",precision_rate)

feature_names = numpy.asarray(cv.get_feature_names())
print("top 30 keywords per class:")
top30 = numpy.argsort(clf.coef_[0])[-30:]
print(feature_names[top30])

  

