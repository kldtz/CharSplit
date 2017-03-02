from scipy import sparse
import numpy
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.cross_validation import cross_val_predict
from fractions import Fraction

import ebrevia.learn.util as util
import ebrevia.learn.readarff as readarff

SMALL_N = 30

class Sgd:
  def predict(self,Xbigram_te,X_te,Y_te,overrides):
    print("original shape of X = ",X_te.shape)
    X_te = X_te[:, self.cols_to_keep] #  remove cols where sum is zero
    print("shape of Xbigram",Xbigram_te.shape)
    X_te = sparse.hstack((X_te,Xbigram_te))
    print("new shape of X = ",X_te.shape)
    X_te = X_te.tolil()

    sgd_pred = self.sgd.predict(X_te)
    probs = self.sgd.predict_proba(X_te)[:,1]    
    recall_rate = sklearn.metrics.recall_score(Y_te,sgd_pred)
    print("recall_rate no threshold ",recall_rate)
    precision_rate = sklearn.metrics.precision_score(Y_te,sgd_pred)
    print("precision_rate no threshold ",precision_rate)
    f1 = sklearn.metrics.f1_score(Y_te,sgd_pred)
    print("f1 score ",f1)
    
    self.printWithThreshold(probs,Y_te,overrides)
    self.applyOverrides(probs, overrides)
    return probs
  
  def applyOverrides(self,probs,overrides):
    for i,override in enumerate(overrides):
      if override != None:
        probs[i] = override
      
  def printWithThreshold(self,probs,Y_te,overrides):
    # compute thresholded recall/precision
    THRESHOLD = .06
    rec_den = 0
    prec_den = 0
    tp = 0
    o_prec_den = 0
    o_tp = 0

    for i,pred in enumerate(list(probs)):
      if Y_te[i] == 1:
        rec_den += 1
      if pred > THRESHOLD:
        prec_den += 1
        if Y_te[i] == 1:
          tp += 1
      if (pred > THRESHOLD and overrides[i] != 0) or overrides[i] == 1:
        o_prec_den += 1
        if Y_te[i] == 1:
          o_tp += 1
          
    print ("rec_den: " ,rec_den)
    print ("pred_den: " ,prec_den)
    print ("tp: ",tp)
    print ("o_prec_den: " ,o_prec_den)
    print ("o_tp: ",o_tp)

    if(rec_den > 0):
      adj_rec = tp / rec_den
      o_rec = o_tp / rec_den
    else:
      adj_rec = 0
      o_rec = 0

    if(prec_den > 0):
      adj_prec = tp / prec_den
    else:
      adj_prec = 0

    if(o_prec_den > 0):
      o_prec = o_tp / o_prec_den
    else:
      o_prec = 0
      
    print ("TH:",THRESHOLD)
    print ("adjusted recall: " , adj_rec)
    print ("adjusted prec: " , adj_prec)
    print ("overridden recall: " , o_rec)
    print ("overridden prec: " , o_prec)

  def trainAndTest(self,input_file,Xbigram,X,Y,Xbigram_te=None,X_te=None,Y_te=None,overrides=None):    
    
    multiplier = 0
    iterations = 50
    print("shape of Xbag = ",X.shape)
    colSum = X.sum(axis=0)
    colSum = numpy.squeeze(numpy.asarray(colSum))
    zerofind = list(numpy.where(colSum==0))
    all_cols = numpy.arange(X.shape[1])
    self.cols_to_keep = numpy.where(numpy.logical_not(numpy.in1d(all_cols, zerofind)))[0]
    X = X[:, self.cols_to_keep] #  remove cols where sum is zero
    print("zerofind= ",zerofind)
    print("shape of Xbigram",Xbigram.shape)
    X = sparse.hstack((X,Xbigram),format='csr')
    print("combined shape of X = ",X.shape)
    # end of removal zero columns in both X and header_list
    # lets do feature selection
    #b = SelectKBest(score_func=chi2,k=5000)
    #b.fit_transform(X,Y)      
         
    
    if(Y_te is not None):
      X_tr = X
      Y_tr = Y
      print("original shape of X = ",X_te.shape)
      X_te = X_te[:, self.cols_to_keep] #  remove cols where sum is zero
      print("zerofind= ",zerofind)
      print("shape of Xbigram",Xbigram_te.shape)
      X_te = sparse.hstack((X_te,Xbigram_te),format='csr')
      print("new shape of X = ",X_te.shape)
    else:
      X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,test_size = 0.3,random_state=42)
     
    print("multiplier = ",multiplier)
    print("number of iterations = ",iterations)        
    if(multiplier > 0):
      X_tr,Y_tr = multiply(X_tr,Y_tr,multiplier)
    print("Start SGD")

    parameters = {'alpha': 10.0**-numpy.arange(3,8)}
    sgd_clf = SGDClassifier( loss='log',alpha=10**(-8), n_iter=iterations, penalty='l2', shuffle=True)
#    parameters = {'C': [.01,.1,1,10,100]}
#    sgd_clf = LogisticRegression()

    num_positive = numpy.count_nonzero(Y_tr) + numpy.count_nonzero(Y_te)
    # small dataset mode (useful for self-training)
    if(num_positive < SMALL_N):
      print("dataset too small, using cross validation instead")
      # combine them
      print("X shape",X_tr.shape,X_te.shape)
      print("Y shape",Y_tr.shape,Y_te.shape)

      X_comb = sparse.vstack((X_tr,X_te),format='csr')
      Y_comb = numpy.concatenate((Y_tr,Y_te))
      print("X shape",X_comb.shape)
      print("Y shape",Y_comb.shape)
      sgd_pred = cross_val_predict(sgd_clf,X_comb,Y_comb,cv=5)
      Y_stats = Y_comb
      sgd_clf.fit(X_comb,Y_comb)
      self.sgd = sgd_clf

    else:
      # this performs better with loss='hing' but that doesn't
      # provide probabilities
      gs_clf = GridSearchCV(sgd_clf,parameters,n_jobs=-1,scoring='f1')
      gs_clf = gs_clf.fit(X_tr,Y_tr)
      sgd_pred = gs_clf.predict(X_te)
      Y_stats = Y_te

      print("grid scores are: ")
      for params,mean_score,scores in gs_clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score,scores.std() / 2, params))
      best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
      print("best params are: ")
      for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

      probs = gs_clf.predict_proba(X_te)[:,1]
      self.printWithThreshold(probs,Y_te,overrides)
      self.sgd = gs_clf

    self.stats = {
      'recall':sklearn.metrics.recall_score(Y_stats,sgd_pred),
      'precision':sklearn.metrics.precision_score(Y_stats,sgd_pred),
      'fscore':sklearn.metrics.f1_score(Y_stats,sgd_pred),
      'confusion_matrix':sklearn.metrics.confusion_matrix(Y_stats,sgd_pred).tolist()}
    print("recall_rate no threshold ",self.stats['recall'])
    print("precision_rate no threshold ",self.stats['precision'])
    print("fscore no threshold ",self.stats['fscore'])


def multiply(X_tr,Y_tr,multiplier):    
  #  separate positives from negatives
  z=numpy.where(Y_tr==1)
  ls = list(z[0])
  Y_pos = numpy.ones(len(ls)*multiplier)
  X_pos = X_tr[ls,:]
  # add the positives to the training set
  matrices = [X_tr]
  for item in range(multiplier):
    matrices.append(X_pos)
  Y_tr = numpy.concatenate((Y_tr,Y_pos),axis=0)
  X_tr = sparse.vstack(matrices,format='coo')
  print("new shape of X = ",X_tr.shape)
  return X_tr,Y_tr
    
def printPosPred(train_prefix,X_tr,Y_tr):
  X_tr = X_tr.tolil()
  #  separate positives from negatives
  z=numpy.where(Y_tr==1)
  ls = list(z[0])
  Y_pos = Y_tr[ls]
  X_pos = X_tr[ls,:]

  pos_pred = self.sgd.predict(X_pos)
  error_loc = numpy.where(pos_pred==0)
  ls1 = numpy.array(ls)
  er_loc = list( ls1[error_loc])
  ##########################################################
  bag_of_words=[]
  output_file = train_prefix+'change_of_control.'
  with open(input_file, newline='', encoding='utf-8') as f:
    reader,headers = readarff.getArffReader(f)
    bag_of_words_column=headers.index('bag_of_words')
    for  row in reader:
      row_bag = row[bag_of_words_column].replace('\u201c', '"')
      row_bag = row_bag.replace('\u201d', '"')
      row_bag = row_bag.replace('\u2019', "'")
      bag_of_words.append(row_bag)
  f.close()
  f=open(output_file, 'wt', encoding='utf-8') 
  for loc in er_loc:
    f.write(bag_of_words[loc])
    f.write('#########################################################################' +'\n')
  f.close()

if __name__ == "__main__":  
  train_prefix = util.get_commandline_prefix()
  input_file = train_prefix+'new.csv'
  sgd(input_file,train_prefix)  
