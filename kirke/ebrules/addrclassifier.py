#!/usr/bin/env python3

from collections import defaultdict
import pickle
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
import pandas as pd
import random
import re
from scipy import sparse
from sklearn import preprocessing
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection.from_model import SelectFromModel

DATA_DIR = './dict/addresses/'

US_ZIP = '\d{5}(-\d{4})?'
UK_STD = '[A-Z]{1,2}[0-9ROI][0-9A-Z]? *(?:(?![CIKMOV])[A-Z]?[0-9O][a-zA-Z]{2})'
CAN_STD = '[A-Z][0-9][A-Z] +[0-9][A-Z][0-9]'
TREEBANK_WORD_TOKENIZER = TreebankWordTokenizer()

def load_keywords():
    # Create a dictionary object to return
    keywords = {}

    # Read constituencies (e.g. states, provinces) and their abbreviations
    countries = ['us', 'uk', 'aus', 'can']
    keywords['constituencies'] = []
    for cat in countries:
        # pylint: disable=invalid-name
        df = pd.read_csv(DATA_DIR + 'constituencies_' + cat + '.csv').dropna()
        keywords[cat] = df['name'].tolist() + df['abbr'].tolist()
        keywords['constituencies'] += keywords[cat]

    # Read single-column CSVs
    categories = ['address_terms', 'apt_abbrs', 'apt_terms',
                  'business_suffixes', 'country_names', 'numbers', 'road_abbrs']
    for cat in categories:
        # pylint: disable=invalid-name
        df = pd.read_csv(DATA_DIR + cat + '.csv', header=None).dropna()
        keywords[cat] = df[0].tolist()

    keywords['can'] += ['Toronto', 'Vancouver']
    keywords['apt_abbrs'] += ['P.O.', 'PO', 'Box', 'P.', 'O.']
    keywords['road_abbrs'] += ['Broadway', 'Republic', 'N.E.', 'N.W.', 'S.E.', 'S.W.', 'NE', 'NW', 'North', 'South', 'East', 'West', 'N.', 'S.', 'E.', 'W.']

    # Save title case and uppercase versions, padded, for each keyword
    for category in keywords:
        title_case_keywords = [kwd.title().strip() for kwd in keywords[category]]
        uppercase_keywords = [kwd.upper().strip() for kwd in keywords[category]]
        keywords[category] = list(set(title_case_keywords + uppercase_keywords))

    return keywords

KEYWORDS = load_keywords()

class LogRegModel:
    def __init__(self,model = None, vec= None, featureselector= None):
      if (model is None):
          self.model = LogisticRegression()
          self.ngram_vec = CountVectorizer(min_df=2, ngram_range=(1,4), lowercase=False)
          self.min_max_scaler = preprocessing.MinMaxScaler()
      else:
        self.model = model
        self.ngram_vec = ngram_vec

    # return generic version of addr to be put into ngram vectorizer
    def extract_features(self, addrs):
        ngram_features = []
        num_features = np.zeros(shape=(len(addrs), 6))
        for i, addr in enumerate(addrs):
            addr = " ".join(TREEBANK_WORD_TOKENIZER.tokenize(addr))
            addr = re.sub(r'\b({}|{}|{})\b'.format(US_ZIP, UK_STD, CAN_STD), '[ZIP]', addr)
            addr = re.sub(r'\b(\d+|one|two)\b', '[NUM]', addr, re.I)
            addr = re.sub(r'\b({})\b'.format('|'.join(KEYWORDS['country_names'])), '[COUNTRY]', addr)
            addr = re.sub(r'\b({})\b'.format('|'.join(KEYWORDS['us'])), '[US]', addr)
            addr = re.sub(r'\b({})\b'.format('|'.join(KEYWORDS['uk'])), '[UK]', addr)
            addr = re.sub(r'\b({})\b'.format('|'.join(KEYWORDS['can'])), '[CAN]', addr)
            addr = re.sub(r'\b({})\b'.format('|'.join(KEYWORDS['apt_abbrs'])), '[APT]', addr)
            addr = re.sub(r'\b({})\b'.format('|'.join(KEYWORDS['road_abbrs'])), '[ROAD]', addr)
            ngram_features.append(addr)
            
            addr_split = addr.split()
            num_features[i, 0] = len(addr_split)
            num_features[i, 1] = addr_split.count('[NUM]')
            num_features[i, 2] = addr_split[0] == '[NUM]'
            num_features[i, 3] = (addr_split[-1] == '[ZIP]' or addr_split[-1] == '[COUNTRY]')
            num_features[i, 4] = addr_split.count('[COUNTRY]')
            num_features[i, 5] = addr_split.count('[ZIP]')
 
        return ngram_features, num_features

    def fitModel(self, train, labels):
        ngram_feats, num_feats = self.extract_features(train)
        tr_ngram_feats = self.ngram_vec.fit_transform(ngram_feats).toarray().astype(np.float)
        tr_num_feats = self.min_max_scaler.fit_transform(num_feats).astype(np.float)
        X = np.append(tr_ngram_feats, tr_num_feats, 1)
        y = np.array(labels).astype(np.float)

        # deterministic without RF feature selection
        #self.featSelect = SelectFromModel(RandomForestClassifier()).fit(X,y)
        #X = self.featSelect.transform(X)
        
        self.model.fit(X, y)

    def predict(self, addr):
        ngram_feats, num_feats = self.extract_features([addr])
        x_ngram_feats = self.ngram_vec.transform(ngram_feats).toarray().astype(np.float)
        x_num_feats = self.min_max_scaler.transform(num_feats).astype(np.float)
        x = np.append(x_ngram_feats, x_num_feats, 1)
        #x = self.featSelect.transform(x)
        probs = self.model.predict_proba(x)[0]
        pred_label = int(self.model.predict(x)[0])
        return probs, pred_label

# read, shuffle, and separate training and dev data
def parse_train(fname):
    train_data = []
    train_labels = []
    dev_data = []
    dev_labels = []

    with open(fname) as train:
        train_list = train.readlines()
        cutoff = int(len(train_list) * 0.9)
        random.Random(0).shuffle(train_list)
        train = train_list[:cutoff]
        dev = train_list[cutoff:]

        for line in train:
            data, label = line.split("\t")
            train_data.append(data)
            train_labels.append(int(label.strip()))
        for line in dev:
            data, label = line.split("\t")
            dev_data.append(data)
            dev_labels.append(int(label.strip()))

    return train_data, train_labels, dev_data, dev_labels


if __name__ == '__main__':
    model = LogRegModel()
    train_data, train_labels, dev_data, dev_labels = parse_train("addr_annots.tsv")
    model.fitModel(train_data, train_labels)
    
    tp, fp, fn = 0, 0, 0
    fps = []
    fns = []
    for i, addr in enumerate(dev_data):
        gold_label = dev_labels[i]
        probs, pred_label = model.predict(addr)
        if gold_label == 1 and pred_label == 1:
            tp += 1
        elif gold_label == 1 and pred_label == 0:
            fns.append([addr, probs[1]])
            fn += 1
        elif gold_label == 0 and pred_label == 1:
            fps.append([addr, probs[1]])
            fp += 1
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * ((prec * rec)/(prec+rec))
    print("TP = {}, FP = {}, FN = {}".format(tp, fp, fn))
    print("P = {}, R = {}, F = {}".format(prec, rec, f1))

    outfile = open('addr_classifier.pkl', 'wb')
    pickle.dump(model, outfile, -1)
    outfile.close()

 
