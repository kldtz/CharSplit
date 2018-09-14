#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np
from nltk.tokenize import TreebankWordTokenizer

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from kirke.utils import osutils

DATA_DIR = './dict/addresses/'

US_ZIP = r'\d{5}(-\d{4})?'
UK_STD = r'[A-Z]{1,2}[0-9ROI][0-9A-Z]? *(?:(?![CIKMOV])[A-Z]?[0-9O][a-zA-Z]{2})'
CAN_STD = r'[A-Z][0-9][A-Z] +[0-9][A-Z][0-9]'
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
    keywords['road_abbrs'] += ['Broadway', 'Republic', 'N.E.', 'N.W.', 'S.E.',
                               'S.W.', 'NE', 'NW', 'North', 'South', 'East',
                               'West', 'N.', 'S.', 'E.', 'W.']

    # Save title case and uppercase versions, padded, for each keyword
    for category in keywords:
        title_case_keywords = [kwd.title().strip() for kwd in keywords[category]]
        uppercase_keywords = [kwd.upper().strip() for kwd in keywords[category]]
        keywords[category] = list(set(title_case_keywords + uppercase_keywords))

    return keywords

KEYWORDS = load_keywords()

class LogRegModel:
    def __init__(self) -> None:
        self.model = LogisticRegression()
        self.ngram_vec = CountVectorizer(min_df=2, ngram_range=(1, 4), lowercase=False)
        self.min_max_scaler = preprocessing.MinMaxScaler()

    def save_model_file(self, model_file_name: str) -> None:
        osutils.joblib_atomic_dump(self, model_file_name)
        print('wrote model file: {}'.format(model_file_name))

    # return generic version of addr to be put into ngram vectorizer
    # pylint: disable=no-self-use
    def extract_features(self, addrs):
        ngram_features = []
        num_features = np.zeros(shape=(len(addrs), 6))
        for i, addr in enumerate(addrs):
            addr = " ".join(TREEBANK_WORD_TOKENIZER.tokenize(addr))
            addr = re.sub(r'\b({}|{}|{})\b'.format(US_ZIP, UK_STD, CAN_STD), '[ZIP]', addr)
            addr = re.sub(r'\b(\d+|one|two)\b', '[NUM]', addr, re.I)
            addr = re.sub(r'\b({})\b'.format('|'.join(KEYWORDS['country_names'])),
                          '[COUNTRY]', addr)
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

    def fit_model(self, train, labels):
        ngram_feats, num_feats = self.extract_features(train)
        tr_ngram_feats = self.ngram_vec.fit_transform(ngram_feats).toarray().astype(np.float)
        tr_num_feats = self.min_max_scaler.fit_transform(num_feats).astype(np.float)
        # pylint: disable=invalid-name
        X = np.append(tr_ngram_feats, tr_num_feats, 1)
        # pylint: disable=invalid-name
        y = np.array(labels).astype(np.float)

        # deterministic without RF feature selection
        #self.featSelect = SelectFromModel(RandomForestClassifier()).fit(X,y)
        #X = self.featSelect.transform(X)

        self.model.fit(X, y)

    def predict(self, addr):
        ngram_feats, num_feats = self.extract_features([addr])
        x_ngram_feats = self.ngram_vec.transform(ngram_feats).toarray().astype(np.float)
        x_num_feats = self.min_max_scaler.transform(num_feats).astype(np.float)
        # pylint: disable=invalid-name
        x = np.append(x_ngram_feats, x_num_feats, 1)
        #x = self.featSelect.transform(x)
        probs = self.model.predict_proba(x)[0]
        pred_label = int(self.model.predict(x)[0])
        return probs, pred_label
