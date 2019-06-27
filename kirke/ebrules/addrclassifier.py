#!/usr/bin/env python3

import re
import pandas as pd
import numpy as np

from nltk.tokenize import TreebankWordTokenizer

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from kirke.utils import osutils

IS_DEBUG_ADDRESS = False

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
        tmp_kwd_list = []  # type: List[str]
        for kwd in keywords[category]:
            kwd = kwd.strip()
            # don't title 'IN' or 'OH'
            if not (kwd.isupper() and len(kwd) == 2):
                tmp_kwd_list.append(kwd.title())
            tmp_kwd_list.append(kwd.upper())
            # in case there is lower, such as 'rue' in French
            tmp_kwd_list.append(kwd)
        keywords[category] = list(set(tmp_kwd_list))

    if IS_DEBUG_ADDRESS:
        for akey in sorted(keywords.keys()):
            aval = keywords[akey]
            print('addr keywords [{}], len = {}]'.format(akey, len(aval)))
            print('addr keywords [{}] = {}'.format(akey, sorted(aval)))
    return keywords

KEYWORDS = load_keywords()

ADDR_ZIP_PAT = re.compile(r'\b({}|{}|{})\b'.format(US_ZIP, UK_STD, CAN_STD))
ADDR_NUM_PAT = re.compile(r'\b(\d+|one|two)\b', re.I)
ADDR_COUNTRY_PAT = re.compile(r'\b({})\b'.format('|'.join(KEYWORDS['country_names'])))
ADDR_US_PAT = re.compile(r'\b({})\b'.format('|'.join(KEYWORDS['us'])))
ADDR_UK_PAT = re.compile(r'\b({})\b'.format('|'.join(KEYWORDS['uk'])))
ADDR_CAN_PAT = re.compile(r'\b({})\b'.format('|'.join(KEYWORDS['can'])))
ADDR_APT_PAT = re.compile(r'\b({})\b'.format('|'.join(KEYWORDS['apt_abbrs'])))
ADDR_ROAD_PAT = re.compile(r'\b({})\b'.format('|'.join(KEYWORDS['road_abbrs'])))


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
            orig_addr = addrs
            addr = ' '.join(TREEBANK_WORD_TOKENIZER.tokenize(addr))
            addr = re.sub(ADDR_ZIP_PAT, '[ZIP]', addr)
            addr = re.sub(ADDR_NUM_PAT, '[NUM]', addr)
            addr = re.sub(ADDR_COUNTRY_PAT, '[COUNTRY]', addr)
            addr = re.sub(ADDR_US_PAT, '[US]', addr)
            addr = re.sub(ADDR_UK_PAT, '[UK]', addr)
            addr = re.sub(ADDR_CAN_PAT, '[CAN]', addr)
            addr = re.sub(ADDR_APT_PAT, '[APT]', addr)
            addr = re.sub(ADDR_ROAD_PAT, '[ROAD]', addr)

            if IS_DEBUG_ADDRESS:
                print('orig_addr: [{}]'.format(orig_addr))
                print('     addr: [{}]'.format(addr))
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
        if IS_DEBUG_ADDRESS:
            print('addr.predict() = {}'.format(probs[1]))
        return probs, pred_label
