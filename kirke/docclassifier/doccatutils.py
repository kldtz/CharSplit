#!/usr/bin/env python

import json
import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from kirke.docclassifier import doccategory

# based on eval set, 250 seems to work best
# TEXT_SIZE = 1000
TEXT_SIZE = 250


def load_data(txt_fn_list_fn, is_step1=False):
    doc_text_list, catids_list = [], []

    catnames = []
    catname_catid_map = {}
    if is_step1:
        catnames, catname_catid_map, catid_catname_map = doccategory.init_doccats_step1()
    else:
        catnames, catname_catid_map, catid_catname_map = doccategory.load_doccats_prod()

    with open(txt_fn_list_fn, 'rt') as fin:
        for line in fin:
            txt_fn = line.strip()
            ebdata_fn = txt_fn.replace('.txt', '.ebdata')

            with open(ebdata_fn, 'rt') as ebdata_fin:
                tags = json.loads(ebdata_fin.read())['tags']
                tags = [tag.strip() for tag in tags]
                catids = doccategory.tags_to_catids(tags, catname_catid_map)
                # print('catids: [{}]'.format(catids))
                # if not catids:  # skip all document with no coretags
                #    continue
                catids_list.append(catids)

            with open(txt_fn, 'rt') as txt_fin:
                doc_text = txt_fin.read()
                doc_text_list.append(doc_text_to_docfeats(doc_text))

    # print('len(doc_text_list) = {}'.format(len(doc_text_list)))
    # print('len(catid_list) = {}'.format(len(catids_list)))

    return doc_text_list, catids_list, catnames


_STEMMER = SnowballStemmer("english")
_EN_STOPWORD_SET = stopwords.words('english')

def doc_text_to_docfeats(doc_text, wanted_text_len=TEXT_SIZE):
    lc_doc_text = doc_text.lower()
    # Based on the training and testing set, wanted_text_len = 250 is
    # the best (0.88), but our corpus might not reflect real life.
    # Currently, setting it to 1000 instead (0.85).
    # tried 100, 250, 500, 1000, 2000, 4000
    tokens = re.findall(r'\b[A-Za-z]+\b', lc_doc_text[:wanted_text_len])

    return ' '.join([_STEMMER.stem(tok) for tok in tokens if tok not in _EN_STOPWORD_SET])


SCORE_PAT = re.compile(r'avg / total\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
# return a tuple of precision, recall, f1
def report_to_eval_scores(lines):
    for line in lines.split('\n'):
        mat = SCORE_PAT.search(line)

        if mat:
            return float(mat.group(1)), float(mat.group(2)), float(mat.group(3))
    return -1, -1, -1