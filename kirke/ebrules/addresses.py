
import pickle
import re
import string
from typing import List, Tuple
from collections import defaultdict
import pandas as pd
from kirke.ebrules.addrclassifier import KEYWORDS
from kirke.ebrules.addrclassifier import LogRegModel

from kirke.utils import strutils

DATA_DIR = './dict/addresses/'
NUM_DIGIT_CHUNKS = 10
MIN_ADDRESS_LEN = 5
MAX_ADDRESS_LEN = 100

US_ZIP = '\d{5}(-\d{4})?'
UK_STD = '[A-Z]{1,2}[0-9ROI][0-9A-Z]? *(?:(?![CIKMOV])[A-Z]?[0-9O][a-zA-Z]{2})'
CAN_STD = '[A-Z][0-9][A-Z] +[0-9][A-Z][0-9]'
ZIP_CODE_YEAR = re.compile(r'\b\d{4,5}\b' + r'|\b{}\b'.format(UK_STD))

LOCS = KEYWORDS['uk'] + KEYWORDS['us'] + KEYWORDS['can'] + KEYWORDS['country_names']
LOCS = sorted(LOCS, reverse = True)

# pylint: disable=too-many-locals
def find_addresses(text: str) -> List[Tuple[int, int, str]]:
    matches = re.finditer(r'(?=(\b(\d+|P\.? ?[0O]\.?|One) +.+?(\b{}\b|\b{}\b|\b{}\b)|\b(\d+|P\.? ?[0O]\.?|One) +.+?((\b({})\b)[,\. ]+)+))'.format(US_ZIP, UK_STD, CAN_STD, "|".join(LOCS)), text, re.DOTALL)
    all_spans = [match.span(1) for match in matches]
    addr_se_st_list = []  # type: List[Tuple[int, int, str]]
    prev_start, prev_end, prev_prob = 0, 0, 0
    for ad_start, ad_end in all_spans:
        addr_st = text[ad_start:ad_end]
        if len(addr_st.split()) > 3 and len(addr_st.split()) < 25:  # an address must have at least 4 words
            address_prob = classify(addr_st)
            print(">>>>>>>", addr_st.replace("\n", " "), "<<", address_prob)
            if address_prob >= 0.5:
                if ad_start > prev_start and ad_start < prev_end:
                    if address_prob >= prev_prob:
                        addr_se_st_list.pop()
                        addr_se_st_list.append((ad_start, ad_end, addr_st))
                        prev_start, prev_end, prev_prob = ad_start, ad_end, address_prob
                else:
                    addr_se_st_list.append((ad_start, ad_end, addr_st))
                    prev_start, prev_end, prev_prob = ad_start, ad_end, address_prob
    return addr_se_st_list

"""Load and prepare the classifier"""
with open(DATA_DIR+'addr_classifier.pkl', 'rb') as f:
    ADDR_CLASSIFIER = pickle.load(f)

def classify(line: str) -> float:
    """Returns probability that line is an addr"""
    len_s = len(line)
    if (len_s <= MIN_ADDRESS_LEN or len_s >= MAX_ADDRESS_LEN + 1) or \
       not strutils.has_alpha(line):
        return 0
    probs, label = ADDR_CLASSIFIER.predict(line)

    return probs[1]
