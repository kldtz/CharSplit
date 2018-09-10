
import pickle
import re
import string
from typing import List, Tuple
from collections import defaultdict
import pandas as pd
from kirke.ebrules.addrclassifier import KEYWORDS
from kirke.ebrules.addrclassifier import LogRegModel

#from kirke.utils import strutils

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
def find_addresses(text: str, constituencies: List[str]) -> List[Tuple[int, int, str]]:
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

"""Find features for featuresets"""


# Regex and string constants
DIGIT = re.compile(r'\d')
ZIP_CODE_FORMATS = [re.compile(r'\b\d{5}[-\s]+\d{4}\b'),
                    re.compile(r'\b\d{5}\b'),
                    re.compile(r'\b\d{4}\b'),
                    re.compile(r'\b\d{6}\b'),
                    re.compile(r'\b' + UK_STD + r'\b')]
ALNUM_SET = set(string.ascii_letters).union(string.digits)
NON_ALNUM = re.compile(r'[^A-Za-z\d]')

UK_ZIP_PAT = re.compile(r'\b' + UK_STD + r'\b')


def split(line: str, num_chunks: int):
    """Splits a string into the indicated number of chunks."""
    # pylint: disable=invalid-name
    q, m = divmod(len(line), num_chunks)
    # pylint: disable=invalid-name
    r = range(num_chunks)
    return [line[i * q + min(i, m):(i + 1) * q + min(i + 1, m)] for i in r]


def find_digit_features(line: str, num_chunks: int):
    """Returns a dictionary of whether each of num_chunks chunks has a digit."""
    chunks = split(line, num_chunks)
    return {str(i): bool(DIGIT.search(chunks[i])) for i in range(num_chunks)}


def find_zip_code_features(line: str):
    """Returns a dictionary of whether each zip code pattern is in a string."""
    return {str(z): bool(z.search(line)) for z in ZIP_CODE_FORMATS}


def find_uk_zip_code(line: str) -> bool:
    return bool(UK_ZIP_PAT.search(line))


def find_keyword_features(line: str, keywords):
    """Returns a dictionary of whether each category is present in a string."""

    # Strip non-alphanumeric characters then pad with spaces
    non_alnum_chars = str(set(line) - ALNUM_SET)
    line = pad(line.lstrip(non_alnum_chars).rstrip(non_alnum_chars))

    # Also consider where non-alphanumeric chars replaced w/ ' ' (e.g. City  ST)
    line2 = NON_ALNUM.sub(' ', line)
    # Use category (helps e.g. 3 Edison Way) and keyword features (e.g. Suite)
    keyword_features = {}
    for category in keywords:
        category_in_s = False
        for k in keywords[category]:
            if k in line.split() or k in line2.split():
                category_in_s = True
                keyword_features[k] = True
            else:
                keyword_features[k] = False
        keyword_features[category] = category_in_s

    # Return compiled dictionary
    return keyword_features


def find_features(line: str, num_chunks, keywords):
    """Finds all features given a string and relevant options."""
    digit_features = find_digit_features(line, num_chunks)
    zip_code_features = find_zip_code_features(line)
    keyword_features = find_keyword_features(line, keywords)

    # Return the union of the dictionaries
    return dict(i for d in (digit_features, zip_code_features, keyword_features)
                for i in d.items())


"""Load and prepare the classifier"""


with open('kirke/ebrules/addr_classifier.pkl', 'rb') as f:
    ADDR_CLASSIFIER = pickle.load(f)

# it takes around 7 ms per call
def classify(line: str) -> float:
    """Returns probability (range 0-1) s is an addresses (accept if >= 0.5)."""

    # if (len(s) not in range(MIN_ADDRESS_LEN, MAX_ADDRESS_LEN + 1)
    #    or not any(c.isalpha() for c in s)):
    #    return 0
    len_s = len(line)
    if (len_s <= MIN_ADDRESS_LEN or len_s >= MAX_ADDRESS_LEN + 1) or \
       not strutils.has_alpha(line):
        return 0
    # s = unidecode(s)
    probs, label = ADDR_CLASSIFIER.predict(line)

    return label
