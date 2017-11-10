import pandas as pd
import pickle
import re
import string
from unidecode import unidecode


"""Config. RETRAIN retrains the classifier."""


DATA_DIR = './dict/addresses/'
NUM_DIGIT_CHUNKS = 10
MIN_ADDRESS_LEN = 5
MAX_ADDRESS_LEN = 100


"""Aggregate keyword data"""


def pad(s):
    """Add a space before and after a string to ensure whole-word matches."""
    return ' ' + s + ' '


def all_constituencies():
    all_keywords = load_keywords()
    all_const = []
    stop_keywords = ["Incorporated", "INCORPORATED", "&", "Corporation", "CORPORATION", "Inc", "INC", "LLC", "Llc", "llc"]
    for category in all_keywords.keys():
        for const in all_keywords[category]:
            if const.strip() not in stop_keywords:
                all_const.append(const.strip())
    return all_const+['P.O', 'BOX', 'Box', 'Creek', 'CREEK', 'Las', 'LAS', 'Rio', 'RIO', 'New', 'NEW', 'York', 'YORK', 'San', 'SAN', 'Santa', 'SANTA', 'Los', 'Los', 'NE', 'NW', 'SE', 'SW', 'N', 'S', 'W', 'E', 'North', 'NORTH', 'South', 'SOUTH', 'East', 'EAST', 'West', 'WEST']

def load_keywords():
    # Create a dictionary object to return
    keywords = {}

    # Read constituencies (e.g. states, provinces) and their abbreviations
    countries = ['us', 'uk', 'aus', 'can']
    keywords['constituencies'] = []
    for c in countries:
        df = pd.read_csv(DATA_DIR + 'constituencies_' + c + '.csv').dropna()
        keywords[c] = df['name'].tolist() + df['abbr'].tolist()
        keywords['constituencies'] += keywords[c]

    # Read single-column CSVs
    categories = ['address_terms', 'apt_abbrs', 'apt_terms',
                  'business_suffixes', 'country_names', 'numbers', 'road_abbrs']
    for c in categories:
        df = pd.read_csv(DATA_DIR + c + '.csv', header=None).dropna()
        keywords[c] = df[0].tolist()

    # Save title case and uppercase versions, padded, for each keyword
    for category in keywords:
        title_case_keywords = [pad(k.title()) for k in keywords[category]]
        uppercase_keywords = [pad(k.upper()) for k in keywords[category]]
        keywords[category] = set(title_case_keywords + uppercase_keywords)

    return keywords


"""Find features for featuresets"""


# Regex and string constants
DIGIT = re.compile(r'\d')
UK_STD = '[A-Z]{1,2}[0-9R][0-9A-Z]? (?:(?![CIKMOV])[0-9][a-zA-Z]{2})'
ZIP_CODE_FORMATS = [re.compile('\b\d{5}[-\s]+\d{4}\b'), re.compile('\b\d{5}\b'),
                    re.compile('\b\d{4}\b'), re.compile('\b\d{6}\b'),
                    re.compile('\b' + UK_STD + '\b')]
ALNUM_SET = set(string.ascii_letters).union(string.digits)
NON_ALNUM = re.compile(r'[^A-Za-z\d]')


def split(s, num_chunks):
    """Splits a string into the indicated number of chunks."""
    q, m = divmod(len(s), num_chunks)
    r = range(num_chunks)
    return [s[i * q + min(i, m):(i + 1) * q + min(i + 1, m)] for i in r]


def find_digit_features(s, num_chunks):
    """Returns a dictionary of whether each of num_chunks chunks has a digit."""
    chunks = split(s, num_chunks)
    return {str(i): bool(DIGIT.search(chunks[i])) for i in range(num_chunks)}


def find_zip_code_features(s):
    """Returns a dictionary of whether each zip code pattern is in a string."""
    return {str(z): bool(z.search(s)) for z in ZIP_CODE_FORMATS}


def find_keyword_features(s, keywords):
    """Returns a dictionary of whether each category is present in a string."""

    # Strip non-alphanumeric characters then pad with spaces
    non_alnum_chars = str(set(s) - ALNUM_SET)
    s = pad(s.lstrip(non_alnum_chars).rstrip(non_alnum_chars))

    # Also consider where non-alphanumeric chars replaced w/ ' ' (e.g. City  ST)
    s2 = NON_ALNUM.sub(' ', s)

    # Use category (helps e.g. 3 Edison Way) and keyword features (e.g. Suite)
    keyword_features = {}
    for category in keywords:
        category_in_s = False
        for k in keywords[category]:
            if k in s or k in s2:
                category_in_s = True
                keyword_features[k] = True
            else:
                keyword_features[k] = False
        keyword_features[category] = category_in_s

    # Return compiled dictionary
    return keyword_features


def find_features(s, num_chunks, keywords):
    """Finds all features given a string and relevant options."""
    digit_features = find_digit_features(s, num_chunks)
    zip_code_features = find_zip_code_features(s)
    keyword_features = find_keyword_features(s, keywords)

    # Return the union of the dictionaries
    return dict(i for d in (digit_features, zip_code_features, keyword_features)
                for i in d.items())


"""Load and prepare the classifier"""


with open(DATA_DIR + 'address_classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)


def classify(s):
    """Returns probability (range 0-1) s is an addresses (accept if >= 0.5)."""
    if (len(s) not in range(MIN_ADDRESS_LEN, MAX_ADDRESS_LEN + 1)
        or not any(c.isalpha() for c in s)):
        return 0
    s = unidecode(s)
    features = find_features(s, NUM_DIGIT_CHUNKS, load_keywords())
    return classifier.prob_classify(features).prob(1)

