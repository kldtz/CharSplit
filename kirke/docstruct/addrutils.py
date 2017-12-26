#!/usr/bin/env python

import argparse
import logging
from pprint import pprint
import sys
import warnings
import re

import nltk

from kirke.utils import strutils

states = {
            'AK': 'Alaska',
            'AL': 'Alabama',
            'AR': 'Arkansas',
            'AS': 'American Samoa',
            'AZ': 'Arizona',
            'CA': 'California',
            'CO': 'Colorado',
            'CT': 'Connecticut',
            'DC': 'District of Columbia',
            'DE': 'Delaware',
            'FL': 'Florida',
            'GA': 'Georgia',
            'GU': 'Guam',
            'HI': 'Hawaii',
            'IA': 'Iowa',
            'ID': 'Idaho',
            'IL': 'Illinois',
            'IN': 'Indiana',
            'KS': 'Kansas',
            'KY': 'Kentucky',
            'LA': 'Louisiana',
            'MA': 'Massachusetts',
            'MD': 'Maryland',
            'ME': 'Maine',
            'MI': 'Michigan',
            'MN': 'Minnesota',
            'MO': 'Missouri',
            'MP': 'Northern Mariana Islands',
            'MS': 'Mississippi',
            'MT': 'Montana',
            'NA': 'National',
            'NC': 'North Carolina',
            'ND': 'North Dakota',
            'NE': 'Nebraska',
            'NH': 'New Hampshire',
            'NJ': 'New Jersey',
            'NM': 'New Mexico',
            'NV': 'Nevada',
            'NY': 'New York',
            'OH': 'Ohio',
            'OK': 'Oklahoma',
            'OR': 'Oregon',
            'PA': 'Pennsylvania',
            'PR': 'Puerto Rico',
            'RI': 'Rhode Island',
            'SC': 'South Carolina',
            'SD': 'South Dakota',
            'TN': 'Tennessee',
            'TX': 'Texas',
            'UT': 'Utah',
            'VA': 'Virginia',
            'VI': 'Virgin Islands',
            'VT': 'Vermont',
            'WA': 'Washington',
            'WI': 'Wisconsin',
            'WV': 'West Virginia',
            'WY': 'Wyoming'
    }

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ZIP_PAT = re.compile(r'\b[A-Z][A-Z]\s+\d{5}\b')

def zip_finder(line):
    mat = ZIP_PAT.search(line)
    return mat
    #if mat:
    #    return mat.start(), mat.end(), mat.group()
    # return False


state_list = strutils.load_str_list('dict/us.states.dict')
capital_list = strutils.load_str_list('dict/us.capitals.dict')
state_capital_list = state_list + capital_list

# intentionally no re.IGNORECASE
lc_state_abbrev = [abbrev[0] + abbrev[1].lower() for abbrev in states.keys()]
upper_lc_state_abbrev = [abbrev for abbrev in states.keys()]
upper_lc_state_abbrev.extend(lc_state_abbrev)
state_abbrev_pat = re.compile('({})\s*$'.format('|'.join(upper_lc_state_abbrev)))
state_capital_pat = re.compile('\b({})\b'.format('|'.join([loc_st.replace('.', r'\.') for loc_st in state_capital_list])), re.IGNORECASE)

# st = 'st. paul'
# mat = state_capital_pat.find(st)
# print("state_capital_pat = '{}'".format(state_capital_pat))
invalid_address_words = strutils.load_str_list('dict/address.words.dict')

# '100" TV', '#333 xxx"
ADDR_WORD_PAT_ST = r'(#\d+|\d\"|\b({})\b)'.format('|'.join([word.replace('.', r'\.') for word in invalid_address_words]))
ADDR_WORD_PAT = re.compile(ADDR_WORD_PAT_ST, re.IGNORECASE)
# print("ADDR_WORD_PAT_ST = '{}'".format(ADDR_WORD_PAT_ST))


def has_state_or_capital(line) -> bool:
    mat = state_capital_pat.search(line)
    return bool(mat)


def has_state_abbrev_or_zipcode(line) -> bool:
    mat = state_abbrev_pat.search(line)
    if mat:
        return bool(mat)
    mat = ZIP_PAT.search(line)
    if mat:
        return bool(mat)
    return False


def is_address_line(line: str) -> bool:
    return (has_state_or_capital(line) or
            has_state_abbrev_or_zipcode(line) or
            bool(ADDR_WORD_PAT.search(line)))

