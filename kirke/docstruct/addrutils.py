#!/usr/bin/env python3

import re

from kirke.utils import strutils


STATES = {'AK': 'Alaska',
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
          'WY': 'Wyoming'}


ZIP_PAT = re.compile(r'\b[A-Z][A-Z]\s+\d{5}\b')

def zip_finder(line):
    mat = ZIP_PAT.search(line)
    return mat
    #if mat:
    #    return mat.start(), mat.end(), mat.group()
    # return False


STATE_LIST = strutils.load_str_list('dict/us.states.dict')
CAPITAL_LIST = strutils.load_str_list('dict/us.capitals.dict')
STATE_CAPITAL_LIST = STATE_LIST + CAPITAL_LIST

# intentionally no re.IGNORECASE
LC_STATE_ABBREV = [abbrev[0] + abbrev[1].lower() for abbrev in STATES]
UPPER_LC_STATE_ABBREV = [abbrev for abbrev in STATES]
UPPER_LC_STATE_ABBREV.extend(LC_STATE_ABBREV)
STATE_ABBREV_PAT = re.compile(r'({})\s*$'.format('|'.join(UPPER_LC_STATE_ABBREV)))
STATE_CAPITAL_PAT = re.compile(r'\b({})\b'.format('|'.join([loc_st.replace('.', r'\.')
                                                            for loc_st in STATE_CAPITAL_LIST])),
                               re.IGNORECASE)

# st = 'st. paul'
# mat = STATE_CAPITAL_PAT.find(st)
# print("STATE_CAPITAL_PAT = '{}'".format(STATE_CAPITAL_PAT))
INVALID_ADDRESS_WORDS = strutils.load_str_list('dict/address.words.dict')

# '100" TV', '#333 xxx"
ADDR_WORD_PAT_ST = r'(#\d+|\d\"|\b({})\b)'.format('|'.join([word.replace('.', r'\.')
                                                            for word in INVALID_ADDRESS_WORDS]))
ADDR_WORD_PAT = re.compile(ADDR_WORD_PAT_ST, re.IGNORECASE)
# print("ADDR_WORD_PAT_ST = '{}'".format(ADDR_WORD_PAT_ST))


def has_state_or_capital(line) -> bool:
    mat = STATE_CAPITAL_PAT.search(line)
    return bool(mat)


def has_state_abbrev_or_zipcode(line) -> bool:
    mat = STATE_ABBREV_PAT.search(line)
    if mat:
        return bool(mat)
    mat = ZIP_PAT.search(line)
    if mat:
        return bool(mat)
    return False


def is_address_line(line: str) -> bool:
    return has_state_or_capital(line) or \
        has_state_abbrev_or_zipcode(line) or \
        bool(ADDR_WORD_PAT.search(line))
