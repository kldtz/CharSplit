
from itertools import groupby
import re
import string
from typing import List, Optional, Tuple

from kirke.ebrules import titles, addresses
from kirke.utils import strutils


# """Config"""


DATA_DIR = './dict/parties/'
PARTY_CANT_HAVE = [r'\bon the \w+ hand\b']
TERM_CANT_HAVE = [r'\bdate\b', r'\boffice\b', r'\bdefined\b']


# """Get keywords"""

PARENS = re.compile(r'\([^\)]*?(?:“|")[^\)]*?(?:"|”)[^\)]*?\)')
NON_COMMA_SEPARATORS = re.compile(r',\sand|\sand|;')
PAREN_SYMBOL = re.compile(r'(⭐️)')
# this was replace later in the code.  Keeping it for now.  Probably due to UK.
# ZIP_CODE_YEAR = re.compile(r'\b\d{5}(?:\-\d{4})?\b|\b(?:19|20)\d{2}\b')
QUOTE = re.compile(r'[“"”]')

# Supports US (5 or 5-4), UK, Australia (4), Switzerland (4), Shanghai (6)
UK_STD = '[A-Z]{1,2}[0-9ROI][0-9A-Z]? +(?:(?![CIKMOV])[0-9O][a-zA-Z]{2})'
ZIP_CODE_YEAR = re.compile(r'\d{{4}}|\b{}\b'.format(UK_STD))
DOT_SPACE = re.compile(r'[\.\s]+')


def cap_rm_dot_space(astr: str) -> str:
    """Allows comparison between different cases (e.g. L.L.C. and Llc)."""
    return DOT_SPACE.sub('', astr.strip().upper())


with open(DATA_DIR + 'states.list') as f:
    STATES = [cap_rm_dot_space(state) for state in f.read().splitlines()]
with open(DATA_DIR + 'business_suffixes.list') as f:
    BUSINESS_SUFFIXES_X1 = f.read().splitlines()
with open(DATA_DIR + 'name_suffixes.list') as f:
    NAME_SUFFIXES = f.read().splitlines()
SUFFIXES_X1 = BUSINESS_SUFFIXES_X1 + NAME_SUFFIXES
MAX_SUFFIX_WORDS = max(len(s.split()) for s in SUFFIXES_X1)
BUSINESS_SUFFIXES = [cap_rm_dot_space(s) for s in BUSINESS_SUFFIXES_X1]
SUFFIXES = BUSINESS_SUFFIXES + [cap_rm_dot_space(s) for s in NAME_SUFFIXES]

# load invalid party phrases
INVALID_PARTIES_ST_LIST = strutils.load_str_list('dict/parties/invalid.parties.txt')
INVALID_PARTIES_SET = set(INVALID_PARTIES_ST_LIST)

VALID_1WORD_PARTY_SET = ['supplier', 'customer']
ADDRESS_PARTS = ['floor', 'road', 'court', 'street', 'drive']
ZIP_PAT = re.compile(r'[loO\d]{5,6}')


def is_valid_1word_party(line) -> bool:
    return line.lower() in VALID_1WORD_PARTY_SET or line.isupper()


# pylint: disable=too-many-return-statements
def is_invalid_party(line, is_party=True) -> bool:
    #checks for some punctuation
    if ':' in line or '/' in line:
        return True

    #too short to be a party
    if len(line) <= 2:
        return True

    lc_line = line.lower()
    #catches things that are likely part of an address
    if is_party:
        for part in ADDRESS_PARTS:
            if part in lc_line:
                return True
        if addresses.classify(line) > 0.5:
            return True

    #catches other likely fps
    if ' this ' in lc_line:
        return True
    if 'page ' in lc_line:
        return True

    #catches zip code pulls
    if ZIP_CODE_YEAR.search(line):
        return True

    #if not invalid, return
    return lc_line in INVALID_PARTIES_SET

# """Process parts of strings (already split by comma, and, & semicolon)"""


# Companies list from wikipedia.org/wiki/List_of_companies_of_the_United_States
with open(DATA_DIR + 'companies.list') as f:
    VALID_LOWER = {w for w in f.read().split() if w.isalpha() and w.islower()}


def invalid_lower(word):
    return word.isalpha() and word.islower() and word not in VALID_LOWER


def keep(astr: str) -> bool:
    """Eliminate titles (the "Agreement") as potential parties and terms."""
    alphanum_chars = ''.join([c for c in astr if c.isalnum()])
    return titles.title_ratio(alphanum_chars) < 73 if alphanum_chars else False


PARTY_REGEXES = [re.compile(party, re.IGNORECASE) for party in PARTY_CANT_HAVE]


def process_part(apart: str) -> str:
    # Reject if just a state or state abbreviations (only 2 characters)
    if len(apart) < 3 or cap_rm_dot_space(apart) in STATES:
        return ''

    # Terminate at a single-word business suffix or business suffix abbreviation
    for word in apart.split():
        if cap_rm_dot_space(word) in BUSINESS_SUFFIXES or invalid_lower(word):
            apart = apart[:apart.index(word) + len(word)]
            break

    # Remove certain phrases like 'on the one hand'
    for regex in PARTY_REGEXES:
        apart = regex.sub('', apart)

    # Take away any lowercase words from end e.g. 'organized'
    while apart.strip() and apart.split()[-1].islower():
        apart = apart[:len(apart) - len(apart.split()[-1]) - 1]

    # Return the processed part if not a title
    return apart if keep(apart) else ''


# """Process terms"""


TERM_REGEXES = [re.compile(t, re.IGNORECASE) for t in TERM_CANT_HAVE]


def process_term(term: str) -> str:
    """Ignore if term is a title or contains words that a term cannot have."""
    return term if keep(term) and not any(r.search(term) for r in TERM_REGEXES) else ''


# """Helper functions and regexes for extracting parties from party line"""


PARTY_CHARS = set(string.ascii_letters).union(string.digits).union('⭐️.')


def party_strip(astr: str) -> str:
    non_party_chars = set(astr) - PARTY_CHARS
    return astr.strip(str(non_party_chars)) if non_party_chars else astr


SENTENCE_DOES_NOT_CONTINUE = r'(?=\s+(?:[A-Z0-9].|[a-z][A-Z0-9]))'
NOT_FEW_LETTERS = r'(?<!\b[A-Za-z]\.)(?<!\b[A-Za-z]{2}\.)'
NOT_NUMBER = r'(?<!\bN(O|o)\.)(?<!\bN(O|o)(S|s)\.)'
REAL_PERIOD = r'\.' + SENTENCE_DOES_NOT_CONTINUE + NOT_FEW_LETTERS + NOT_NUMBER
FIRST_SENT = re.compile(r'(.*?' + REAL_PERIOD + ')')


def first_sentence(astr: str) -> str:
    """Trying to avoid sentence tokenizing since occurs before CoreNLP."""
    match = FIRST_SENT.search(astr)
    return match.group() if match else astr


# """Extract parties from party line"""


def zipcode_replace(apart: str, new_parts: List) -> List:
    # If zip code or year in part and not already deleting ('❌'), mark '🏡'
    if ZIP_CODE_YEAR.search(apart):
        new_parts.append('🏡')
    return new_parts

def zipcode_remove(grps):
    # Going backwards, when see a zip code/ year, remove up to prev removed line

    # This loop modified grps in-place
    # pylint: disable=consider-using-enumerate
    for i in range(len(grps)):
        zip_code_inds = [j for j, part in enumerate(grps[i]) if part == '🏡']
        if zip_code_inds:
            new_start = max(zip_code_inds) + 1
            terms_before = [part for part in grps[i][:new_start] if part == '⭐️']
            new_parts = grps[i][new_start:]
            grps[i] = terms_before + new_parts
    return grps

# returns: [['TUP, LLC', '(hereinafter called "Landlord”)'], ['CERTIFIED DIABETIC SERWCES, INC.',
#                                                             '(hereinafter called "Tenant”)']]
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def extract_between_among(astr: str, is_party: bool = True) -> List[List[str]]:
    """Return parties for party lines containing either 'between' o[List[str]]r 'among'."""
    astr = astr.split('between')[-1].split('among')[-1]

    # Temporarily sub defined terms with '=' to avoid splitting on their commas
    terms = PARENS.findall(astr)

    #sub common delimiters with commas to split on
    astr = re.sub(r'(between)|(being)|(\n)|\(?[\div]+\)', ', ', astr)
    astr = NON_COMMA_SEPARATORS.sub(',', PARENS.sub('⭐️', astr))

    # Split the string into parts, applying party_strip between each step
    parts = [party_strip(part) for part in party_strip(astr).split(', ')]
    parts = [party_strip(q) for p in parts for q in PAREN_SYMBOL.split(p) if q]
    parts = [q for q in parts if q]

    # Process parts and decide which parts to keep
    new_parts = ['']
    for part in parts:
        # If p is a term, keep the term and continue
        if part == '⭐️':
            new_parts.append(part)
            continue

        # If first word is a suffix (MD MBA), add to previous and remove from p
        seen_suffixes = ''
        check_again = True
        while check_again:
            check_again = False
            words = part.strip().split()
            for i in range(MAX_SUFFIX_WORDS):
                words_considered = cap_rm_dot_space(''.join(words[:i]))
                if any(words_considered == s for s in SUFFIXES):
                    seen_suffixes += ' ' + ' '.join(words[:i])
                    part = ' '.join(words[i:])
                    check_again = True
                    break
        if seen_suffixes:
            # Append suffixes
            new_parts[-1] += ',' + seen_suffixes

        # Take out 'the ' from beginning of string; bail if remaining is empty
        while part.startswith('the '):
            part = part[4:]
        if not part.strip():
            continue

        # Mark for deletion if first word has no uppercase letters or digits
        first_word = part.split()[0]
        if not any(c.isupper() or c.isdigit() for c in first_word):
            new_parts.append('❌')
            continue

        if is_party:
            new_parts = zipcode_replace(part, new_parts)

        # Process then keep the part if not a title, etc.
        processed_part = process_part(part)
        if processed_part:
            new_parts.append(processed_part)

    # Remove lines marked for deletion (❌)
    parts = new_parts if new_parts[0] else new_parts[1:]

    # grps is List[List[str]]
    # example: [['TUP, LLC'], ['⭐️', 'CERTIFIED DIABETIC SERWCES, INC.'], ['⭐️']]
    grps = [list(g) for k, g in groupby(parts, lambda p: '❌' in p) if not k]
    #if is_party:
        #grps = zipcode_remove(grps)
    parts = [part
             for g in grps
             for part in g]
    # Add terms back in
    terms = [process_term(t) for t in terms]

    current_term = 0
    parties = []  # type: List[List[str]]
    part_types = []  # type: List[int]
    part_type_bools = []  # type: List[bool]
    for part in parts:
        # Record term type {0: party, 1: (), 2: ("")} and substitute term
        # part_type_bool, is_separate=0, is_party=1, is_term=2
        part_type_bool = part == '⭐️'
        part_type = int(part_type_bool)
        if part_type_bool:
            part = terms[current_term]
            current_term += 1
            if QUOTE.search(part):
                part_type = 2
        if not part:
            # Occurs e.g. if term was a title or a date
            continue

        # Append if first party/term or term follows term (if 2, no other 2)
        if part_types:
            if part_type_bool not in part_type_bools \
               or part_types[-1] \
               and (part_type == 1 or (part_type == 2 and 2 not in part_types)):
                parties[-1].append(part)
                part_types.append(part_type)
                part_type_bools.append(part_type_bool)
                continue
        # Otherwise start a new party
        parties.append([part])
        part_types = [part_type]
        part_type_bools = [part_type_bool]

    # Remove parties that only contain defined terms, then return
    parties = [xpart
               for xpart in parties
               if not all(PARENS.search(word)
                          for word in xpart)]
    return parties


def extract_parties_from_party_line(astr: str, is_party: bool = True) -> List[List[str]]:
    """
    Return list of parties (which are lists of strings) of s (party line).
    is_party flag should be true for party provision but false for landlord / tenant provisions
    when is_party is false it will keep address-like extractions
    """
    astr = first_sentence(astr)

    #bullet type parties won't contain between / among, extract anyway
    if re.match(r'\(?[\div]\)', astr):
        return extract_between_among(astr, is_party)

    # Try possible rules
    if ('between' in astr or 'among' in astr) or not is_party:
        return extract_between_among(astr, is_party)

    return []


# """Extract parties from debug file"""
def parties_to_offsets(parties: List[List[str]],
                       party_line: str) -> List[List[Tuple[int, int]]]:
    """Converts string parts of parties to offsets (relative to party line)."""
    result = []  # type: List[List[Tuple[int, int]]]
    if parties:
        # this loop modify parties in-place
        # pylint: disable=consider-using-enumerate
        for i in range(len(parties)):
            offsets = []  # type: List[Tuple[int, int]]
            for part in parties[i]:
                # because the original string can be regex meta characters, such as
                # [] (), we must escape first.  space => '\\ '.  We want to handle
                # multiple spaces here.
                part_pat = re.compile(re.sub(r'(\\ )+', r'\s+', re.escape(part)))
                part_mat = part_pat.search(party_line)
                if part_mat:
                    offsets.append((part_mat.start(0), part_mat.end(0)))
            if offsets:
                result.append(offsets)
    return result


def extract_parties(filepath: str) -> List[List[Tuple[int, int]]]:
    """Return list of parties (lists of (start, inclusive-end) offsets)."""

    # Find the party line in the file
    party_line = None
    with open(filepath) as fin:
        for line in fin:
            tags = line.split('\t')[0].split('|')
            if 'party_line' in tags:
                after_first_bracket = ''.join(line.split('[')[1:])
                between_brackets = ''.join(after_first_bracket.split(']')[:-1])
                party_line = between_brackets
                break

    # Return None if no party_line was found
    if not party_line:
        return []

    # Extract parties and return their offsets
    parties = extract_parties_from_party_line(party_line, is_party=True)
    return parties_to_offsets(parties, party_line)


def extract_party_line(paras_attr_list):
    # lines = []
    offset = 0
    # start_end_list = []
    for i, (line_st, para_attrs) in enumerate(paras_attr_list):
        # attrs_st = '|'.join([str(attr) for attr in para_attrs])
        # print('\t'.join([attrs_st, '[{}]'.format(line_st)]), file=fout1)
        line_st_len = len(line_st)
        # whitespace_line = '\n'
        # checks if bullet type party, joins all bullets into a line
        if 'party_line' in para_attrs and 'toc' not in para_attrs:
            if re.match(r'\(?[\div]+\)', line_st, re.I):
                next_idx = i+1
                next_line, unused_next_attrs = paras_attr_list[next_idx]
                offset_add = line_st_len
                return_st = line_st
                while not next_line or re.match(r'\(?[\div]+\)', next_line, re.I) or \
                      re.match(r'^[\-\s]*and', next_line, re.I):
                    offset_add += len(next_line)
                    return_st += "\n" + next_line
                    next_idx += 1
                    next_line, unused_next_attrs = paras_attr_list[next_idx]
                return offset, offset+ offset_add, return_st

            return offset, offset + line_st_len, line_st
        offset += line_st_len + 1

        # don't bother if party_line is too far from start of the doc
        if i > 2000:
            return None

    return None


# paras_text is not used for title right now
# The first Tuple[int, int] is the party offset
# the Optional[Tuple[int, int]] is the defined term offsets
def extract_offsets(paras_attr_list, unused_para_text: str) \
    -> List[Tuple[Tuple[int, int],
                  Optional[Tuple[int, int]]]]:
    """Return list of parties (lists of (start, inclusive-end) offsets)."""

    out_list = []  # type: List[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]]

    # Grab lines from the file
    start_end_partyline = extract_party_line(paras_attr_list)
    if start_end_partyline:
        start, unused_end, party_line = start_end_partyline

        # Extract parties and return their offsets
        parties = extract_parties_from_party_line(party_line)
        offset_pair_list = parties_to_offsets(parties, party_line)
        # logging.info("offset_pair_list: {}".format(offset_pair_list))
        for party_term_ox_list in offset_pair_list:
            if len(party_term_ox_list) == 2:
                party_start, party_end = party_term_ox_list[0]
                defined_term_start, defined_term_end = party_term_ox_list[1]
                out_list.append(((start + party_start, start + party_end),
                                 (start + defined_term_start, start + defined_term_end)))
            else:
                party_start, party_end = party_term_ox_list[0]
                out_list.append(((start + party_start, start + party_end),
                                 None))

    # """
    # non_partyline_parties = party_islands.extract_party_islands_offset(paras_attr_list)
    # for start, end in non_partyline_parties:
    #    out_list.append(((start, end), None))
    # """
    #logging.info("out_list: {}".format(out_list))
    return out_list


# pylint: disable=too-few-public-methods
class PartyAnnotator:

    # pylint: disable=unused-argument
    def __init__(self, provision: str) -> None:
        self.provision = 'party'

    # pylint: disable=no-self-use
    def extract_provision_offsets(self, paras_with_attrs, paras_text) \
        -> List[Tuple[Tuple[int, int],
                      Optional[Tuple[int, int]]]]:
        return extract_offsets(paras_with_attrs, paras_text)
