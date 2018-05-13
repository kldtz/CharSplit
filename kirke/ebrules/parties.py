# pylint: disable=too-many-lines
import itertools
import logging
import re
import string
from typing import List, Optional, Tuple

# from nltk.tokenize import sent_tokenize

from kirke.docstruct import partyutils
from kirke.ebrules import titlesold, addresses
from kirke.utils import nlputils, strutils


IS_DEBUG_DISPLAY_TEXT = False
IS_DEBUG_MODE = False


# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# IS_DEBUG_DISPLAY_TEXT = True
# IS_DEBUG_MODE = True


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
# bravo to Jason, figured out to use {{}}.
# ZIP_CODE_YEAR = re.compile(r'\b\d{{4,5}}\b|\b{}\b'.format(UK_STD))
ZIP_CODE_YEAR = re.compile(r'\b\d{4,5}\b' + r'|\b{}\b'.format(UK_STD))
# TODO, jshaw, this is introducing zip code as party names.  Remove for now.
# 'DELL RECEIVABLES FINANCING 2016 D.A.C", year is valid
# ZIP_CODE_YEAR = re.compile(r'\b{}\b'.format(UK_STD))
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

VALID_1WORD_PARTY_ST_LIST = ['supplier', 'customer', 'purchaser', 'guarantor']
VALID_1WORD_PARTY_SET = set(VALID_1WORD_PARTY_ST_LIST)
VALID_1WORD_PARTY_SET.update([v1word + 's' for v1word in VALID_1WORD_PARTY_ST_LIST])


ADDRESS_PARTS = ['floor', 'road', 'court', 'street', 'drive']
ADDRESS_PARTS_PAT = re.compile(r'\b(' + '|'.join(ADDRESS_PARTS) + r')\b', re.I)
ZIP_PAT = re.compile(r'[loO\d]{5,6}')


def is_valid_1word_party(line) -> bool:
    # experimentally, line.isupper() is 0.7925 over 0.7918
    # but, theoretically, 'Modularis ("MOD")' or company name with just 1 word
    return line.lower() in VALID_1WORD_PARTY_SET or \
           line.isupper() or \
           (len(line) > 1 and line[1].isupper())      # coopany name like, 'eBrevia'
    # return line.lower() in VALID_1WORD_PARTY_SET or line.isupper()


# """Process parts of strings (already split by comma, and, & semicolon)"""


# Companies list from wikipedia.org/wiki/List_of_companies_of_the_United_States
with open(DATA_DIR + 'companies.list') as f:
    VALID_LOWER = {w for w in f.read().split() if w.isalpha() and w.islower()}


def invalid_lower(word):
    return word.isalpha() and word.islower() and word not in VALID_LOWER


def keep(astr: str) -> bool:
    """Eliminate titles (the "Agreement") as potential parties and terms."""
    alphanum_chars = ''.join([c for c in astr if c.isalnum()])
    return titlesold.title_ratio(alphanum_chars) < 73 if alphanum_chars else False


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

def load_us_state_end_pat():
    state_list = strutils.load_str_list(DATA_DIR + 'us_state.list')
    # the abbreviation is intentionally missing 'co', which is often a org suffix
    state_incomplete_list = strutils.load_str_list(DATA_DIR + 'us_state_abbr.incomplete.list')
    state_list.extend(state_incomplete_list)
    state_regex = r'\b(' + '|'.join(state_list) + r')\.?\s*$'
    return state_regex

US_STATE_END_PAT = load_us_state_end_pat()

def is_ending_in_us_state(line: str) -> bool:
    return bool(re.search(US_STATE_END_PAT, line, re.I))


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

# returns: [['TUP, LLC', '(hereinafter called "Landlord”)'],
#           ['CERTIFIED DIABETIC SERWCES, INC.', '(hereinafter called "Tenant”)']]
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def extract_between_among(astr: str, is_party: bool = True) \
    -> List[List[str]]:
    """Return parties for party lines containing either 'between' o[List[str]]r 'among'."""
    astr = astr.split('between')[-1].split('among')[-1]

    # Temporarily sub defined terms with '=' to avoid splitting on their commas
    terms = PARENS.findall(astr)

    #sub common delimiters with commas to split on
    astr = re.sub(r'(between)|(being)|(\n)|\(?\s*[\divx]+\s*\)', ', ', astr)
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
    grps = [list(g) for k, g in itertools.groupby(parts, lambda p: '❌' in p) if not k]
    #if is_party:
        #grps = zipcode_remove(grps)
    parts = [part
             for g in grps
             for part in g]
    # Add terms back in
    terms = [process_term(t) for t in terms]
    # remove empty terms
    terms = list(filter(bool, terms))

    current_term = 0
    parties = []  # type: List[List[str]]
    part_types = []  # type: List[int]
    part_type_bools = []  # type: List[bool]
    for part in parts:
        # Record term type {0: party, 1: (), 2: ("")} and substitute term
        # part_type_bool, is_separate=0, is_party=1, is_term=2
        part_type_bool = part == '⭐️'
        part_type = int(part_type_bool)
        if part_type_bool and current_term < len(terms):
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

BTWN_AMONG_PAT = re.compile(r' (between|among) ', re.I)
A_COMPANY_PAT = re.compile(r',? ((an?\s+(company|individual|business))|whose) ', re.I)
AND_PAT = re.compile(r' and,? ', re.I)
PARENS_PAT = re.compile(r'\(([^\(]+)\)')

# for Spanish org names
ORG_PREFIX_PAT = re.compile(r'\b(asociacion|hospital)\.?', re.I)




def is_all_less_3char_words(term: str) -> bool:
    words = list(re.findall(r'\w+', term))
    for word in words:
        # print("word: [{}]".format(word))
        if len(word) > 2:
            return False
    return True


# returns (start, end, (entity_st, entity_type))
def get_title_phrase_list(line: str,
                          is_all_upper: bool = False,
                          is_single_word_allowed: bool = False) -> List[Tuple[int, int, str]]:
    """Return a list of all title phrases.

    We will skip all phrase inside parenthesis.  Also entity with just one
    title word.  When is_all_upper, if there are more org prefixes than title phrases,
    reject.
    """
    word_offsets_list = strutils.split_with_offsets_xparens(line)

    # if IS_DEBUG_MODE:
    #     print_debug_list(word_offsets_list, 'zz', title='get_title_words')
    paren_level = 0
    current_term = []  # type: List[Tuple[int, int, str]]
    title_phrase_list = []  # type: List[List[Tuple[int, int, str]]]
    # need to peek for '(', so word_idx
    for unused_word_idx, word_offsets in enumerate(word_offsets_list):
        unused_start, unused_end, word_st = word_offsets
        if word_st == '(':
            # a capitalized word before paren, and hasn't been added
            # 'Modularis ("MOD")'
            if current_term:
                title_phrase_list.append(current_term)
                current_term = []
            paren_level += 1
        elif word_st == ')':
            paren_level -= 1

        # we don't do anything if inside a parenthesis
        if paren_level == 0:
            # TODO, jshaw, the use of 'of', 'de' need more research
            # print("word_offsets: {}".format(word_offsets))
            # TODO, it seems that the first 2 if's can be collapsed into one
            if (is_all_upper and word_st.isupper()) or \
               word_st.lower() in set(['&', 'dba', 'de']):
                # print("is_upper word, add to term")
                current_term.append(word_offsets)
            elif (not is_all_upper and word_st[0].isupper()) or \
                 word_st.lower() in set(['&', 'dba', 'de']):
                # print("is_title word, add to term")
                current_term.append(word_offsets)
            else:  # not a title word
                # is_next_word_left_paren = False
                # if word_idx + 1 < len(word_offsets_list):
                #    _, _, next_word_st = word_offsets_list[word_idx + 1]
                #    print('next_word_st = [{}]'.format(next_word_st))
                #    if next_word_st[0] == '(':
                #        is_next_word_left_paren = True

                # print("not upper or title word")
                if len(current_term) > 1:
                    # print("word complete: {}".format(current_term))
                    title_phrase_list.append(current_term)
                elif is_single_word_allowed and current_term:
                    # print("word complete: {}".format(current_term))
                    title_phrase_list.append(current_term)
                current_term = []
                # nothing else
    # just in case a title word is the last word in the line
    if len(current_term) > 1:
        # print("word complete: {}".format(current_term))
        title_phrase_list.append(current_term)
    elif is_single_word_allowed and current_term:
        # print("word complete: {}".format(current_term))
        title_phrase_list.append(current_term)

    # When all upper, we might find one or two examples of company
    # names, but reject other valid ones.  reject in this case.
    # TODO, jshaw, will check
    # There might be other cases, with all personal names in title_phrases
    # where this check is not useful.  Might need to explore further about
    # checking for all_upper_case
    num_org_suffix = len(nlputils.get_org_suffix_mat_list(line))
    if is_all_upper and len(title_phrase_list) < num_org_suffix:
        return []

    # When is_all_upper, remove all entries with < 3 characters.
    # They are likely to be states, such as 'VA.'
    if is_all_upper:
        tmp_title_phrase_list = []  # type: List[List[Tuple[int, int, str]]]
        for title_phrase in title_phrase_list:
            fstart, unused_fend, unused_first_word = title_phrase[0]
            unused_lstart, lend, unused_last_word = title_phrase[-1]
            tmp_term = line[fstart:lend]
            if not is_all_less_3char_words(tmp_term):
                tmp_title_phrase_list.append(title_phrase)
        title_phrase_list = tmp_title_phrase_list

    out_list = []  # type: List[Tuple[int, int, str]]
    for title_phrase in title_phrase_list:
        fstart, unused_fend, unused_first_word = title_phrase[0]
        unused_lstart, lend, unused_last_word = title_phrase[-1]
        out_list.append((fstart, lend, line[fstart:lend]))
    return out_list


def get_itemized_entity_span_list(line: str) -> List[Tuple[Tuple[int, int, str],
                                                           Tuple[int, int, str]]]:
    print("get_itemized_entity_span_list({})".format(line))
    parens1_mat_list = strutils.find_one_char_paren_mats(line)

    len_line = len(line)
    span_list = []
    prev_span_start = parens1_mat_list[0].end()
    for parens1_mat in parens1_mat_list[1:]:
        mstart, mend, unused_mat_st = parens1_mat.start(), parens1_mat.end(), parens1_mat.group()
        start = prev_span_start
        span_list.append((start, mstart, line[start:mstart]))
        prev_span_start = mend
    if prev_span_start != len_line:
        span_list.append((prev_span_start, len_line, line[prev_span_start:len_line]))

    # for i, span in enumerate(span_list):
    #    print("  spantext[{}] = [{}]".format(i, span))

    result = []  # type: List[Tuple[Tuple[int, int, str], Tuple[int, int, str]]]
    for start, end, span_st in span_list:
        if len(span_st) > 80:
            title_phrase_list = get_title_phrase_list(span_st)
        else:
            title_phrase_list = get_title_phrase_list(span_st, is_single_word_allowed=True)
        if title_phrase_list:  # if there is title_phrase
            # we only take first one, the rest is span.  Ok this doesn't work, 39014.txt,

            if len(title_phrase_list) == 1:
                # 39811.txt
                tstart, tend, tst = title_phrase_list[0]
                not_space_mat = re.search(r'\S', span_st[tend:])
                if not_space_mat:  # found text after title phrase
                    x2span_start = start + tend + not_space_mat.start()
                    result.append(((start + tstart, start + tend, tst),
                                   (x2span_start,
                                    end,
                                    span_st[tend+not_space_mat.start():])))
                else:
                    result.append(((start + tstart, start + tend, tst),
                                   (start + tend, start + tend, '')))
            elif len(title_phrase_list) > 1:
                # 39014.txt

                len_span_st = len(span_st)
                tt_entity_list = []
                prev_tphrase_start = title_phrase_list[0][1]  # end
                for tphrase in title_phrase_list[1:]:
                    tstart, tend, tst = tphrase
                    out_start = prev_tphrase_start
                    tt_entity_list.append((out_start, tstart, span_st[out_start:tstart]))
                    prev_tphrase_start = tend
                if prev_tphrase_start != len_span_st:
                    tt_entity_list.append((prev_tphrase_start,
                                           len_span_st,
                                           span_st[prev_tphrase_start:len_span_st]))

                #print('len(title_phrase_list) = %d, len(tt_entity_list) = %d' %
                #      (len(title_phrase_list), len(tt_entity_list)))
                #for i, (tt, span) in enumerate(zip(title_phrase_list, tt_entity_list)):
                #    print("  tt_text[{}] = tt={}, span={}".format(i, tt, span))

                # pylint: disable=invalid-name
                for i, (tt, span) in enumerate(zip(title_phrase_list, tt_entity_list)):
                    tstart, tend, tst = tt
                    sstart, send, sst = span
                    not_space_mat = re.search(r'\S', sst)
                    if not_space_mat:  # found text after title phrase
                        result.append(((start + tstart, start + tend, tst),
                                       (start + sstart + not_space_mat.start(),
                                        start + send,
                                        sst[not_space_mat.start():])))
                    else:
                        result.append(((start + tstart, start + tend, tst),
                                       (start + send, start + send, '')))

    if IS_DEBUG_MODE:
        for i, pp_span in enumerate(result):
            print("  span_entity[{}] = [{}]".format(i, pp_span))
    return result


def is_all_title(line: str) -> bool:
    if not line:
        return False
    words = line.split()
    for word in words:
        if not word[0].isupper():
            return False
    return True


def is_all_title_or_the(line: str) -> bool:
    words = line.split()
    if not words:
        return False
    for word in words:
        if not (word.isupper() or word.istitle() or word.lower() == 'the'):
            return False
    return True

def is_last_title_or_quote(line: str) -> bool:
    words = line.split()
    if not words:
        return False
    word = words[-1]
    # first character is capitalized or has quote
    return word[0].isupper() or strutils.has_quote(word)



def split_by_phrase_offsets(line: str, entity_list: List[Tuple[int, int, str]]) \
    -> List[Tuple[Tuple[int, int, str],
                  Tuple[int, int, str]]]:
    if not entity_list:
        return []
    out_list = []
    len_line = len(line)
    span_list = []
    if entity_list and entity_list[0][0] > 0:
        end = entity_list[0][1]
        span_list.append((0, end, line[0:end]))

    prev_eend = entity_list[0][1]
    for entity in entity_list[1:]:
        estart, eend, unused_entity_st = entity
        span_list.append((prev_eend, estart, line[prev_eend:estart]))
        prev_eend = eend
    # add the last span
    if prev_eend != len_line:
        last_span = line[prev_eend:len_line]
        if last_span.strip():  # make sure it is not empty
            span_list.append((prev_eend, len_line, line[prev_eend:len_line]))
        else:  # add empty span
            span_list.append((len_line, len_line, ''))
    else:
        span_list.append((len_line, len_line, ''))

    if len(span_list) > len(entity_list):
        out_list.append(((0, 0, ''), span_list[0]))
        for entity, span in zip(entity_list, span_list[1:]):
            out_list.append((entity, span))
    else:
        for entity, span in zip(entity_list, span_list):
            out_list.append((entity, span))

    return out_list


def span_to_party(span_offsets: Tuple[int, int, str]) \
    -> Tuple[int, int, str]:
    """Fix potential issue with party names.

    For example, "FOR VALUE RECEIVED, Blue Calypso, Inc.,"
    35836.txt
    """

    start, end, span_st = span_offsets

    word_offsets_list = strutils.split_with_offsets(span_st)

    num_upper, num_title = 0, 0
    is_upper_only_before_title = False
    saw_title = False
    title_start_offset = -1
    for word_offsets in word_offsets_list:
        start, end, word_st = word_offsets

        if word_st.isupper():
            # print("word_st [{}] is upper".format(word_st))
            num_upper += 1
            # jshaw, I don't fully understand this suggestion from pylint
            # pylint: disable=simplifiable-if-statement
            if not saw_title:
                is_upper_only_before_title = True
            else:  # saw title
                is_upper_only_before_title = False
        elif word_st.istitle():
            # print("word_st [{}] is title".format(word_st))
            num_title += 1
            if not saw_title:
                title_start_offset = start
            saw_title = True

    # print('num_upper= %d, num_title= %d, is_upper_only_before_title = %s' %
    #      (num_upper, num_title, is_upper_only_before_title))

    # everything is normal
    if num_upper == 0 or num_title == 0:
        return span_offsets

    # special case: 'FOR VALUE RECEIVED, Blue Calypso, Inc.,', 35836.txt
    # But, then you have 'SAVVIs, Inc.' in 38703.txt
    if is_upper_only_before_title and \
       span_st[:title_start_offset].strip().endswith(',') and \
       span_st[:title_start_offset].strip().startswith('FOR '):
        # something strange, skip to title start
        diff = title_start_offset - start
        return title_start_offset, end, span_st[start + diff:end]

    return span_offsets


# pylint: disable=too-many-return-statements
def span_to_dterm(span_offsets: Tuple[int, int, str]) \
    -> Optional[Tuple[int, int, str]]:
    start, unused_end, span_st = span_offsets

    parens_mat_list = list(PARENS_PAT.finditer(span_st))

    maybe_dterm_list = []
    # found a defined term
    if parens_mat_list:
        for mat in parens_mat_list:
            if re.search(r'\b(agreement|registered|as\s+defined\s+below)\b', mat.group(), re.I):
                pass
            else:
                maybe_dterm_list.append(mat)
    else:  # no parens found, try "as"
        # aq_mat = after_quote_mat
        afquote_mat = re.search(r'[^“"”]+([“"”][^“"”]+[“"”])', span_st, re.I)
        if afquote_mat:
            # check if all capitalized
            if IS_DEBUG_MODE:
                print("checking quote_mat: [{}]".format(afquote_mat.group(1)))
            if is_all_title_or_the(afquote_mat.group(1)):
                return (start + afquote_mat.start(1),
                        start + afquote_mat.end(1),
                        afquote_mat.group(1))

        as_mat = re.search(r'.*\bas (.+)$', span_st, re.I)
        if as_mat:
            # check if all capitalized
            if IS_DEBUG_MODE:
                print("checking as_mat: [{}]".format(as_mat.group(1)))
            if is_all_title_or_the(as_mat.group(1)):
                return start + as_mat.start(1), start + as_mat.end(1), as_mat.group(1)

    if len(maybe_dterm_list) == 1:
        dterm = maybe_dterm_list[0]
        dterm_start, dterm_end, dterm_st = dterm.start(), dterm.end(), dterm.group()
        return start + dterm_start, start + dterm_end, dterm_st
    elif len(maybe_dterm_list) > 1:
        if IS_DEBUG_MODE:
            print("strange, more than 1 dterm for a party:")
            for i, dterm in enumerate(maybe_dterm_list):
                print("   question dterm #{}: {}".format(i, (dterm.start(),
                                                             dterm.end(),
                                                             dterm.group())))
        # first look through the possibility, pick one if seemed right
        # otherwise, finish and take the first one.  The the last one is really
        # risky since it's far away.
        for i, dterm in enumerate(maybe_dterm_list):
            dterm_start, dterm_end, dterm_st = dterm.start(), dterm.end(), dterm.group()
            if re.search(r'hereinafter\s+(referred|designated)', dterm_st):
                return start + dterm_start, start + dterm_end, dterm_st
            # '(the "Company") over "formerly known as xxx" in 41204.txt
            if re.search(r'^\(the\b', dterm_st):
                return start + dterm_start, start + dterm_end, dterm_st

        dterm = maybe_dterm_list[0]
        dterm_start, dterm_end, dterm_st = dterm.start(), dterm.end(), dterm.group()
        return start + dterm_start, start + dterm_end, dterm_st

    return None

# pylint: disable=too-many-return-statements
def is_invalid_party_phrase(line: str) -> bool:
    # this has to be exact phrase,
    # 'dominican republic bank'
    if re.search(r'^(dominican republic)$', line, re.I):
        return True

    # sometimes, a fully all-cap sentence will get there.
    # THIS DEED OF CHARGE (THIS “DEED”) IS DATED 2017 AND MADE BETWEEN:   tdocs/file21.txt
    if re.search(r'\b(this|dated|made)\b', line, re.I):
        return True

    # 'llc' is 'through or to Credit Suisse Securities (USA) LLC (“Credit Suisse”) and
    # Deutsche Bank Securities Inc. (“Deutsche Bank”)' in 40281.txt.
    # This needs special deal with '(USA)' by replacing it with something non-parens?
    if re.search(r'^(parent|company|manager|llc)$', line, re.I):
        return True

    if re.search(r'\b(received|^for\b|contracts?)\b', line, re.I):
        return True

    # 'California, USA', 'San Jose, Costa Rica'
    if re.search(r'\b(USA|Costa Rica|avenidas?)\s*$', line, re.I):
        if re.search(r'\b(bank|banco)\b', line, re.I):
            return False
        return True

    if is_ending_in_us_state(line):
        return True

    if is_address(line):
        return True

    #too short to be a party
    if len(line) <= 2:
        return True

    # catches other likely FP
    # the space here are significant
    if re.search(r'\b( this |page |agreements?)\b', line, re.I):
        return True

    #catches zip code pulls
    if ZIP_CODE_YEAR.search(line):
        return True

    lc_line = line.lower()
    if lc_line in INVALID_PARTIES_SET:
        return True

    return False


def is_address(line: str) -> bool:
    if re.search(r'(, new york|plaza)', line, re.I):
        return True
    if re.search(r'\b(street|cyprus|box|tbilisi)\b', line, re.I):
        return True

    # catches things that are likely part of an address
    if ADDRESS_PARTS_PAT.search(line):
        return True
    if addresses.classify(line) > 0.5:
        return True

    return False


# NOTE: jshaw
# This is called by somebody, so I am not modifying this
# pylint: disable=too-many-return-statements
def is_invalid_party(line, is_party=True) -> bool:
    #checks for some punctuation
    if ':' in line:  # or '/' in line:  'MY/ZP IP Group, Ltd.' 41207.txt
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


def remove_invalid_entities(entity_list: List[Tuple[int, int, str]], is_all_upper=False) \
    -> List[Tuple[int, int, str]]:
    out_list = []
    for entity in entity_list:
        unused_start, unused_end, entity_st = entity
        # this should be using some ML address detector
        # This is is_invalid_party_phrase(), not is_invalid_party()
        if is_invalid_party_phrase(entity_st):
            pass
        elif is_all_upper and is_all_less_3char_words(entity_st):
            pass
        else:
            out_list.append(entity)
    return out_list


def print_debug_list(alist: List, prefix: str, title: str = '') -> None:
    if IS_DEBUG_MODE:
        print('\n{}'.format(title))
        for i, elt in enumerate(alist):
            print('   {} elt #{}: {}'.format(prefix, i, elt))
        print()

def get_last_word(line: str, index: int) -> str:
    words = line[:index].split()
    if words:
        return words[-1]
    return ''

# if person party, following ' and ', is probably OK
def select_highly_likely_parties(entities: List[Tuple[int, int, str]],
                                 line: str) \
    -> List[Tuple[int, int, str]]:

    if IS_DEBUG_MODE:
        print_debug_list(entities, 'zz', title='select_highly_likely_parties()')

    out_list = []  # type: List[Tuple[int, int, str]]
    for entity in entities:
        start, end, entity_st = entity

        last_word = get_last_word(line, start)

        # print("select entity: {}".format(entity))
        # anything that's has org suffix, add it
        if re.search(r'\b(law|code)$', entity_st, re.I):
            pass
        elif nlputils.org_person_suffix_search(entity_st):
            # print('last word = [{}], entity_st = [{}]'.format(last_word, entity_st))
            # a Delaware Corporation (bad capitalization),
            # a Limited Liability company (bad capitalization also?)
            if last_word not in set(['a', 'an']):
                out_list.append(entity)
        # invalid word pattern
        elif is_invalid_party_phrase(entity_st):
            pass
            # print("skipping 1 [{}]".format(entity_st))
        elif entity_st[0] == '(':
            # skip all defined terms
            # print("skipping 1 [{}]".format(entity_st))
            pass
        else:
            # check if a person right after and
            prefix = line[max(0, start-5):start]
            # print("prefix = [{}]".format(prefix))
            if prefix == ' and ':
                out_list.append(entity)
                continue

            # check if entity_st is right before a paren, whatever
            # it is, it is a party name
            suffix = line[end:]
            # print("    sssufix = [{}]".format(suffix))
            if suffix.strip() and suffix.strip()[0] == '(':
                out_list.append(entity)
                continue

            #else:
            #    print("skipping 2 [{}]".format(entity_st))
    if IS_DEBUG_MODE:
        print('select highly out: {}'.format(out_list))
    return out_list



def extract_parties_term_list_from_itemized_line(line: str) -> List[Tuple[List[Tuple[int, int]],
                                                                          Optional[Tuple[int, int]]]]:
    print("extract_parties_term_list_from_itemized_line({})".format(line))
    paren1_mat_list = strutils.find_one_char_paren_mats(line)

    len_line = len(line)
    span_list = []
    prev_span_start = paren1_mat_list[0].end()
    for paren1_mat in paren1_mat_list[1:]:
        mstart, mend, unused_mat_st = paren1_mat.start(), paren1_mat.end(), paren1_mat.group()
        start = prev_span_start
        span_list.append((start, mstart, line[start:mstart]))
        prev_span_start = mend
    if prev_span_start != len_line:
        span_list.append((prev_span_start, len_line, line[prev_span_start:len_line]))

    for i, span in enumerate(span_list):
       print("  spantext[{}] = [{}]".format(i, span))

    result = []  # type: List[Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]]
    # will try 39811.txt, 39014.txt
    for start, end, span_st in span_list:
        parsed_sent = nlputils.PhrasedSent(span_st, is_chopped=True)
        parties_term_offset = parsed_sent.extract_orgs_term_offset()
        if parties_term_offset:
            parties_offset, term_offset = parties_term_offset

            parties_offset_out = []  # List[Tuple[int, int]]
            for pstart, pend in parties_offset:
                parties_offset_out.append((start + pstart, start + pend))
                print("234 party: [{}]".format(line[start + pstart: start + pend]))
            term_offset_out = None
            if term_offset:
                term_offset_out = (start + term_offset[0], start + term_offset[1])
                print("234 term: [{}]".format(line[term_offset_out[0]:term_offset_out[1]]))
            if parties_offset_out or term_offset:
                result.append((parties_offset_out, term_offset_out))

    return result


def extract_parties_term_list_from_party_line(line: str) \
    -> List[Tuple[List[Tuple[int, int]],
                  Optional[Tuple[int, int]]]]:
    """Extract all the party and its defined term from party_line.

    line is expected to be party_line.
    """

    start_offset = 0
    tmp_mat = re.match(r'for\s+value\s+received,?\s+(.*)$', line, re.I)
    if tmp_mat:
        start_offset = tmp_mat.start(1)
        line = line[start_offset:]
    else:
        between_list = list(BTWN_AMONG_PAT.finditer(line))
        # chop at first xxx entered ... by and between (wanted)
        if between_list:
            # last_between = between_list[-1]
            last_between = between_list[0]
            start_offset = last_between.end()
            line = line[start_offset:]
            # everything afterward is based on this line
            # need to set it back right before returning
    if IS_DEBUG_MODE:
        print("\nextract_parties_term_list_from_party_line()")
        print("chopped party_line = [{}]".format(line))

    # sometimes, a line can be chopped by accident before of form filling
    # in pdf docs.
    # 'Box.com (UK) Ltd, a company registered in England and Wales (company number'
    # We don't want to parse 'THIS AGREEMENT is dated 24th March 2015 and made between:'
    # which has no parties
    if partyutils.is_party_line_prefix_without_parties(line):
        return []

    # first try this aggressive itemize match inside party_line
    if re.match(r'\(?\S\)', line) and len(strutils.find_one_char_paren_mats(line)) > 1:
        parties_term_offset_list = extract_parties_term_list_from_itemized_line(line)
    else:
        # now find the verb, such as 'is entered', the 2nd 'agree' is just a guess
        entered_mat = re.search(r'( is\s+entered\s+into| agreed?\b)', line, re.I)
        if entered_mat:  # we don't try to find party across verb in party_line
            line = line[:entered_mat.start()]
        phrased_sent = nlputils.PhrasedSent(line, is_chopped=True)
        # phrased_sent.print_parsed()
        parties_term_offset_list = phrased_sent.extract_orgs_term_offset_list()

    if IS_DEBUG_MODE:
        for i, orgs_term_offset in enumerate(parties_term_offset_list):
            orgs, term = orgs_term_offset
            print("634 orgs_term #{}:".format(i))
            print("    orgs:")
            for j, org in enumerate(orgs):
                print("      #{} {}".format(j, line[org[0]:org[1]]))
            if term:
                print("    term:")
                print("         {}".format(line[term[0]:term[1]]))

    parties_term_offset_list = adjust_start_offset_ptoffset_list(parties_term_offset_list,
                                                                 start_offset)

    return parties_term_offset_list


def adjust_start_offset_ptoffset(parties_term_offset: Tuple[List[Tuple[int, int]],
                                                            Optional[Tuple[int, int]]],
                                 start: int) \
                                 -> Tuple[List[Tuple[int, int]],
                                          Optional[Tuple[int, int]]]:
    parties_offset, term_offset = parties_term_offset
    adj_parties_offset = []  # List[Tuple[int, int]]
    for party_offset in parties_offset:
        pstart, pend = party_offset
        adj_parties_offset.append((start + pstart, start + pend))
    term_out = None
    if term_offset:
        term_out = (start + term_offset[0], start + term_offset[1])
    return adj_parties_offset, term_out

def adjust_start_offset_ptoffset_list(parties_term_offset_list: List[Tuple[List[Tuple[int, int]],
                                                                           Optional[Tuple[int, int]]]],
                                      start: int) \
                                      -> List[Tuple[List[Tuple[int, int]],
                                                    Optional[Tuple[int, int]]]]:
    adj_parties_term_offset_list = [adjust_start_offset_ptoffset(parties_term_offset, start)
                                    for parties_term_offset in parties_term_offset_list]
    return adj_parties_term_offset_list


def extract_party_defined_term_list(line: str) \
    -> List[Tuple[Optional[Tuple[int, int, str, str]],
                  Optional[Tuple[int, int, str, str]]]]:
    """Extract all the party and its defined term from party_line.

    line is expected to be party_line.
    """

    start_offset = 0
    tmp_mat = re.match(r'for\s+value\s+received,?\s+(.*)$', line, re.I)
    if tmp_mat:
        start_offset = tmp_mat.start(1)
        line = line[start_offset:]
    else:
        between_list = list(BTWN_AMONG_PAT.finditer(line))
        # chop at first xxx entered ... by and between (wanted)
        if between_list:
            # last_between = between_list[-1]
            last_between = between_list[0]
            start_offset = last_between.end()
            line = line[start_offset:]
            # everything afterward is based on this line
            # need to set it back right before returning
    if IS_DEBUG_MODE:
        print("\nextract_party_defined_term_list()")
        print("chopped party_line = [{}]".format(line))

    # first try this aggressive itemize match inside party_line
    if re.match(r'\(?\S\)', line) and len(strutils.find_one_char_paren_mats(line)) > 1:
        entity_span_list = get_itemized_entity_span_list(line)
    else:
        # try with all entities in upper()
        entities = get_title_phrase_list(line, is_all_upper=True)
        num_org_suffix = len(nlputils.get_org_suffix_mat_list(line))

        if IS_DEBUG_MODE:
            print_debug_list(entities, 'zz', title='before_remove_invalid')
        entities = remove_invalid_entities(entities, is_all_upper=True)
        if IS_DEBUG_MODE:
            print_debug_list(entities, 'zz2', title='after_remove_invalid')
        # if only found 1 all_cap entities, try something else
        # Maybe only 1 company is all cap, the others are normal case
        # don't want to skip normal ones.  35436.txt
        if len(entities) < 2 or len(entities) < num_org_suffix:
            # otherwise, try with title(), but this might get addresses
            entities = get_title_phrase_list(line)

            # firs try only company names, if work, keep just that
            # otherwise, backdown.
            # Then, need to try less accurate way of removing addresses
            obvious_entities = select_highly_likely_parties(entities, line)
            if obvious_entities:
                entities = obvious_entities
            else:
                entities = remove_invalid_entities(entities)
        if IS_DEBUG_MODE:
            print()
            for i, entity in enumerate(entities):
                print("  y entity #{}: {}".format(i, entity))
            print()
        entity_span_list = split_by_phrase_offsets(line, entities)

    # pylint: disable=line-too-long
    paired_result = []  # type: List[Tuple[Optional[Tuple[int, int, str, str]], Optional[Tuple[int, int, str, str]]]]
    current_party = None
    for entity_span in entity_span_list:
        entity, span = entity_span
        if IS_DEBUG_MODE:
            print("\nentity_span:")
            print("   entity: {}".format(entity))
            print("   span: {}".format(span))

        estart, eend, entity_st = entity
        if estart == eend:
            current_party = None
        else:
            # normalize party, if needed
            estart, eend, entity_st = span_to_party(entity)
            current_party = (start_offset + estart, start_offset + eend, entity_st, 'party')

        sstart, send, span_st = span
        if sstart == send:  # must have current_party
            paired_result.append((current_party, None))
        else:
            # this will select the best dterm
            dterm = span_to_dterm(span)
            if dterm:
                sstart, send, span_st = dterm
                paired_result.append((current_party,
                                      (start_offset + sstart, start_offset + send, span_st, 'defined_term')))
            else:
                paired_result.append((current_party, None))

    if IS_DEBUG_MODE:
        print()
        for i, spanx in enumerate(paired_result):
            print('  #{} paired_party_dterm: {}'.format(i, spanx))
        print()

    return paired_result


REGISTERED_PAT = re.compile(r'\bregistered\b', re.I)


def extract_parties_from_party_line(astr: str, is_party: bool = True) -> List[List[str]]:
    """
    Return list of parties (which are lists of strings) of s (party line).
    is_party flag should be true for party provision but false for landlord / tenant provisions
    when is_party is false it will keep address-like extractions
    """
    astr = nlputils.first_sentence(astr)

    #bullet type parties won't contain between / among, extract anyway
    if partyutils.is_party_list_prefix_with_validation(astr):
        return extract_between_among(astr, is_party)

    # Try possible rules
    if ('between' in astr or 'among' in astr) or not is_party:
        return extract_between_among(astr, is_party)

    if len(list(REGISTERED_PAT.finditer(astr))) > 1:
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


def is_end_party_list(line: str, attrs: List[str]) -> bool:
    # IT IS AGREED
    if re.search(r'\b(background|whereas|definitions?|interpretation|it is)\b', line, re.I):
        return True
    # if there is 1)
    if partyutils.is_party_list_prefix_with_validation(line):
        return False
    if 'sechead' in attrs:  # sechead ends party lines
        return True
    # 'INDEX TO NOTE PURCHASE AGREEMENT', 39749.txt
    if re.search(r'\b(agreement|contract|lease)\b', line, re.I):
        return True
    # check for non-party words, section headings
    if re.search('(agreed\s+as\s+follows)\b', line, re.I):
        return True
    words = line.split(line)
    if len(words) > 2 and words[0].isupper() and words[1].isupper():
        return True
    if len(line) > 400:
        return True
    if len(line) > 200 and words[0].istitle():
        return True
    return False

# pylint: disable=invalid-name
def get_next_not_empty_se_paras_list(se_paras_attr_list: List[Tuple[int, int, str, List[str]]],
                                     i: int) \
                                     -> Optional[Tuple[int,
                                                       Tuple[int, int, str, List[str]]]]:
    if i < len(se_paras_attr_list):
        for jnum, se_paras_attr in enumerate(se_paras_attr_list[i+1:], i+1):
            _, _, line_st, unused_para_attrs = se_paras_attr
            if line_st.strip():
                words = strutils.get_regex_wwplus(line_st)

                if len(words) <= 2 and set(words).issubset(set(['the', 'The', 'THE',
                                                                'this', 'This', 'THIS',
                                                                'Agreement', 'AGREEMENT'])):
                    continue

                return jnum, se_paras_attr
    return None

def skip_non_english_line(se_paras_attr_list: List[Tuple[int, int, str, List[str]]],
                          i: int) \
                          -> Optional[Tuple[int,
                                            Tuple[int, int, str, List[str]]]]:
    first_time = True
    if i < len(se_paras_attr_list):
        for jnum, se_paras_attr in enumerate(se_paras_attr_list[i+1:], i+1):
            _, _, line_st, para_attrs = se_paras_attr
            if line_st.strip():
                words = strutils.get_regex_wwplus(line_st)

                # THIS AGREEMENT is made on the  “Agreement")
                # BETW EEN:
                # 20 16 (thedav of
                # (1) XXX Limited xxx
                # (2) XXX Bank Plc xxx as lender (the "Lender")
                # there are some bad cases which warrant skpping
                if len(words) < 5 and 'not_eng' in para_attrs:
                    if first_time:
                        continue
                    else:
                        return jnum, se_paras_attr

                return jnum, se_paras_attr
    return None


# mytest/doc101.txt
def extract_party_line_as_date_between(se_paras_attr_list: List[Tuple[int, int, str, List[str]]]) \
    -> Optional[Tuple[Tuple[int, int, str],
                      bool,
                      List[Tuple[int, int, str, List[str]]]]]:
    """Extract 3 lines with just "...Agreement\nDATED XXXX\nBetween\n"""

    nempty_se_paras_attr_list = []  # type: List[Tuple[int, int, str, List[str]]]
    # remember where in the original list is
    nempty_idx_list = []  # type: List[int]
    nempty_line_st_list = []  # type: List[str]
    count = 0
    for i, se_para_attr in enumerate(se_paras_attr_list):
        (sx, ex, line_st, para_attrs) = se_para_attr
        if not line_st.strip():
            continue
        nempty_se_paras_attr_list.append(se_para_attr)
        nempty_line_st_list.append(line_st)
        nempty_idx_list.append(i)
        count += 1
        if count > 500:
            break

    len_try_match = len(nempty_se_paras_attr_list)
    for j, nempty_se_paras_attr in enumerate(nempty_se_paras_attr_list):
        if j+3 < len_try_match:
            if re.search(r'\b(agreement|contract|lease)\b', nempty_line_st_list[j], re.I) and \
               re.match(r'(date|dated)\b', nempty_line_st_list[j+1], re.I) and \
               re.match(r'(between|among)\b', nempty_line_st_list[j+2], re.I):

                # make sure we are not in the title page.  Too diffiult to parse
                # a list of names without context
                num_and = 0
                num_as = 0
                for tmp_i in range(j+3, min(j+3+25, len(nempty_line_st_list))):
                    tmp_st = nempty_line_st_list[tmp_i]
                    if len(tmp_st) < 40 and \
                        re.search(r'\band\b', tmp_st, re.I):
                        num_and += 1
                    if len(tmp_st) < 60 and \
                        re.search(r'\bas\b', tmp_st, re.I):
                        num_as += 1
                    if num_and >= 3 or num_as >= 3:
                        return None

                sx, ex, party_line_st, _ = nempty_se_paras_attr_list[j+2]  # between
                is_party_list = True
                orig_idx = nempty_idx_list[j+3]
                return (sx, ex, party_line_st), is_party_list, se_paras_attr_list[orig_idx:]
    return None


def extract_party_line(paras_attr_list: List[Tuple[str, List[str]]]) \
    -> Optional[Tuple[Tuple[int, int, str],
                      bool,
                      List[Tuple[int, int, str, List[str]]]]]:
    """Extract party line.

    'party_line' detection is already performed and stored in paras_attr_list.
    Such detection is happening in htmltxtparser or pdftxtparser

    Returns: start-end-of-party-line: Tuple[start, end, party-line]
             is-party-list: bool
             list of start-end-line-attrs:
                  if is-party-list == True:
                     the lines containing the list
                  else:
                     whatever after the party line
    """

    # transform para_attr_list to se_paras_attr_list
    offset = 0
    # we want to know the start ane end of each line
    se_paras_attr_list = []  # type: List[Tuple[int, int, str, List[str]]]
    for line_st, para_attrs in paras_attr_list:
        line_st_len = len(line_st)
        se_paras_attr_list.append((offset, offset + line_st_len, line_st, para_attrs))
        offset += line_st_len + 1

    # try to detect 'agreement...\ndated xxx\namong\n'
    # if found, simply return the result
    pline_after_lines = extract_party_line_as_date_between(se_paras_attr_list)
    if pline_after_lines:
        return pline_after_lines

    # prev_line_st = ''
    # pylint: disable=invalid-name
    for i, (sx, ex, line_st, para_attrs) in enumerate(se_paras_attr_list):
        # print display party_line here
        if IS_DEBUG_DISPLAY_TEXT:
            attrs_st = '|'.join([str(attr) for attr in para_attrs])
            print("234623", i, '\t'.join([attrs_st, '[{}]'.format(line_st)]))

        # checks if bullet type party, joins all bullets into a line
        if 'party_line' in para_attrs and 'toc' not in para_attrs:
            if IS_DEBUG_MODE:
                print("extract_party_line(), party_line")

            # OCR makes the recognition of '.' and ',' unreliable, as a result
            # sent split on party_paragraph will lower f1 by around 5% (74% vs 69%).
            # 36468.txt,
            # nltk also split at a number,  'No. 581' 37114.txt,
            # 'This AMENDMENT NO. 6 (this “Amendment No. 6” )' in 40324.txt
            # Basically, any org name with abbreviations is at risk:
            # 'THIS AGREEMENT, ... between NSJ. Co., Ltd.,...'
            # sent_tokenize_list = sent_tokenize(line_st)
            # for sent in sent_tokenize_list:
            #    print("found sent: {}".format(sent))

            if partyutils.is_party_list_prefix_with_validation(line_st):
                is_party_list = True
                # include this line as a list
                return (sx, ex, line_st), is_party_list, se_paras_attr_list[i:]
            elif partyutils.is_party_list_with_end_between(line_st):
                between_and_mat = re.search(r'\b(between\s+(.*)\s*and)\s*$', line_st, re.I)
                between_mat = re.search(r'\b(between)\s*$', line_st, re.I)
                if between_mat or between_and_mat:
                    is_party_list = True
                    return (sx, ex, line_st), is_party_list, se_paras_attr_list[i+1:]


            # peek at the next line
            maybe_next_line = get_next_not_empty_se_paras_list(se_paras_attr_list, i)
            if maybe_next_line:
                next_i, (unused_next_sx, unused_next_ex, next_line_st, unused_next_para_attrs) = \
                        maybe_next_line
                # if the next line has only 'among' or 'between, the party groups are
                # after.  38608.txt
                if IS_DEBUG_MODE:
                    print("checking next line: [{}]".format(next_line_st))

                # this is a "match" not "search", ignore ";" if there is one
                if re.match(r'(betw\s*een|among|by and between|by)', next_line_st, re.I):
                    is_party_list = True
                    # skip some blank lines
                    maybe_nx2 = skip_non_english_line(se_paras_attr_list, next_i)
                    # print("maybe_nx2: {}".format(maybe_nx2))
                    if maybe_nx2:
                        nx2_i, (unused_nx2_sx, unused_nx2_ex,
                                unused_nx2_line_st, unused_nx2_para_attrs) = maybe_nx2
                        """
                        return ((next_sx, next_ex, next_line_st),
                                is_party_list, se_paras_attr_list[nx2_i:])
                        """
                        # (nx2_sx, nx2_ex, nx2_line_st),
                        return ((sx, ex, line_st),
                                is_party_list, se_paras_attr_list[nx2_i:])

                    # (next_sx, next_ex, next_line_st),
                    return ((sx, ex, line_st),
                            is_party_list, se_paras_attr_list[next_i+1:])


            # is_party_list = bool(re.search(r'(:|among|between)\s*$', line_st))
            is_party_list = is_list_party_line(line_st)
            return (sx, ex, line_st), is_party_list, se_paras_attr_list[i+1:]


        # if line_st.strip():
        #    prev_line_st = line_st

        # don't bother if party_line is too far from start of the doc
        if i > 2000:
            return None
    return None

# only for export-train/37320.txt
# This Stock Purchase Agreement is entered into as of April 26, 2011, by and between
#
# “SAFEDOX”:

# “PURCHASER”:
#
# SafedoX, Inc.
#
# New Beginnings Life Center, LLC
def seline_attrs_to_tabled_party_list_terms(se_after_paras_attr_list: List[Tuple[int, int, str, List[str]]],
                                            se_curline_idx: int) \
                                            -> List[Tuple[List[Tuple[int, int]],
                                                          Optional[Tuple[int, int]]]]:
    dterm_list = []  # type: List[Tuple[int, int]]
    party_list = []  # type: List[Tuple[int, int]]
    party_st_list = []  # type: List[str]
    len_doc_lines = len(se_after_paras_attr_list)

    while se_curline_idx < len_doc_lines:
        se_line_attrs = se_after_paras_attr_list[se_curline_idx]
        fstart, fend, linex, attr_list = se_line_attrs

        colon_mat = re.search(r':\s*', linex)
        if colon_mat:
            fend = fstart + colon_mat.start()
            party_st = linex[colon_mat.end():].strip()
            if party_st:
                party_list.append((colon_mat.end(), len(linex)))
                party_st_list.append(party_st)

        if linex[0] in '“"”':
            dterm_list.append((fstart, fend))
            se_curline_idx = move_next_non_empty_se_after_list(se_after_paras_attr_list,
                                                             se_curline_idx)
        else:
            break

    # cannot be just one party
    if len(dterm_list) < 2:
        return []

    if not party_list:
        # now capture the parties corresponding to those terms
        while se_curline_idx < len_doc_lines:
            se_line_attrs = se_after_paras_attr_list[se_curline_idx]
            fstart, fend, linex, attr_list = se_line_attrs
            party_list.append((fstart, fend))
            party_st_list.append(linex)
            se_curline_idx = move_next_non_empty_se_after_list(se_after_paras_attr_list,
                                                                se_curline_idx)

    has_valid_person_or_org = False
    out_list = []
    for i in range(len(dterm_list)):
        if i > len(party_list):
            break
        dterm = dterm_list[i]
        party = party_list[i]
        if party:
            party_se_other = partyutils.find_uppercase_party_name(party_st_list[i])
            if party_se_other:
                (unused_party_start, unused_party_end), other_start, is_valid = party_se_other
                if is_valid:
                    has_valid_person_or_org = True
            out_list.append(([party], dterm))
        else:
            out_list.append(([], dterm))

    # verified there is party or person found
    if not has_valid_person_or_org:
        return []

    return out_list


# (2) HSBC BANK PLC acting through its office at 8 Canada Square. 1 ondon LI4 5HQ as
# lender (the “Lender")
# it seems taking the last "as" or "paren" is the best heuristic
def extract_term_in_party_term(after_line: str, index: int) \
                               -> Optional[Tuple[int, int, str]]:
    """Try to find the term after party_name is found.

    First try to find the last "as xxxx".
    Then try the last parenthesis.
    """
    last_as_mat = None
    last_paren_mat = None

    start = index
    as_mat_list = list(re.finditer(r'\bas (.+)$', after_line, re.I))
    if as_mat_list:
        as_mat = as_mat_list[-1]  # last as_mat
        as_mat_start = as_mat.start(1)
        as_mat_end = as_mat.end(1)
        as_mat_st = as_mat.group(1)

        and_mat = re.search(r'(\s*;\s+and|\s+and)\b', as_mat.group(1), re.I)
        if and_mat:
            as_mat_end = as_mat.start(1) + and_mat.start()
            as_mat_st = after_line[as_mat_start:as_mat_end]

        if IS_DEBUG_MODE:
            print("checking last as_mat: [{}]".format(as_mat_st))
        # if is_all_title_or_the(as_mat_st):
        if is_last_title_or_quote(as_mat_st):   # 'the initial Borrower'
            # maybe clean up the extra words at the end
            last_as_mat_start = start + as_mat_start
            last_as_mat = last_as_mat_start, start + as_mat_end, as_mat_st

    paren_mat_list = list(re.finditer(r'\(([^\(]+)\)', after_line))
    if paren_mat_list:
        paren_mat = paren_mat_list[-1]  # last paren_mat

        if IS_DEBUG_MODE:
            print("checking last paren_mat: [{}]".format(paren_mat.group(1)))
        last_paren_mat_start = start + paren_mat.start(1)
        last_paren_mat = last_paren_mat_start, start + paren_mat.end(1), paren_mat.group(1)

    if last_as_mat and last_paren_mat:
        # take the later one
        if last_paren_mat_start > last_as_mat_start:
            return last_paren_mat
        return last_as_mat
    elif last_as_mat:
        return last_as_mat
    elif last_paren_mat:
        return last_paren_mat

    return None


def extract_and_parties_in_party_term(after_line: str, index: int) \
                                      -> List[Tuple[int, int]]:
    """Try to find the "and" parties after party_name is found.

    # (2) XXX International Insurance Limited, registered in England under...
    # and XXX Insurance  Limited, registered in...
    """

    result = []  # type: List[Tuple[int, int]]
    and_mat_list = list(re.finditer(r'\band\s+', after_line, re.I))
    for and_mat in and_mat_list:

        after_and_st = after_line[and_mat.end():]
        print("after_and_st: [{}]".format(after_and_st))
        # try to find term
        mat_with_start = partyutils.find_uppercase_party_name(after_and_st)

        party_start, party_end, party_st = -1, -1, ''
        # print("mat_with_start = {}".format(mat_with_start))
        if mat_with_start:
            offset = index + and_mat.end()
            (party_start, party_end), unused_other_start, is_valid = mat_with_start
            if is_valid:
                party_st = after_and_st[party_start:party_end]
                print("party_st: [{}]".format(party_st))
                result.append((offset + party_start, offset + party_end))

    return result

def seline_attrs_to_party_list_term(se_after_paras_attr_list: List[Tuple[int, int, str, List[str]]],
                                    se_curline_idx: int) \
                                    -> Tuple[Optional[Tuple[List[Tuple[int, int]],
                                                            Optional[Tuple[int, int]]]],
                                             int]:
    """Find party_list_term from a list of se_line_attrs.

    Returns
       1. None or (party_list, term)
       2. next_idx into se_after_paras_attr_list
    """
    if se_curline_idx >= len(se_after_paras_attr_list):
        return [], se_curline_idx

    se_line_attrs = se_after_paras_attr_list[se_curline_idx]
    fstart, _, first_line, attr_list = se_line_attrs

    # re.match(r'\(?[\div]\)\s*(.*)', first_line)
    party_or_term_st = first_line
    offset = 0
    mat = partyutils.match_party_list_prefix(party_or_term_st)
    if mat:
        party_or_term_st = mat.group(1)
        offset = mat.start(1)

    fstart += offset

    # try to find party
    party_se_list = partyutils.find_uppercase_party_name_list(party_or_term_st)

    if not party_se_list:
        return (None, se_curline_idx + 1)  # move toward next line

    result_party_se_list = []
    last_party_end = -1
    for party_se in party_se_list:
        pstart, pend = party_se
        if IS_DEBUG_MODE:
            party_st = party_or_term_st[pstart:pend]
            print('found party in group list: [{}]'.format(party_st))
        result_party_se_list.append((fstart + pstart, fstart + pend))
        last_party_end = pend

    if last_party_end != -1:
        term_start_idx = strutils.find_next_not_space_idx(party_or_term_st, last_party_end)
    else:
        term_start_idx = 0

    term_semat = extract_term_in_party_term(party_or_term_st[term_start_idx:],
                                            term_start_idx)
    term_start, term_end = 0, 0
    if term_semat:
        term_start, term_end, unused_term_st = term_semat
        result_term = (fstart + term_start, fstart + term_end)  # type: Optional[Tuple[int, int]]
    else:
        result_term = None

    # now see if there is any extra parties involved in that line group
    # (2) XXX International Insurance Limited, registered in England under...
    # and XXX Insurance  Limited, registered in...
    extra_party_se_list = extract_and_parties_in_party_term(party_or_term_st[term_start_idx:],
                                                            term_start_idx)
    for extra_party_se in extra_party_se_list:
        estart, eend = extra_party_se
        result_party_se_list.append((fstart + estart, fstart + eend))

    return (result_party_se_list, result_term), se_curline_idx + 1


def table_formatted_quote_list_org_list(line: str) -> bool:
    if re.match(r'\s*[“"”][^“"”]+[“"”]\s*:?\s*', line):
        return True
    return False


def move_next_if_is_empty_se_after_list(se_after_paras_attr_list: List[Tuple[int, int, str, List[str]]],
                                      se_curline_idx: int) \
                                      -> int:
    """Move to the next line if the current line is empty.

    Line with just 'and' are skipped also.
    """
    len_doc_lines = len(se_after_paras_attr_list)
    if se_curline_idx >= len_doc_lines:
        return se_curline_idx

    # move to next non-empty line
    se_line_attrs = se_after_paras_attr_list[se_curline_idx]
    _, _, linex, attr_list = se_line_attrs
    # 'and' or '- and -'
    while not linex or (len(linex) < 10 and re.search(r'\band\b', linex, re.I)):
        se_curline_idx += 1
        if se_curline_idx >= len_doc_lines:
            return se_curline_idx
        se_line_attrs = se_after_paras_attr_list[se_curline_idx]
        _, _, linex, attr_list = se_line_attrs
    return se_curline_idx


def move_next_non_empty_se_after_list(se_after_paras_attr_list: List[Tuple[int, int, str, List[str]]],
                                      se_curline_idx: int) \
                                      -> int:
    """Move to next non-empty line.

    Line with just 'and' are skipped also.
    """
    len_doc_lines = len(se_after_paras_attr_list)
    se_curline_idx += 1
    if se_curline_idx >= len_doc_lines:
        return se_curline_idx

    # move to next non-empty line
    se_line_attrs = se_after_paras_attr_list[se_curline_idx]
    _, _, linex, attr_list = se_line_attrs
    while not linex or (len(linex) < 10 and re.search(r'\band\b', linex, re.I)):
        se_curline_idx += 1
        if se_curline_idx >= len_doc_lines:
            return se_curline_idx
        se_line_attrs = se_after_paras_attr_list[se_curline_idx]
        _, _, linex, attr_list = se_line_attrs
    return se_curline_idx

def debug_parties_term_offsets(msg: str,
                               parties_term_offset: Tuple[List[Tuple[int, int]],
                                                           Optional[Tuple[int, int]]],
                               text: str) -> None:
    print(msg)
    if not parties_term_offset:
        print("[]")
        return
    parties_se, term_se = parties_term_offset
    for j, (party_start, party_end) in enumerate(parties_se):
        print("  party #{}\t({}, {})\t[{}]".format(j, party_start, party_end,
                                                   text[party_start:party_end]))
    if term_se:
        term_start, term_end = term_se
        print("  term\t({}, {})\t[{}]".format(term_start, term_end,
                                              text[term_start:term_end]))


# pylint: disable=line-too-long
def extract_parties_term_list_from_list_lines(se_after_paras_attr_list: List[Tuple[int, int, str, List[str]]]) \
    -> List[Tuple[List[Tuple[int, int]],
                  Optional[Tuple[int, int]]]]:
    if not se_after_paras_attr_list:
        return []

    # sometime, lines with only list prefix, such as '(1)' might get
    # deleted because they were considered as page numbers.  HTML specific
    # issue.  Why are such thing on a separate line?  Table in HTML?
    # for debug purpose
    if IS_DEBUG_MODE:
        print()
        print("\nextract_parties_from_list_lines()")
        for i in range(min(10, len(se_after_paras_attr_list))):
            print("  list_line #{}: {}".format(i, se_after_paras_attr_list[i]))
        print()

    result = []  # type: List[Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]]:
    se_curline_idx = 0
    se_line_attrs = se_after_paras_attr_list[se_curline_idx]
    _, _, linex, attr_list = se_line_attrs
    len_doc_lines = len(se_after_paras_attr_list)

    while not linex:
        se_curline_idx += 1
        if se_curline_idx >= len_doc_lines:
            return []
        se_line_attrs = se_after_paras_attr_list[se_curline_idx]
        _, _, linex, attr_list = se_line_attrs

    # now linex is not empty

    if partyutils.is_party_list_prefix_with_validation(linex):
        # party_list_term, se_curline_idx = \
        #     seline_attrs_to_party_list_term(se_after_paras_attr_list, se_curline_idx)
        item_prefix_offset = 0
        item_prefix_mat = partyutils.match_party_list_prefix(linex)
        if item_prefix_mat:
            linex_no_prefix = item_prefix_mat.group(1)
            item_prefix_offset = item_prefix_mat.start(1)
        phrased_sent = nlputils.PhrasedSent(linex_no_prefix, is_chopped=True)
        tmp_parties_term_offset = phrased_sent.extract_orgs_term_offset()
        debug_parties_term_offsets('tmp_parties_term_offset',
                                   tmp_parties_term_offset,
                                   linex_no_prefix)
        while tmp_parties_term_offset:
            se_line_attrs = se_after_paras_attr_list[se_curline_idx]
            fstart, _, first_line, attr_list = se_line_attrs
            party_list_term = adjust_start_offset_ptoffset(tmp_parties_term_offset, fstart + item_prefix_offset)

            result.append(party_list_term)

            # should be next non-empty line, but if empty, move on
            se_curline_idx = move_next_non_empty_se_after_list(se_after_paras_attr_list,
                                                               se_curline_idx)
            if se_curline_idx >= len_doc_lines:
                break

            # settled on the next non-empty line
            se_line_attrs = se_after_paras_attr_list[se_curline_idx]
            _, _, linex, attr_list = se_line_attrs
            # if no longer in found good list line
            if is_end_party_list(linex, attr_list):
                se_curline_idx += 1
                break

            #  party_list_term, se_curline_idx = \
            #     seline_attrs_to_party_list_term(se_after_paras_attr_list, se_curline_idx)

            item_prefix_offset = 0
            item_prefix_mat = partyutils.match_party_list_prefix(linex)
            if item_prefix_mat:
                linex_no_prefix = item_prefix_mat.group(1)
                item_prefix_offset = item_prefix_mat.start(1)
            else:
                linex_no_prefix = linex
            phrased_sent = nlputils.PhrasedSent(linex_no_prefix, is_chopped=True)
            tmp_parties_term_offset = phrased_sent.extract_orgs_term_offset()
            debug_parties_term_offsets('tmp_parties_term_offset',
                                       tmp_parties_term_offset,
                                       linex_no_prefix)

    elif table_formatted_quote_list_org_list(linex):
        result.extend(seline_attrs_to_tabled_party_list_terms(se_after_paras_attr_list,
                                                              se_curline_idx))
    else:
        # try to match party_name at beginning of a line, try twice
        # if the intervening line is short or of some patterns.
        num_attempt = 0
        while num_attempt < 3:

            phrased_sent = nlputils.PhrasedSent(linex, is_chopped=True)
            tmp_parties_term_offset = phrased_sent.extract_orgs_term_offset()
            print("tmp_parties_term_offset = {}".format(tmp_parties_term_offset))
            # party_name_sentinel = partyutils.find_uppercase_party_name(linex)
            if tmp_parties_term_offset:

                while tmp_parties_term_offset:
                    se_line_attrs = se_after_paras_attr_list[se_curline_idx]
                    fstart, _, first_line, attr_list = se_line_attrs
                    party_list_term = adjust_start_offset_ptoffset(tmp_parties_term_offset, fstart)

                    result.append(party_list_term)

                    # should be next non-empty line, but if empty, move on
                    se_curline_idx = move_next_non_empty_se_after_list(se_after_paras_attr_list,
                                                                         se_curline_idx)

                    if se_curline_idx >= len_doc_lines:
                        break

                    # settled on the next non-empty line
                    se_line_attrs = se_after_paras_attr_list[se_curline_idx]
                    start_linex, end_linex, linex, attr_list = se_line_attrs
                    # if no longer in found good list line
                    if is_end_party_list(linex, attr_list):
                        se_curline_idx += 1
                        num_attempt = 3
                        break

                    maybe_term_mat = re.match(r'hereinafter\s*', linex, re.I)
                    if maybe_term_mat:
                        term_line = linex[maybe_term_mat.end():]
                        last_parties_term = result[-1]
                        last_parties, _ = last_parties_term
                        result[-1] = (last_parties, (start_linex + maybe_term_mat.end(), end_linex))
                        # now move to next line
                        se_curline_idx = move_next_non_empty_se_after_list(se_after_paras_attr_list,
                                                                           se_curline_idx)
                        if se_curline_idx >= len_doc_lines:
                            break
                        # settled on the next non-empty line
                        se_line_attrs = se_after_paras_attr_list[se_curline_idx]
                        start_linex, end_linex, linex, attr_list = se_line_attrs

                    phrased_sent = nlputils.PhrasedSent(linex, is_chopped=True)
                    tmp_parties_term_offset = phrased_sent.extract_orgs_term_offset()
                    print("tmp_parties_term_offset222 = {}".format(tmp_parties_term_offset))
            else:
                if len(linex) < 40:
                    se_curline_idx = move_next_non_empty_se_after_list(se_after_paras_attr_list,
                                                                       se_curline_idx)
                    if se_curline_idx >= len_doc_lines:
                        return []
                    se_line_attrs = se_after_paras_attr_list[se_curline_idx]
                    _, _, linex, attr_list = se_line_attrs
                else:
                    # a long line, but has no uppercase party name prefix
                    return []
            num_attempt += 1

    if IS_DEBUG_MODE:
        print()
        for offsets_pair in result:
            print("7342 offsets_pair: {}".format(offsets_pair))
        print()

    return result


# pylint: disable=line-too-long
def extract_parties_term_list_from_list_lines_backup(se_after_paras_attr_list: List[Tuple[int, int, str, List[str]]]) \
    -> List[Tuple[List[Tuple[int, int]],
                  Optional[Tuple[int, int]]]]:
    if not se_after_paras_attr_list:
        return []

    count_other_line = 0
    is_last_char_lower = False

    # sometime, lines with only list prefix, such as '(1)' might get
    # deleted because they were considered as page numbers.  HTML specific
    # issue.  Why are such thing on a separate line?  Table in HTML?
    # for debug purpose
    if IS_DEBUG_MODE:
        print()
        print("\nextract_parties_from_list_lines()")
        for i in range(min(10, len(se_after_paras_attr_list))):
            print("  list_line #{}: {}".format(i, se_after_paras_attr_list[i]))
        print()

    result = []  # type: List[Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]]:
    se_curline_idx = 0
    se_line_attrs = se_after_paras_attr_list[se_curline_idx]
    _, _, linex, attr_list = se_line_attrs
    len_doc_lines = len(se_after_paras_attr_list)

    while not linex:
        se_curline_idx += 1
        if se_curline_idx >= len_doc_lines:
            return []
        se_line_attrs = se_after_paras_attr_list[se_curline_idx]
        _, _, linex, attr_list = se_line_attrs

    # now linex is not empty

    if partyutils.is_party_list_prefix_with_validation(linex):
        party_list_term, se_curline_idx = \
            seline_attrs_to_party_list_term(se_after_paras_attr_list, se_curline_idx)
        while party_list_term:
            result.append(party_list_term)

            # should be next non-empty line, but if empty, move on
            se_curline_idx = move_next_if_is_empty_se_after_list(se_after_paras_attr_list,
                                                                 se_curline_idx)
            if se_curline_idx >= len_doc_lines:
                break

            # settled on the next non-empty line
            se_line_attrs = se_after_paras_attr_list[se_curline_idx]
            _, _, linex, attr_list = se_line_attrs
            # if no longer in found good list line
            if is_end_party_list(linex, attr_list):
                se_curline_idx += 1
                break

            party_list_term, se_curline_idx = \
                seline_attrs_to_party_list_term(se_after_paras_attr_list, se_curline_idx)
    elif table_formatted_quote_list_org_list(linex):
        result.extend(seline_attrs_to_tabled_party_list_terms(se_after_paras_attr_list,
                                                              se_curline_idx))
    else:
        # try to match party_name at beginning of a line, try twice
        # if the intervening line is short or of some patterns.
        num_attempt = 0
        if num_attempt < 3:
            phrased_sent = nlputils.PhrasedSent(linex, is_chopped=True)
            tmp_parties_term_offset = phrased_sent.extract_orgs_term_offset()
            print("tmp_parties_term_offset = {}".format(tmp_parties_term_offset))
            party_name_sentinel = partyutils.find_uppercase_party_name(linex)
            if party_name_sentinel:

                party_list_term, se_curline_idx = \
                    seline_attrs_to_party_list_term(se_after_paras_attr_list, se_curline_idx)
                while party_list_term:
                    result.append(party_list_term)

                    # should be next non-empty line, but if empty, move on
                    se_curline_idx = move_next_if_is_empty_se_after_list(se_after_paras_attr_list,
                                                                         se_curline_idx)

                    if se_curline_idx >= len_doc_lines:
                        break

                    # settled on the next non-empty line
                    se_line_attrs = se_after_paras_attr_list[se_curline_idx]
                    _, _, linex, attr_list = se_line_attrs
                    # if no longer in found good list line
                    if is_end_party_list(linex, attr_list):
                        se_curline_idx += 1
                        num_attempt = 3
                        break

                    party_list_term, se_curline_idx = \
                        seline_attrs_to_party_list_term(se_after_paras_attr_list, se_curline_idx)
            else:
                if len(linex) < 40:
                    se_curline_idx = move_next_non_empty_se_after_list(se_after_paras_attr_list,
                                                                       se_curline_idx)
                    if se_curline_idx >= len_doc_lines:
                        return []
                    se_line_attrs = se_after_paras_attr_list[se_curline_idx]
                    _, _, linex, attr_list = se_line_attrs
                else:
                    # a long line, but has no uppercase party name prefix
                    return []
            num_attempt += 1

    if IS_DEBUG_MODE:
        print()
        for offsets_pair in result:
            print("7342 offsets_pair: {}".format(offsets_pair))
        print()

    return result


def is_list_party_line(line: str) -> bool:
    # any party_line ends with ':' is considered a list party prefix
    org_suffix_list = nlputils.get_org_suffix_mat_list(line)
    # print("org_suffix_list 32523: {}".format(org_suffix_list))
    # xxx, yyy, and my confirms its agreements as follow:
    if len(org_suffix_list) > 1:
        # parties are already mentioned, not list_party_line
        return False
    if len(line) > 250:
        # too long, parties are probably mentioned
        return False
    # 40349
    if re.search(r'\b(confirm)', line):
        return False
    if re.search(r'(:|among|between)\s*$', line):
        return True
    return False

# paras_text is not used for title right now
# The first Tuple[int, int] is the party offset
# the Optional[Tuple[int, int]] is the defined term offsets
def extract_offsets(paras_attr_list: List[Tuple[str, List[str]]],
                    para_text: str) \
    -> List[Tuple[Optional[Tuple[int, int]],
                  Optional[Tuple[int, int]]]]:
    """Return list of parties (lists of (start, inclusive-end) offsets)."""

    out_list = []  # type: List[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]]

    pline_after_lines = extract_party_line(paras_attr_list)
    # pylint: disable=too-many-nested-blocks
    if pline_after_lines:
        start_end_partyline, is_list_party, after_se_paras_attr_list = pline_after_lines
        start, unused_end, party_line = start_end_partyline

        if IS_DEBUG_MODE:
            print('\nparty_line: (%d, %d)' % (start, unused_end))
            print(party_line)
            print("is_list_party = {}".format(is_list_party))

        # Sometimes if is_list_party, still have parties in party line only.
        # So, try that first.  If found parties, don't bother with the is_party list
        # print("ok, party_line: [{}]".format(nlputils.first_sentence(party_line)))
        first_sent = nlputils.first_sentence(party_line)
        sec_sent_start = strutils.find_non_space_index(party_line[len(first_sent):])
        second_sent = party_line[len(first_sent) + sec_sent_start:] if sec_sent_start != -1 else ''
        parties_term_offset_list = extract_parties_term_list_from_party_line(first_sent)

        if not parties_term_offset_list and \
           not is_list_party and \
           second_sent:  # didn't find parties in the first sentence
            # maybe check for number of org_suffix before going into 2nd sent
            parties_term_offset_list = extract_parties_term_list_from_party_line(second_sent)
            if parties_term_offset_list:
                start = start + len(first_sent) + sec_sent_start

        if parties_term_offset_list:
            # need to adjust the offset because used first_sent
            parties_term_offset_list = adjust_start_offset_ptoffset_list(parties_term_offset_list, start)
        elif is_list_party:
            # all the parties are in after_se_paras_attr_list
            parties_term_offset_list = extract_parties_term_list_from_list_lines(after_se_paras_attr_list)

        # print('parties_term_offset_list: {}'.format(parties_term_offset_list))

        for parties_term_offset in parties_term_offset_list:
            parties_offset, term_offset = parties_term_offset
            if len(parties_offset) > 1:
                for party_offset in parties_offset[:-1]:
                    # print('party_offset: {}'.format(party_offset))
                    out_list.append((party_offset, None))

            if parties_offset:  # there must be some parties
                out_list.append((parties_offset[-1],
                                 term_offset))
            else:
                out_list.append((None, term_offset))

    if IS_DEBUG_MODE:
        print()
        for i, (party_y, term_y) in enumerate(out_list):
            if party_y:
                start, end = party_y
                print("  #{} found party: [{}]".format(i, para_text[start:end]))
            if term_y:
                start, end = term_y
                print("  #{} found dterm: [{}]".format(i, para_text[start:end]))
        print()

    return out_list


# pylint: disable=too-few-public-methods
class PartyAnnotator:

    # pylint: disable=unused-argument
    def __init__(self, provision: str) -> None:
        self.provision = 'party'

    # pylint: disable=no-self-use
    def extract_provision_offsets(self,
                                  paras_with_attrs: List[Tuple[str, List[str]]],
                                  paras_text: str) \
        -> List[Tuple[Optional[Tuple[int, int]],
                      Optional[Tuple[int, int]]]]:
        return extract_offsets(paras_with_attrs, paras_text)
