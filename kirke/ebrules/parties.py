# pylint: disable=too-many-lines
import itertools
import re
import string
from typing import List, Match, Optional, Tuple

# from nltk.tokenize import sent_tokenize

from kirke.ebrules import titlesold, addresses
from kirke.utils import strutils

IS_DEBUG_DISPLAY_TEXT = False
IS_DEBUG_MODE = False

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

# 'DELL RECEIVABLES FINANCING 2016 D.A.C", year is valid
# ZIP_CODE_YEAR = re.compile(r'\d{{4}}|\b{}\b'.format(UK_STD))
ZIP_CODE_YEAR = re.compile(r'\b{}\b'.format(UK_STD))
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
           line[0].isupper() or \
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

# bank is after 'n.a.' because 'bank, n.a.' is more desirable
# 'Credit Suisse Ag, New York Branch', 39893.txt,  why 'branch' is early
# TODO, handle "The bank of Nova Scotia", this is NOT org suffix case
# TODO, not handling 'real estate holdings fiv'
# TODO, remove 'AS AMENDED' as a party, 'the customer'?
# TODO, 'seller, seller' the context?
ORG_SUFFIX_PAT = re.compile(r' ('
                            r'branch|ag|company|co|corp|corporation|d\.\s*a\.\s*c|inc|'
                            r'incorporated|llc|gmbh|foundation|enterprises?'
                            r'l\.\s*l\.\s*c|ulc|'
                            r'ltd|limited|lp|l\.\s*p|limited partnership|n\.\s*a|plc|'
                            r's\.\s*r\.\s*l|'
                            r'pca|pty|holdings?|hospital|'
                            r'bank|trust|association|group|sas|s\.\s*a|sa|c\.\s*v|cv'
                            r')\b\.?',
                            re.I)

# for Spanish org names
ORG_PREFIX_PAT = re.compile(r'\b(asociacion|hospital)\.?', re.I)

def get_org_suffix_mat_list(line: str) -> List[Match[str]]:
    """Get all org suffix matching mat extracted from line.

     Because of capitalization concerns, we are making
     a pass to make sure, it is not just 'a limited company'"""

    lc_mat_list = list(ORG_SUFFIX_PAT.finditer(line))
    # print("2342 lc_mat_list = {}".format(lc_mat_list))
    result = []  # type: List[Match[str]]
    for lc_mat in lc_mat_list:
        # we need the strip because we might have prefix space
        if lc_mat.group().strip()[0].isupper():
            result.append(lc_mat)

    # when there is adjcent ones, take the last one
    # 'xxx Group, Ltd.', will return 'ltd'
    prev_mat = None
    result2 = [] # type: List[Match[str]]
    # Only if we now that the current mat is not adjacent to
    # the previous mat, we can add previous mat.
    # Remember the last one.
    for amat in result:
        # print('amt = ({}, {}) {}'.format(amat.start(), amat.end(), amat.group()))
        # 2 is chosen, just in case, normally the diff is 1
        if prev_mat and amat.start() - prev_mat.end() > 2:
            result2.append(prev_mat)
        prev_mat = amat
    if prev_mat:
        result2.append(prev_mat)

    #if IS_DEBUG_MODE:
    #    print()
    #    for i, mat in enumerate(result2):
    #        print("mat #{}: {}".format(i, mat))
    #    print()
    return result2


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
    num_org_suffix = len(get_org_suffix_mat_list(line))
    if is_all_upper and len(title_phrase_list) < num_org_suffix:
        return []

    out_list = []  # type: List[Tuple[int, int, str]]
    for title_phrase in title_phrase_list:
        fstart, unused_fend, unused_first_word = title_phrase[0]
        unused_lstart, lend, unused_last_word = title_phrase[-1]
        out_list.append((fstart, lend, line[fstart:lend]))
    return out_list


def get_itemized_entity_span_list(line: str) -> List[Tuple[Tuple[int, int, str],
                                                           Tuple[int, int, str]]]:
    parens1_mat_list = strutils.get_consecutive_one_char_parens_mats(line)

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


def is_all_title_or_the(line: str) -> bool:
    words = line.split()
    if not words:
        return False
    for word in words:
        if not (word.isupper() or word.istitle() or word.lower() == 'the'):
            return False
    return True

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

    # 'llc' is 'through or to Credit Suisse Securities (USA) LLC (“Credit Suisse”) and
    # Deutsche Bank Securities Inc. (“Deutsche Bank”)' in 40281.txt.
    # This needs special deal with '(USA)' by replacing it with something non-parens?
    if re.search(r'^(parent|company|manager|llc)$', line, re.I):
        return True

    if re.search(r'\b(received|^for\b|contracts?)\b', line, re.I):
        return True

    # 'California, USA', 'San Jose, Costa Rica'
    if re.search(r'\b(USA|Costa Rica|avenidas?)\s*$', line, re.I):
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


def remove_invalid_entities(entity_list: List[Tuple[int, int, str]]) \
    -> List[Tuple[int, int, str]]:
    out_list = []
    for entity in entity_list:
        unused_start, unused_end, entity_st = entity
        # this should be using some ML address detector
        # This is is_invalid_party_phrase(), not is_invalid_party()
        if is_invalid_party_phrase(entity_st):
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

# if person party, following ' and ', is probably OK
def select_highly_likely_parties(entities: List[Tuple[int, int, str]],
                                 line: str) \
    -> List[Tuple[int, int, str]]:

    if IS_DEBUG_MODE:
        print_debug_list(entities, 'zz', title='select_highly_likely_parties()')

    out_list = []  # type: List[Tuple[int, int, str]]
    for entity in entities:
        start, end, entity_st = entity

        # print("select entity: {}".format(entity))
        # anything that's has org suffix, add it
        if re.search(r'\b(law|code)$', entity_st, re.I):
            pass
        elif ORG_SUFFIX_PAT.search(entity_st):
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


def extract_party_defined_term_list(line: str) \
    -> List[Tuple[Optional[Tuple[int, int, str, str]],
                  Optional[Tuple[int, int, str, str]]]]:
    """Extract all the party and its defined term from party_line."""

    between_list = list(BTWN_AMONG_PAT.finditer(line))
    # chop at first xxx entered ... by and between (wanted)
    start_offset = 0
    if between_list:
        last_between = between_list[-1]
        start_offset = last_between.end()
        line = line[start_offset:]
        # everything afterward is based on this line
        # need to set it back right before returning
    if IS_DEBUG_MODE:
        print("\nextract_party_defined_term_list()")
        print("chopped party_line = [{}]".format(line))

    # first try this aggressive itemize match inside party_line
    if re.match(r'\(\S\)', line) and len(strutils.get_consecutive_one_char_parens_mats(line)) > 1:
        entity_span_list = get_itemized_entity_span_list(line)
    else:
        # try with all entities in upper()
        entities = get_title_phrase_list(line, is_all_upper=True)
        # if IS_DEBUG_MODE:
        #     print_debug_list(entities, 'zz', title='before_remove_invalid')
        entities = remove_invalid_entities(entities)
        # if IS_DEBUG_MODE:
        #    print_debug_list(entities, 'zz2', title='after_remove_invalid')
        # if only found 1 all_cap entities, try something else
        # Maybe only 1 company is all cap, the others are normal case
        # don't want to skip normal ones.  35436.txt
        if len(entities) < 2:
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


# TODO, jshaw, del later
# pylint: disable=invalid-name
def extract_party_defined_term_list_old(line: str) \
    -> List[Tuple[Optional[Tuple[int, int, str, str]],
                  Optional[Tuple[int, int, str, str]]]]:
    """Extract all the party and its defined term from party_line."""

    between_list = list(BTWN_AMONG_PAT.finditer(line))
    # chop at first xxx entered ... by and between (wanted)
    start_offset = 0
    if between_list:
        last_between = between_list[-1]
        start_offset = last_between.end()
        line = line[start_offset:]
        # everything afterward is based on this line
        # need to set it back right before returning
    print("chopped party_line = [{}]".format(line))

    entities = get_title_phrase_list(line)
    print()
    for i, entity in enumerate(entities):
        print("  x entity #{}: {}".format(i, entity))
    print()

    entity_span_list = split_by_phrase_offsets(line, entities)
    for entity_span in entity_span_list:
        entity, span = entity_span
        print("entity_span:")
        print("   entity: {}".format(entity))
        print("   span: {}".format(span))

    company_list = list(A_COMPANY_PAT.finditer(line))
    and_list = list(AND_PAT.finditer(line))
    se_mat_list = []  # type: List[Tuple[int, int, str]]
    for mat in itertools.chain(company_list, and_list):
        se_mat_list.append((mat.start(), mat.end(), mat.group()))
    if IS_DEBUG_MODE:
        sorted_se_mat_list = sorted(se_mat_list)
        print("sorted_se_mat_list: {}".format(sorted_se_mat_list))

    line_len = len(line)
    span_st_list = []  # type: List[Tuple[int, int, str]]
    start = 0
    for se_mat in sorted(se_mat_list):
        mat_start, mat_end, _ = se_mat
        span_st_list.append((start, mat_start, line[start:mat_start]))
        start = mat_end

    if start < line_len:
        span_st_list.append((start, line_len, line[start:]))

    if IS_DEBUG_MODE:
        for i, spanx in enumerate(span_st_list):
            print('  span_x[{}] = {}'.format(i, spanx))

    result = []  # type: List[Tuple[int, int, str, str]]
    # put back the start_offset because we might have done some chopping
    # to remove non-party prefix before
    for span_x in span_st_list:
        start, end, span_st = span_x
        parens_mat_list = list(PARENS_PAT.finditer(span_st))
        # found a defined term
        if parens_mat_list:
            last_parens_mat = parens_mat_list[-1]
            result.append((start_offset + start + last_parens_mat.start(),
                           start_offset + start + last_parens_mat.end(),
                           last_parens_mat.group(),
                           'defined_term'))
        else:
            if is_all_title(span_st):
                print("pass titled test: [{}]".format(span_st))
                result.append((start_offset + start,
                               start_offset + end,
                               span_st,
                               'party'))
            else:
                print("failed titled test: [{}]".format(span_st))

    # pylint: disable=line-too-long
    paired_result = []  # type: List[Tuple[Optional[Tuple[int, int, str, str]], Optional[Tuple[int, int, str, str]]]]
    current_party = None
    for spanx_t4 in result:
        # print('  #{} pdterm: {}'.format(i, spanx_t4))
        start, end, span_st, span_type = spanx_t4
        if span_type == 'party':
            if current_party:  # must be no 'defined_term' before
                paired_result.append((current_party, None))
            current_party = spanx_t4
        else:
            if current_party:
                paired_result.append((current_party, spanx_t4))
                current_party = None
            else:
                paired_result.append((None, spanx_t4))
            # we don't care about defined_term state
    if current_party:
        paired_result.append((current_party, None))

    # if IS_DEBUG_MODE:
    #    for i, spanx_t4 in enumerate(paired_result):
    #        print('  #{} paired_party_dterm: {}'.format(i, spanx_t4))

    return paired_result


REGISTERED_PAT = re.compile(r'\bregistered\b', re.I)


def extract_parties_from_party_line(astr: str, is_party: bool = True) -> List[List[str]]:
    """
    Return list of parties (which are lists of strings) of s (party line).
    is_party flag should be true for party provision but false for landlord / tenant provisions
    when is_party is false it will keep address-like extractions
    """
    astr = first_sentence(astr)

    #bullet type parties won't contain between / among, extract anyway
    if is_list_prefix(astr):
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


def re_match_any_list_prefix(line: str) -> Match[str]:
    # I have seen up to 13 companies
    return re.match(r'\(?\s*[\divx]+\s*\)\s*(.*)', line, re.I)
    # re.match(r'\(?[\divx]+\)', line) or \

def is_list_prefix(line: str) -> bool:
    if re_match_any_list_prefix(line) or \
       re.match(r'Party \S+\s*:', line, re.I):
        if re.search(r'\b(engages?|product)\b', line, re.I):
            return False
        return True
    return False

def is_end_party_list(line: str, attrs: List[str]) -> bool:
    if 'sechead' in attrs:  # sechead ends party lines
        return True
    # check for non-party words, section headings
    if re.search(r'\b(background|whereas|agreed\s+as\s+follows)\b', line, re.I):
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
                                     i: int) -> Optional[Tuple[int,
                                                               Tuple[int, int, str, List[str]]]]:
    if i < len(se_paras_attr_list):
        for jnum, se_paras_attr in enumerate(se_paras_attr_list[i+1:], i+1):
            _, _, line_st, unused_para_attrs = se_paras_attr
            if line_st.strip():
                return jnum, se_paras_attr
    return None

def extract_party_line(paras_attr_list: List[Tuple[str, List[str]]]) \
    -> Optional[Tuple[Tuple[int, int, str],
                      bool,
                      List[Tuple[int, int, str, List[str]]]]]:

    offset = 0
    # we want to know the start ane end of each line
    se_paras_attr_list = []  # type: List[Tuple[int, int, str, List[str]]]
    for line_st, para_attrs in paras_attr_list:
        line_st_len = len(line_st)
        se_paras_attr_list.append((offset, offset + line_st_len, line_st, para_attrs))
        offset += line_st_len + 1

    # prev_line_st = ''
    # pylint: disable=invalid-name
    for i, (sx, ex, line_st, para_attrs) in enumerate(se_paras_attr_list):
        # print display party_line here
        if IS_DEBUG_DISPLAY_TEXT:
            attrs_st = '|'.join([str(attr) for attr in para_attrs])
            print(i, '\t'.join([attrs_st, '[{}]'.format(line_st)]))

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

            if is_list_prefix(line_st):
                is_party_list = True
                # include this line as a list
                return (sx, ex, line_st), is_party_list, se_paras_attr_list[i:]

            # peek at the next line
            maybe_next_line = get_next_not_empty_se_paras_list(se_paras_attr_list, i)
            if maybe_next_line:
                next_i, (next_sx, next_ex, next_line_st, unused_next_para_attrs) = \
                maybe_next_line
                # if the next line has only 'among' or 'between, the party groups are
                # after.  38608.txt
                if IS_DEBUG_MODE:
                    print("checking next line: [{}]".format(next_line_st))
                # this is a "match" not "search", ignore ";" if there is one
                if re.match(r'(between|among|by and between|by)', next_line_st, re.I):
                    is_party_list = True
                    # skip some blank lines
                    maybe_nx2 = get_next_not_empty_se_paras_list(se_paras_attr_list, next_i)
                    if maybe_nx2:
                        nx2_i, (unused_nx2_sx, unused_nx2_ex,
                                unused_nx2_line_st, unused_nx2_para_attrs) = maybe_nx2
                        return ((next_sx, next_ex, next_line_st),
                                is_party_list, se_paras_attr_list[nx2_i:])

            # is_party_list = bool(re.search(r'(:|among|between)\s*$', line_st))
            is_party_list = is_list_party_line(line_st)
            return (sx, ex, line_st), is_party_list, se_paras_attr_list[i+1:]


        # if line_st.strip():
        #    prev_line_st = line_st

        # don't bother if party_line is too far from start of the doc
        if i > 2000:
            return None
    return None

def tabled_party_line_group_to_party_terms(party_line_group_list:
                                           List[List[Tuple[int, int, str, List[str]]]]) \
    -> List[Tuple[Optional[Tuple[int, int]],
                  Optional[Tuple[int, int]]]]:
    dterm_list = []
    party_list = []
    for party_line_group in party_line_group_list:
        fstart, fend, first_line, _ = party_line_group[0]
        colon_mat = re.search(r'\s*:\s*$', first_line)
        if colon_mat:  # we remove the column, such as from '"Purchase":'
            fend -= len(colon_mat.group())
        if first_line[0] in '“"”':
            dterm_list.append((fstart, fend))
        else:
            party_list.append((fstart, fend))

    out_list = []
    for party, dterm in itertools.zip_longest(party_list, dterm_list):
        out_list.append((party, dterm))
    return out_list

def find_first_non_title_word_mat(line: str) -> Optional[Tuple[Match[str], int]]:
    # some companies, such as 'eBrevia' is not title_case
    # we also want to handle single letter, such as 'a' in 'a compny'
    maybe_mat = re.search(r' ([a-z\d]|[a-z\d][a-z\d]\S*)\b', line)  # no re.I here

    if not maybe_mat:
        return None

    # run into and "and" of multiple party
    if IS_DEBUG_MODE:
        print('find_first_non_title_word_mat, maybe_mat = [{}]'.format(maybe_mat.group()))
    if maybe_mat.group() == ' and' or strutils.is_digits(maybe_mat.group()):
        after_line = line[maybe_mat.end():]
        if IS_DEBUG_MODE:
            print('after_line = [{}]'.format(after_line))

        maybe_mat2 = re.search(r' ([a-z\d]|[a-z\d][a-z\d]\S*)\b', after_line)  # no re.I here

        if not maybe_mat2:
            return None

        return maybe_mat2, maybe_mat.end() + maybe_mat2.start()
    return maybe_mat, maybe_mat.start()


def is_all_english_title_case(line: str) -> bool:
    words = strutils.get_alpha_words(line, is_lower=False)
    if not words:
        return False
    for word in words:
        # check if preposition, lc
        if not (word.istitle() or
                word.isupper() or
                word in set(['of', 'the', 'as', 'from', 'to',
                             'between', 'among', 'de', 'and', 'or'])):
            # print("failed tt: [{}]".format(word))
            return False
    return True

def split_party_term_effort_1(line: str) -> Tuple[Tuple[int, int],
                                                  Tuple[int, int]]:
    if IS_DEBUG_MODE:
        print("split_party_term_efforts_1({})".format(line))

    if is_all_english_title_case(line):
        as_mat = re.search(r' as ', line)
        if as_mat:
            return ((0, as_mat.start()),
                    (as_mat.end(), len(line)))

        parens_mat_list = list(PARENS_PAT.finditer(line))
        if parens_mat_list:
            paren0_mat = parens_mat_list[0]
            st1 = line[0:paren0_mat.start()]
            espace_mat = re.search(r'\s+$', st1)
            if espace_mat:
                return (0, espace_mat.start()), (paren0_mat.start(), paren0_mat.end())
            return (0, paren0_mat.start()), (paren0_mat.start(), paren0_mat.end())

    return ((0, 0), (0, 0))


# this is for party list
def party_line_group_to_party_term(party_line_list:
                                   List[Tuple[int, int, str, List[str]]]) \
    -> Tuple[Optional[Tuple[int, int]],
             Optional[Tuple[int, int]]]:
    fstart, unused_fend, first_line, _ = party_line_list[0]
    # last_start, unused_last_end, last_line, _ = party_line_list[-1]

    if IS_DEBUG_MODE:
        print()
        print('party_line_group_to_party_term({})'.format(first_line))
        for i, party_linex in enumerate(party_line_list):
            unused_z1_start, unused_z2_end, z2_line, _ = party_linex
            print("  z2_line #{}: [{}]".format(i, z2_line))

    p0_start, p0_end, p0_line, _ = party_line_list[0]
    line_buf = p0_line
    prev_end = p0_end
    for party_linex in party_line_list[1:]:
        pstart, pend, pline, _ = party_linex
        pad_line = ' ' * (pstart -prev_end)
        line_buf += pad_line
        line_buf += pline

    fstart, first_line = p0_start, line_buf

    # re.match(r'\(?[\div]\)\s*(.*)', first_line)
    mat = re_match_any_list_prefix(first_line)
    if not mat:
        # now try "Party A:"
        mat = re.match(r'Party\s*\S+\s*:\s*(.*)', first_line)
    if mat:
        party_start = mat.start(1)
        party_end = mat.end(1)
        party_st = mat.group(1)

        split_party_term_offsets = split_party_term_effort_1(party_st)
        # should return correct offsets now
        (pstart, pend), (tstart, tend) = split_party_term_offsets
        if pstart != pend and tstart != tend:
            # xx_party_st = party_st[pstart:pend]
            # print("xx party_st: [{}]".format(xx_party_st))
            return ((fstart + party_start + pstart, fstart + party_start + pend),
                    (fstart + party_start + tstart, fstart + party_start + tend))
        if pstart == pend and tstart != tend:
            return (None,
                    (fstart + party_start + tstart, fstart + party_start + tend))
        if pstart != pend and tstart == tend:
            return ((fstart + party_start + pstart, fstart + party_start + pend),
                    None)

        # find first non-title words
        # mat = re.search(r' [a-z]', party_st)  # no re.I here
        mat_with_start = find_first_non_title_word_mat(party_st)

        # print("mat_with_start = {}".format(mat_with_start))
        if mat_with_start:
            mat, mat_start = mat_with_start
            party_end = party_start + mat_start
            party_st = first_line[party_start:party_end]

            # print("bbbb232 party_st = [{}]".format(party_st))
            # sometimes the party_st might have address info, so remove
            # if possible
            org_suffix_mat_list = get_org_suffix_mat_list(party_st)

            # print("bbbb23333 org_suffix_mat_list = [{}]".format(org_suffix_mat_list))
            if org_suffix_mat_list:
                # found an org suffix, chop it off
                # "Business Marketing Services, Inc, One Broadway Street,", 37231.txt
                last_org_suffix_mat = org_suffix_mat_list[-1]
                party_end = party_start + last_org_suffix_mat.end()
                party_st = first_line[party_start:party_end]
                # print("final party_st: [{}]".format(party_st))
            # if not found org_suffix, keep the original
    elif first_line.startswith('('):
        pass
    else:
        split_party_term_offsets = split_party_term_effort_1(first_line)
        # should return correct offsets now
        (pstart, pend), (tstart, tend) = split_party_term_offsets
        if pstart != pend and tstart != tend:
            # xx_party_st = party_st[pstart:pend]
            # print("xx party_st: [{}]".format(xx_party_st))
            return ((fstart + party_start + pstart, fstart + party_start + pend),
                    (fstart + party_start + tstart, fstart + party_start + tend))
        if pstart == pend and tstart != tend:
            return (None,
                    (fstart + party_start + tstart, fstart + party_start + tend))
        if pstart != pend and tstart == tend:
            return ((fstart + party_start + pstart, fstart + party_start + pend),
                    None)

        # find first non-title words
        # mat = re.search(r' [a-z]', party_st)  # no re.I here
        mat_with_start = find_first_non_title_word_mat(first_line)
        # print("mat_with_start = {}".format(mat_with_start))
        if mat_with_start:
            party_start = 0
            mat, mat_start = mat_with_start
            party_end = party_start + mat_start
            party_st = first_line[party_start:party_end]

            # print('party_st = [{}]'.format(party_st))

            # sometimes the party_st might have address info, so remove
            # if possible
            org_suffix_mat_list = get_org_suffix_mat_list(party_st)
            if org_suffix_mat_list:
                # found an org suffix, chop it off
                # "Business Marketing Services, Inc, One Broadway Street,", 37231.txt
                last_org_suffix_mat = org_suffix_mat_list[-1]
                party_end = party_start + last_org_suffix_mat.end()
                party_st = first_line[party_start:party_end]
                # print("final party_st: [{}]".format(party_st))
            # if not found org_suffix, keep the original
        # if failed to find title words, mat is already None, do nothing

        """
        org_suffix_mat = ORG_SUFFIX_PAT.search(first_line)
        company_mat = A_COMPANY_PAT.search(first_line)
        if org_suffix_mat:
            party_start = 0
            party_end = party_start + org_suffix_mat.end()
            party_st = first_line[party_start:party_end]
            mat = org_suffix_mat  # value inside is ignore
        elif company_mat:
            party_start = 0
            party_end = party_start + company_mat.start()
            party_st = first_line[party_start:party_end]
            mat = company_mat  # value inside is ignore
        """

    # re.search(r'\(([^\(]+)\)\s*[\.;]?\s*(and|or)?\s*$', first_line)
    term_mat_list = list(re.finditer(r'\(([^\(]+)\)', first_line))
    term_mat = None
    if term_mat_list:
        term_mat = term_mat_list[-1]
        term_start = term_mat.start(1)
        term_end = term_mat.end(1)
        # term_st = term_mat.group(1)

    if mat and term_mat:
        return ((fstart + party_start, fstart + party_end),
                (fstart + term_start, fstart + term_end))
    if not mat and term_mat:
        return (None, (fstart + term_start, fstart + term_end))
    if mat and not term_mat:
        return ((fstart + party_start, fstart + party_end), None)

    return (None, None)

def is_one_party_line(line: str) -> bool:
    words = line.split()
    words10 = ' '.join(words[:10])
    # 38668.txt
    if re.search(r' (ltd|registered|incorporated|established|hereinafter|(having|with).*address|'
                 r'a .*(corporation|company))\b',
                 words10, re.I):
        return True
    return False

def is_one_party_line_no_other(line: str) -> bool:
    if len(line) > 100:
        return False
    # if there is verb, then it is NOT one_party_line_no_other
    if re.search(r'\b(are|registered)\b', line, re.I):
        return False
    if re.search(r'\s*[“"”][^“"”]+[“"”]\s*:?\s*', line):
        return True
    if ORG_SUFFIX_PAT.search(line):
        return True

    if is_likely_person_name(line):
        return True
    return False


def is_likely_person_name(line: str) -> bool:
    """Return if line is likely a person name.

    This can be make more robust by using name DB or ML
    """
    words = line.split()
    if len(words) <= 4:
        for word in words:
            # abbreviation
            if re.search(r'^[a-z]\.?$', word, re.I):
                return True
    return False

# pylint: disable=line-too-long
def extract_parties_from_list_lines(se_after_paras_attr_list: List[Tuple[int, int, str, List[str]]]) \
                                    -> List[Tuple[Optional[Tuple[int, int]],
                                                  Optional[Tuple[int, int]]]]:
    result = []
    count_other_line = 0
    is_last_char_lower = False

    # To capture lines for each party.  Sometime lines in a group can be broken for
    # whatever reason (change of font, too much spaces between lines, etc).
    cur_party_group = []  # type: List[Tuple[int, int, str, List[str]]]
    party_line_group_list = []  # type: List[List[Tuple[int, int, str, List[str]]]]

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

    is_prev_line_all_caps = False
    is_list_party_mode = False
    is_one_line_party_mode = False
    is_one_line_party_no_other_mode = False
    for i in range(min(100, len(se_after_paras_attr_list))):
        se_line_attrs = se_after_paras_attr_list[i]
        _, _, linex, attr_list = se_line_attrs
        if linex:

            if is_list_prefix(linex):
                if IS_DEBUG_MODE:
                    print("\nextract_parties_from_list_lines")
                    print("is_list_prefix: [{}]".format(linex))
                if cur_party_group:
                    party_line_group_list.append(cur_party_group)
                    cur_party_group = []
                cur_party_group.append(se_line_attrs)
                # print("\n     count_other_line: {}".format(count_other_line))
                # print("     after_para: {}".format(se_after_paras_attr_list[i]))
                count_other_line = 0
                is_list_party_mode = True
            # one_line_party_mode and one_line_party_no_other_mode doesn't mix
            elif not is_one_line_party_mode and not is_list_party_mode and is_one_party_line_no_other(linex):
                if IS_DEBUG_MODE:
                    print("\nextract_parties_from_list_lines")
                    print("is_one_party_line_no_other: [{}]".format(linex))
                is_one_line_party_no_other_mode = True
                if cur_party_group:
                    party_line_group_list.append(cur_party_group)
                    cur_party_group = []
                cur_party_group.append(se_line_attrs)
                # print("\n     count_other_line: {}".format(count_other_line))
                # print("     after_para: {}".format(se_after_paras_attr_list[i]))
                count_other_line = 0
            # TODO, jshaw, maybe should check for 'is_one_line_party_no_other_mode'?
            elif not is_list_party_mode and is_one_party_line(linex):
                if IS_DEBUG_MODE:
                    print("\nextract_parties_from_list_lines")
                    print("is_one_party_line: [{}]".format(linex))
                is_one_line_party_mode = True
                if cur_party_group:
                    party_line_group_list.append(cur_party_group)
                    cur_party_group = []
                cur_party_group.append(se_line_attrs)
                # print("\n     count_other_line: {}".format(count_other_line))
                # print("     after_para: {}".format(se_after_paras_attr_list[i]))
                count_other_line = 0
            elif is_end_party_list(linex, attr_list):
                # print("break end_party_list: {}".format(linex))
                break
            elif is_last_char_lower:
                # print("skipping last char lower: {}".format(linex))
                cur_party_group.append(se_line_attrs)
            elif is_prev_line_all_caps:
                # print("skipping prev_line_all_caps: {}".format(linex))
                cur_party_group.append(se_line_attrs)
            else:
                # print("count other line: {}".format(linex))
                count_other_line += 1

            if count_other_line >= 3:
                # print("break count_other_line >= 3")
                break
            is_last_char_lower = linex[-1].islower()
            is_prev_line_all_caps = is_all_english_title_case(linex)

    if cur_party_group:
        party_line_group_list.append(cur_party_group)


    if is_one_line_party_no_other_mode:
        # 37320.txt
        # “SAFEDOX”:
        # “PURCHASER”:
        # SafedoX, Inc.
        # New Beginnings Life Center, LLC
        result.extend(tabled_party_line_group_to_party_terms(party_line_group_list))
    else:
        # now process each group into a party
        for i, party_line_group in enumerate(party_line_group_list):
            # print('party group #{}:'.format(i))
            # for se_line_attrs in party_line_group:
            #     print('253     {}'.format(se_line_attrs))
            result.append(party_line_group_to_party_term(party_line_group))

    if IS_DEBUG_MODE:
        print()
        for offsets_pair in result:
            print("offsets_pair: {}".format(offsets_pair))
        print()

    return result

# bool(re.search,(r'(:|among|between)\s*$', line_st))
def is_list_party_line(line: str) -> bool:
    # any party_line ends with ':' is considered a list party prefix
    org_suffix_list = get_org_suffix_mat_list(line)
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
                    unused_para_text: str) \
    -> List[Tuple[Tuple[int, int],
                  Optional[Tuple[int, int]]]]:
    """Return list of parties (lists of (start, inclusive-end) offsets)."""

    out_list = []  # type: List[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]]

    # Grab lines from the file
    pline_after_lines = extract_party_line(paras_attr_list)
    if pline_after_lines:
        start_end_partyline, is_list_party, after_se_paras_attr_list = pline_after_lines
        start, unused_end, party_line = start_end_partyline

        if IS_DEBUG_MODE:
            print('\nparty_line: (%d, %d)' % (start, unused_end))
            print(party_line)
            print("is_list_party = {}".format(is_list_party))

        if is_list_party:
            # all the parties are in after_se_paras_attr_list
            party_term_offsets_list = extract_parties_from_list_lines(after_se_paras_attr_list)
            for party_term_offsets in party_term_offsets_list:
                party_offset_pair, term_offset_pair = party_term_offsets
                if party_offset_pair and term_offset_pair:
                    # print("xxx {},,,, {}".format(party_offset_pair, term_offset_pair))
                    out_list.append((party_offset_pair, term_offset_pair))
                if party_offset_pair and not term_offset_pair:
                    # print("xxx111 {}".format(party_offset_pair))
                    out_list.append((party_offset_pair, None))
                if not party_offset_pair and term_offset_pair:
                    # print("found defined_term, but not party: {}".format(term_offset_pair))
                    # print("xxx222 {}".format(term_offset_pair))
                    out_list.append((term_offset_pair, None))
        else:  # normal party line
            """
            # Extract parties and return their offsets
            parties = extract_parties_from_party_line(party_line)

            # for ppart in parties:
            #     print("ppart: {}".format(ppart))
            offset_pair_list = parties_to_offsets(parties, party_line)
            # for ppart in offset_pair_list:
            #     print("ppart 22: {}".format(ppart))
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
            """

            # there still can be multiple sentences in the paragraph containing
            # the party line.  Use NLTK to take care of this.
            """
            sent_tokenize_list = sent_tokenize(party_line)
            if IS_DEBUG_MODE:
                for sent in sent_tokenize_list:
                    print("found sent: {}".format(sent))
            party_line = sent_tokenize_list[0]
            """

            party_dterm_list = extract_party_defined_term_list(party_line)
            for party_dterm in party_dterm_list:
                party_x, dterm_x = party_dterm
                if party_x and dterm_x:
                    pstart, pend, _, _ = party_x
                    tstart, tend, _, _ = dterm_x
                    out_list.append(((start + pstart, start + pend),
                                     (start + tstart, start + tend)))
                elif party_x:
                    pstart, pend, _, _ = party_x
                    out_list.append(((start + pstart, start + pend), None))
                elif dterm_x:
                    tstart, tend, _, _ = dterm_x
                    out_list.append(((start + tstart, start + tend), None))

    if IS_DEBUG_MODE:
        print()
        for i, (party_y, term_y) in enumerate(out_list):
            if party_y:
                start, end = party_y
                print("  #{} found party: [{}]".format(i, unused_para_text[start:end]))
            if term_y:
                start, end = term_y
                print("  #{} found dterm: [{}]".format(i, unused_para_text[start:end]))
        print()

    filtered_out_list = []
    # filter bad party or terms right before returning the result
    for party_y, term_y in out_list:
        if party_y:
            start, end = party_y
            party_y_text = unused_para_text[start:end]
            if re.match(r'^(the|each)$', party_y_text, re.I):
                # party_y = None
                # term_y = None
                # skip the append part
                continue
            # there is no bad parties for now
        if term_y:
            start, end = term_y
            term_y_text = unused_para_text[start:end]
            if re.search(r'\b(as\s+defined\s+below)\b', term_y_text, re.I):
                term_y = None
        filtered_out_list.append((party_y, term_y))

    return filtered_out_list


# pylint: disable=too-few-public-methods
class PartyAnnotator:

    # pylint: disable=unused-argument
    def __init__(self, provision: str) -> None:
        self.provision = 'party'

    # pylint: disable=no-self-use
    def extract_provision_offsets(self,
                                  paras_with_attrs: List[Tuple[str, List[str]]],
                                  paras_text: str) \
        -> List[Tuple[Tuple[int, int],
                      Optional[Tuple[int, int]]]]:
        return extract_offsets(paras_with_attrs, paras_text)
