from kirke.ebrules import titles
from itertools import groupby
import re
import string
import logging


"""Config"""


DATA_DIR = './dict/parties/'
PARTY_CANT_HAVE = [r'\bon the \w+ hand\b']
TERM_CANT_HAVE = [r'\bdate\b', r'\boffice\b', r'\bdefined\b']


"""Get keywords"""


dot_space = re.compile(r'[\.\s]+')


def cap_rm_dot_space(s):
    """Allows comparison between different cases (e.g. L.L.C. and Llc)."""
    return dot_space.sub('', s.strip().upper())


with open(DATA_DIR + 'states.list') as f:
    states = [cap_rm_dot_space(state) for state in f.read().splitlines()]
with open(DATA_DIR + 'business_suffixes.list') as f:
    business_suffixes = f.read().splitlines()
with open(DATA_DIR + 'name_suffixes.list') as f:
    name_suffixes = f.read().splitlines()
suffixes = business_suffixes + name_suffixes
max_suffix_words = max(len(s.split()) for s in suffixes)
business_suffixes = [cap_rm_dot_space(s) for s in business_suffixes]
suffixes = business_suffixes + [cap_rm_dot_space(s) for s in name_suffixes]


"""Process parts of strings (already split by comma, and, & semicolon)"""


def keep(s):
    """Eliminate titles (the "Agreement") as potential parties and terms."""
    alphanum_chars = ''.join([c for c in s if c.isalnum()])
    return titles.title_ratio(alphanum_chars) < 73 if alphanum_chars else False


party_regexes = [re.compile(p, re.IGNORECASE) for p in PARTY_CANT_HAVE]


def process_part(p):
    # Reject if just a state or state abbreviations (only 2 characters)
    if len(p) < 3 or cap_rm_dot_space(p) in states:
        return ''

    # Terminate at a single-word business suffix or business suffix abbreviation
    for word in p.split():
        if cap_rm_dot_space(word) in business_suffixes:
            p = p[:p.index(word) + len(word)]
            break

    # Remove certain phrases like 'on the one hand'
    for r in party_regexes:
        p = r.sub('', p)

    # Take away any lowercase words from end e.g. 'organized'
    while p.strip() and p.split()[-1].islower():
        p = p[:len(p) - len(p.split()[-1]) - 1]

    # Return the processed part if not a title
    return p if keep(p) else ''


"""Process terms"""


term_regexes = [re.compile(t, re.IGNORECASE) for t in TERM_CANT_HAVE]


def process_term(t):
    """Ignore if term is a title or contains words that a term cannot have."""
    return t if keep(t) and not any(r.search(t) for r in term_regexes) else ''


"""Helper functions and regexes for extracting parties from party line"""


party_chars = set(string.ascii_letters).union(string.digits).union('=.')


def party_strip(s):
    non_party_chars = set(s) - party_chars
    return s.strip(str(non_party_chars)) if non_party_chars else s


sentence_does_not_continue = r'(?=\s+(?:[A-Z0-9].|[a-z][A-Z0-9]))'
not_few_letters = r'(?<!\b[A-Za-z]\.)(?<!\b[A-Za-z]{2}\.)'
not_number = r'(?<!\bN(O|o)\.)(?<!\bN(O|o)(S|s)\.)'
real_period = r'\.' + sentence_does_not_continue + not_few_letters + not_number
first_sent = re.compile(r'(.*?' + real_period + ')')


def first_sentence(s):
    """Trying to avoid sentence tokenizing since occurs before CoreNLP."""
    match = first_sent.search(s)
    return match.group() if match else s


parens = re.compile(r'\(.*?\)')
non_comma_separators = re.compile(r',\sand|\sand|;')
paren_symbol = re.compile(r'(=)')
zip_code_year = re.compile(r'\b\d{5}(?:\-\d{4})?\b|\b(?:19|20)\d{2}\b')
quote = re.compile(r'“|"|”')


"""Extract parties from party line"""


def extract_between_among(s):
    """Return parties for party lines containing either 'between' or 'among'."""

    # Consider after between. If line ends in between (no 'and'), return None.
    s = s.split('between')[-1].split('among')[-1]
    if 'and' not in s:
        return None

    # Temporarily sub defined terms with '=' to avoid splitting on their commas
    terms = parens.findall(s)
    s = non_comma_separators.sub(',', parens.sub('=', s))

    # Split the string into parts, applying party_strip between each step
    parts = [party_strip(part) for part in party_strip(s).split(', ')]
    parts = [party_strip(q) for p in parts for q in paren_symbol.split(p) if q]
    parts = [q for q in parts if q]

    # Process parts and decide which parts to keep
    new_parts = ['']
    for p in parts:
        # If p is a term, keep the term and continue
        if p == '=':
            new_parts.append(p)
            continue

        # If first word is a suffix (MD MBA), add to previous and remove from p
        seen_suffixes = ''
        check_again = True
        while check_again:
            check_again = False
            words = p.strip().split()
            for i in range(max_suffix_words):
                words_considered = cap_rm_dot_space(''.join(words[:i]))
                if any(words_considered == s for s in suffixes):
                    seen_suffixes += ' ' + ' '.join(words[:i])
                    p = ' '.join(words[i:])
                    check_again = True
                    break
        if seen_suffixes:
            # Append suffixes
            new_parts[-1] += ',' + seen_suffixes

        # Take out 'the ' from beginning of string; bail if remaining is empty
        while p.startswith('the '):
            p = p[4:]
        if not p.strip():
            continue

        # Mark for deletion if first word has no uppercase letters or digits
        first_word = p.split()[0]
        if not any(c.isupper() or c.isdigit() for c in first_word):
            new_parts.append('^')
            continue

        # If zip code or year in part and not already deleting ('^'), mark '+'
        if zip_code_year.search(p):
            new_parts.append('+')
            continue

        # Process then keep the part if not a title, etc.
        processed_part = process_part(p)
        if processed_part:
            new_parts.append(processed_part)

    # Remove lines marked for deletion (^)
    parts = new_parts if new_parts[0] else new_parts[1:]
    grps = [list(g) for k, g in groupby(parts, lambda p: '^' in p) if not k]

    # Going backwards, when see a zip code/ year, remove up to prev removed line
    for i in range(len(grps)):
        zip_code_inds = [j for j, part in enumerate(grps[i]) if part == '+']
        if zip_code_inds:
            grps[i] = grps[i][max(zip_code_inds) + 1:]
    parts = [part for g in grps for part in g]

    # Add terms back in
    terms = [process_term(t) for t in terms]
    current_term = 0
    parties = []
    part_types = []
    part_type_bools = []
    for p in parts:
        # Record term type {0: party, 1: (), 2: ("")} and substitute term
        part_type_bool = p == '='
        part_type = int(part_type_bool)
        if part_type_bool:
            p = terms[current_term]
            current_term += 1
            if quote.search(p):
                part_type = 2
        if not p:
            # Occurs e.g. if term was a title or a date
            continue

        # Append if first party/term or term follows term (if 2, no other 2)
        if part_types:
            if (part_type_bool not in part_type_bools
                or part_types[-1] and (part_type == 1
                                       or (part_type == 2
                                           and 2 not in part_types))):
                parties[-1].append(p)
                part_types.append(part_type)
                part_type_bools.append(part_type_bool)
                continue

        # Otherwise start a new party
        parties.append([p])
        part_types = [part_type]
        part_type_bools = [part_type_bool]

    # Remove parties that only contain defined terms, then return
    parties = [p for p in parties if not all(parens.search(w) for w in p)]
    return parties


def extract_parties_from_party_line(s):
    """Return list of parties (which are lists of strings) of s (party line)."""
    s = first_sentence(s)

    # Try (eventually several) possible rules
    if 'between' in s or 'among' in s:
        return extract_between_among(s)

    # Fall back to NER (not implemented) if none of the above criteria were met
    return None


"""Extract parties from debug file"""


def parties_to_offsets(parties, party_line):
    """Converts string parts of parties to offsets (relative to party line)."""
    if parties:
        for i in range(len(parties)):
            offsets = []
            for part in parties[i]:
                start_index = party_line.find(part)
                if start_index != -1:
                    offsets.append((start_index, start_index + len(part)))
            parties[i] = offsets
        return [p for p in parties if p]
    return []


def extract_parties(filepath):
    """Return list of parties (lists of (start, inclusive-end) offsets)."""

    # Find the party line in the file
    party_line = None
    with open(filepath) as f:
        for line in f:
            tags = line.split('\t')[0].split('|')
            if 'party_line' in tags:
                after_first_bracket = ''.join(line.split('[')[1:])
                between_brackets = ''.join(after_first_bracket.split(']')[:-1])
                party_line = between_brackets
                break

    # Return None if no party_line was found
    if not party_line:
        return None

    # Extract parties and return their offsets
    parties = extract_parties_from_party_line(party_line)
    return parties_to_offsets(parties, party_line)


def extract_party_line(paras_attr_list):
    lines = []
    offset = 0
    start_end_list = []
    for i, (line_st, para_attrs) in enumerate(paras_attr_list):
        # attrs_st = '|'.join([str(attr) for attr in para_attrs])
        # print('\t'.join([attrs_st, '[{}]'.format(line_st)]), file=fout1)
        line_st_len = len(line_st)

        if 'party_line' in para_attrs:
            return offset, offset + line_st_len, line_st
        offset += line_st_len + 1

        # don't bother if party_line is too far from start of the doc
        if i > 2000:  
            return None
        
    return None


# paras_text is not used for title right now
def extract_offsets(paras_attr_list, para_text):
    """Return list of parties (lists of (start, inclusive-end) offsets)."""

    # Grab lines from the file
    start_end_partyline = extract_party_line(paras_attr_list)

    # Return None if no party_line was found
    if not start_end_partyline:
        return None

    start, end, party_line = start_end_partyline
    print("party_line ({}, {})".format(start, end))
    print("[{}]".format(party_line)) 
    
    # Extract parties and return their offsets
    parties = extract_parties_from_party_line(party_line)
    offset_pair_list = parties_to_offsets(parties, party_line)
    out_list = []
    logging.info("offset_pair_list: {}".format(offset_pair_list))
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
        
    logging.info("out_list: {}".format(out_list))
    return out_list


class PartyAnnotator:

    def __init__(self, provision):
        self.provision = 'party'

    def extract_provision_offsets(self, paras_with_attrs, paras_text):
        return extract_offsets(paras_with_attrs, paras_text)
        