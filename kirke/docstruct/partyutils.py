#!/usr/bin/env python3

import argparse
import logging
import re

from typing import List, Match, Optional, Tuple

from kirke.utils import engutils, regexutils, strutils


IS_DEBUG_MODE = False

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')

ST_PAT_LIST = ['is made and entered into by and between',
               'is made and entered into',
               'is entered into between',
               'is entered into by and among',
               'is entered into among',
               'is entered into',
               'by and between',
               'by and among',
               'is by and among',
               'by and between',
               'is among',
               # 'between',
               'confirm their agreement',
               'each confirms its agreement',
               'confirms its agreement',
               'the parties to this',
               'promises to pay',
               'to the order of',
               'promises to pay to']

PARTY_PAT = re.compile(r'\b({})\b'.format('|'.join(ST_PAT_LIST)), re.IGNORECASE)

MADE_BY_PAT = re.compile(r'\bmade(.*)by\b', re.IGNORECASE)
CONCLUDED_PAT = re.compile(r'\bhave concluded.*agreement\b', re.IGNORECASE)

THIS_AGREEMENT_PAT = re.compile(r'this.*agreement\b', re.IGNORECASE)

REGISTERED_PAT = re.compile(r'\bregistered\b', re.I)

def is_made_by_check(line: str) -> bool:
    mat = MADE_BY_PAT.search(line)
    return mat and len(mat.group(1)) < 20


# bank is after 'n.a.' because 'bank, n.a.' is more desirable
# 'Credit Suisse Ag, New York Branch', 39893.txt,  why 'branch' is early
# TODO, handle "The bank of Nova Scotia", this is NOT org suffix case
# TODO, not handling 'real estate holdings fiv'
# TODO, remove 'AS AMENDED' as a party, 'the customer'?
# TODO, 'seller, seller' the context?
ORG_SUFFIX_LIST = strutils.load_non_empty_str_list('dict/parties/organization.suffix.list')
PERS_SUFFIX_LIST = strutils.load_non_empty_str_list('dict/parties/person.suffix.list')

ORG_PERSON_SUFFIX_LIST = list(ORG_SUFFIX_LIST)
ORG_PERSON_SUFFIX_LIST.extend(PERS_SUFFIX_LIST)

# copied from kirke/ebrules/parties.py on 2/4/2016
ORG_PERSON_SUFFIX_PAT = regexutils.phrases_to_igs_pattern(ORG_PERSON_SUFFIX_LIST, re.I)
ORG_PERSON_SUFFIX_END_PAT = \
    re.compile(regexutils.phrases_to_igs_pattern_st(ORG_PERSON_SUFFIX_LIST) + r'\s*$', re.I)

# print("org_person_suffix_pattern_st:")
# print(regexutils.phrases_to_igs_pattern_st(ORG_PERSON_SUFFIX_LIST))

def is_org_suffix(line: str) -> bool:
    # print("is_org_suffix({})".format(line))
    return bool(ORG_PERSON_SUFFIX_END_PAT.match(line))


def get_org_suffix_mat_list(line: str) -> List[Match[str]]:
    """Get all org suffix matching mat extracted from line.

    Because of capitalization concerns, we are making
    a pass to make sure, it is not just 'a limited company'
    """

    lc_mat_list = list(ORG_PERSON_SUFFIX_PAT.finditer(line))
    result = []  # type: List[Match[str]]
    for lc_mat in lc_mat_list:
        prev_space_idx = lc_mat.start() -1
        # the previous word must be capitalized
        pword_start, pword_end, pword = strutils.find_previous_word(line, prev_space_idx)
        if pword_start != -1:
            if pword[0].isupper():
                result.append(lc_mat)

    # when there is adjcent ones, take the last one
    # 'xxx Group, Ltd.', will return 'ltd'
    prev_mat = None
    result2 = [] # type: List[Match[str]]
    # Only if we now that the current mat is not adjacent to
    # the previous mat, we can add previous mat.
    # Remember the last one.
    for amat in result:
        # 2 is chosen, just in case, normally the diff is 1
        if prev_mat and amat.start() - prev_mat.end() > 2:
            result2.append(prev_mat)
        prev_mat = amat
    if prev_mat:
        result2.append(prev_mat)
    return result2


def find_uppercase_party_name(line: str) \
    -> Optional[Tuple[Tuple[int, int], int]]:
    found_party_se_other = find_first_non_title_and_org(line)
    if found_party_se_other:
        (party_start, party_end), other_start = found_party_se_other
        party_st = line[party_start:party_end]
        # sometimes the party_st might have address info, so remove
        # if possible
        org_suffix_mat_list = get_org_suffix_mat_list(party_st)
        if org_suffix_mat_list:
            # found an org suffix, chop it off
            # "Business Marketing Services, Inc, One Broadway Street,", 37231.txt
            # last_org_suffix_mat = org_suffix_mat_list[-1]
            # party_end = party_start + last_org_suffix_mat.end()

            # find a org suffix that's less than 40
            org_suffix_mat_list_less_40 = [mat for mat in org_suffix_mat_list
                                           if mat.start() < 40]

            if org_suffix_mat_list_less_40:
                last_org_suffix_mat = org_suffix_mat_list_less_40[-1]
                party_end = party_start + last_org_suffix_mat.end()
                other_start = strutils.find_next_not_space_idx(line, party_end)
            else:
                # cannot find the org suffix in the first 40 chars
                first_org_suffix_mat = org_suffix_mat_list[0]
                party_end = party_start + first_org_suffix_mat.end()
                other_start = strutils.find_next_not_space_idx(line, party_end)
            return (party_start, party_end), other_start
        else:
            # 'Johnson & Johnson', without corp suffix, or a person's name
            return (party_start, party_end), other_start

    return None


def find_first_non_title_and_org(line: str) -> Optional[Tuple[Tuple[int, int], int]]:
    """Find the first non-title and non-org word.

    The string might have "and", such as "Johnson and Johnson Inc", or
    has digit, "Apartment 3 corp".  Needs to jump to the end of both.

    Return the start, end of company name, followed by the start of the rest of the line
    """
    prev_end = -1
    maybe_se_other_start = None
    other_word_idx, other_word = -1, ''
    se_word_list = list(strutils.nltk_span_tokenize(line))
    for i, (start, end, word) in enumerate(se_word_list):
        if word.islower() and not is_org_suffix(word):
            maybe_se_other_start = 0, prev_end, start
            other_word_idx, other_word = i, word
            break
        # if this is an abbreviation with a period, we will
        # take the period
        if len(word) == 1 and \
           end < len(line) and \
           line[end] == '.':
            prev_end = end + 1
        else:
            prev_end = end

    # cannot find begin title
    if prev_end == -1:
        return None
    elif not maybe_se_other_start:  # the whole line has istitle()
        return (0, prev_end), prev_end

    fx_start, fx_end, other_start = maybe_se_other_start

    after_line = line[other_start:]
    if IS_DEBUG_MODE:
        print('after_line = [{}]'.format(after_line))

    if re.match(r'\band\b', other_word, re.I) or strutils.is_digits(other_word):
        if other_word_idx + 1 < len(se_word_list):
            other_start = se_word_list[other_word_idx+1][0]  # the start of the first word after 'and'

            prev_end = se_word_list[other_word_idx][1]  # the end of the 'and'
            for sc_start, sc_end, word in se_word_list[other_word_idx+1:]:
                if word.islower() and not is_org_suffix(word):
                    return (0, prev_end), sc_start
                prev_end = sc_end

            # reaching here means the whole line is istitle()
            return (0, prev_end), prev_end

        else:  # there is no more words, return everything befefore 'and'
            return (0, prev_end), prev_end
    # if want to handle "Citibank, n.a.", can do it here
    # by regex matching
    elif ORG_PERSON_SUFFIX_PAT.match(after_line):
        # do matching again, this will be rare, tolerate the cost
        mat = ORG_PERSON_SUFFIX_PAT.match(after_line)
        prev_end = other_start + mat.end()
        other_start = strutils.find_next_not_space_idx(line, prev_end+1)

    return (0, prev_end), other_start



# all those heuristics didn't work.
# they eliminated too many tp
# line_notoc_empty > 100
# num_sechead > 60
# num_date > 10
def is_party_line(line: str,
                  num_long_english_line: int = -1) -> bool:

    # print("is_party_line({})".format(line))
    # print("  ln_nempty_toc = {}, eng = {}, num_sechead = {}, num_date = {}"\
    #       .format(line_notoc_empty,
    #               num_long_english_line,
    #               num_sechead,
    #               num_date))
    if num_long_english_line > 10:
        return False

    result = is_party_line_aux(line)

    if IS_DEBUG_MODE:
        print('branch {}, line = [{}]'.format(result, line))

    # do some extra verfication
    if result.startswith('T'):   # result:
        mat = re.match(r'\(?(\S)\)', line)
        if mat:
            # a party line must starts with 1) a) or i) 'l' is for bad OCR 1's
            if mat.group(1) in '1ailA':
                return True
            return False

    if result.startswith('T'):
        return True

    return False


# pylint: disable=too-many-return-statements, too-many-branches
# for debug purpose, return str of 'True\d', or 'False\d'
# pylint: disable=too-many-statements
def is_party_line_aux(line: str) -> str:

    # this is not a party line due to the words used
    # adding this will decrease f1 by 0.001.  Will figure out later.
    # if re.search(r'\bif\b', line, re.I) and re.search(r'\bwithout\b', line, re.I):
    #    return False

    if re.match(r'this\s+agreement\s+is\s+dated\b', line, re.I):
        return 'True0.1'

    if len(line) > 5000:  # sometime the whole doc is a line
        return 'False1'

    if re.search(r'\b(engages?|made\s+available|subordinate\s+to|all\s+liens)\b', line, re.I):
        return 'False2'

    if '.....' in line:
        return 'False2.2'
    # 2/7/2018
    # only impacted 3 files, but negatively on F1
    # if re.search(r'\b(i\s+confirm|signing|i\s+acknowledge|following\s+(meaning|definition)s?)\b',
    #              line, re.I):
    #    return 'False3'

    # 2/6/2018, uk/file3.txt, multiple parties got mentioned and registered, but not a party line
    alpha_words = strutils.get_alpha_words(line, is_lower=False)
    is_all_upper_words = strutils.is_all_upper_words(alpha_words)
    # this is match, not search, uk/file3.txt
    if re.match(r'\d+\-\d+', line) and is_all_upper_words:
        return 'False4'

    # this is a title
    if is_all_upper_words and \
       len(alpha_words) < 20 and \
       alpha_words[0] != 'THIS':
       # (line[-1] in set(['.', ':'])
        return 'False4.1'

    # PROMISSORY NOTE ... IS SUBJECT TO XXX AGREEMENT
    if is_all_upper_words and re.search(r'is\s+subject\s+to', line, re.I):
        return 'False4.2'

    if re.search(r'terms?\s+and\s+conditions?', line, re.I):
        return 'False4.3'

    if re.search(r'should have', line, re.I):
        return 'False4.3.1'

    #returns true if bullet type, and a real line
    # if re.match(r'\(?[\div]+\)', line) and len(line) > 60:
    # this has too many False positives
    # mat = re.match(r'\(?\s*(1|a|i|l)\s*\)\s*(.*)', line, re.I)
    # if mat and len(line) > 60:
        # TODO, jshaw, 36820.txt  Rediculous way of formatting
        # need to pass line number in to disable this aggressive matching
        # will fix later.  Not happening in PDF docs?

        # print("I am hereeeeeeeee")
        # suffix_st = mat.group(2)
        # suffix_mat = re.match(r'\s*party \(?(.*)\b', suffix_st, re.I)
        # if not suffix_mat:
        #     return 'True1'
        # if suffix_mat and \
        #   not (suffix_mat.group(1).startswith('A') or
        #        suffix_mat.group(1).startswith('1')):
        #    return 'False1'
    #    return 'True5'

    # Party A: xxx,
    # Party B:
    if re.match(r'Party \S+\s*:', line, re.I) and line[0].isupper():
        return 'True6'

    num_org_suffix = len(get_org_suffix_mat_list(line))
    if 'among' in line and ' dated ' in line and num_org_suffix > 2:
        return 'True7'

    # this is from a title line, not a party line
    if len(line) < 200 and ORG_PERSON_SUFFIX_END_PAT.search(line) and line.strip()[-1] != '.':
        return 'False8'

    # Removed.  This turns out to be false for UK document multiple times.
    # multipled parties mentioned
    # if len(list(REGISTERED_PAT.finditer(line))) > 1:
    #    if strutils.is_all_upper_words(alpha_words):  #
    #        return 'False1'
    #    return 'True1'


    # 44139.txt, info is attached, yada yada
    # '\$\d' doesn't work, decrease F1 by 20%!  Too many
    # promissory or loan notes has partyline with '$'
    # if re.search(r'\b(is attached|partial)\b', line, re.I):
    #    return 'False1'

    # This is NOT true.  There are agreements that this is not true.
    # 39761.txt.  Around 2% lower.
    # # 'agreement, dated may 24, 2004', is NOT a party line
    # if re.search(r'\bdated\b', line, re.I) and \
    #    not re.search(r'\bis\s+dated\b', line, re.I):
    #    return 'False1'

    if re.search(r'\b(entered)\b', line, re.I) and \
       re.search(r'\b(by\s+and\s+between)\b', line, re.I):
        return 'True9'

    if re.search(r'\b(agreement|contract)\b', line, re.I) and \
       re.search(r'\b(entered\s+into)\b', line, re.I):
        return 'True9.1'

    # TODO, jshaw, look into this
    # [tn=0, fp=1347], [fn=2877, tp=8034]], f1=0.7918
    # => [[tn=0, fp=1335], [fn=2877, tp=8034]] f1= 0.7923
    # so remove this line reduces false positives.
    if re.search(r'\b(hereby\s+enter(ed)?\s+into)\b', line, re.I):
        return 'True10'

    # added on 02/06/2018, jshaw
    if re.search(r'way\s+of\s+deed', line, re.I):
        return 'True11'
    # 'deed of release is made"
    if re.search(r'\b(deed\s+is\s+made|deed.*is\s+made)\b', line, re.I):
        return 'True12'
    # this is slight aggressive
    if re.search(r'^This.*(deed|guarantee).*dated\b', line, re.I):
        return 'True13'

    # uk doc, file2
    # This Agreement is made on 2017
    #        Between:
    # (1) ...
    if re.search(r'agreement\s+is\s+made\s+on', line, re.I):
        return 'True14'

    if line.startswith('T') and \
       re.match('(this|the).*contract.*is made on', line, re.I):
        return 'True15'
    if len(line) < 40:  # don't want to match line "BY AND BETWEEN" in title page
        return 'False16'
    if engutils.is_skip_template_line(line):
        return 'False17'
    if 'means' in line:  # in definition section of 'purchase agreement'
        return 'False18'
    mat = PARTY_PAT.search(line)
    if mat:
        return 'True8.8'  # bool(mat)
    lc_line = line.lower()
    if 'between' in lc_line and engutils.has_date(lc_line):
        return 'True19'
    if 'made' in lc_line and engutils.has_date(lc_line) and 'agreement' in lc_line:
        return 'True20'
    if 'issued' in lc_line and engutils.has_date(lc_line) and 'agreement' in lc_line:
        return 'True21'
    if 'entered' in lc_line and engutils.has_date(lc_line) and 'agreement' in lc_line:
        return 'True22'
    # power of attorney
    if 'made on' in lc_line and engutils.has_date(lc_line) and 'power' in lc_line:
        return 'True23'
    if 'between' in lc_line and 'agreement' in lc_line:
        return 'True24'
    # assigns lease to
    if 'assign' in lc_line and 'lease to' in lc_line:
        return 'True25'
    if is_made_by_check(line) and ('day' in lc_line or
                                   'date' in lc_line):
        return 'True26'
    if CONCLUDED_PAT.search(line):
        return 'True27'

    if THIS_AGREEMENT_PAT.search(line) and "amendment" in lc_line:
        return 'True28'

    # termination agreement
    if 'agree that' in lc_line and 'employment at' in lc_line:
        return 'True29'
    # 'Patent Security Agreement, dated as of December 1, 2009, by BUSCH
    # ENTERTAINMENT LLC (the “Grantor”), in favor of BANK OF AMERICA, N.A...'
    if 'date' in lc_line and ' by ' in lc_line and 'in favor of' in lc_line:
        return 'True30'
    if 'reach an agreement' in lc_line or \
       'the following terms' in lc_line or \
       'terms and condistions' in lc_line or \
       'enter into this contract' in lc_line:
        return 'True31'
    if 'hereinafter' in lc_line and 'agree' in lc_line:
        return 'True32'
    if 'confirm' in lc_line and 'agree' in lc_line:
        return 'True33'
    #"""In this Agreement (unless the context requires otherwise) the following words
    # shall have the following meanings"""
    if 'agree' in lc_line and 'follow' in lc_line and not "meaning" in lc_line:
        return 'True34'
    if 'follow' in lc_line and 'between' in lc_line:
        return 'True35'
    # for warrants, 'the Lenders from time to time party thereto,'
    if 'from time to time party thereto' in lc_line:
        return 'True36'

    #"""This Amendment No. 1 to the Convertible Promissory Note (this
    #   "Amendment") is executed as of October 17, 2011, by SOLAR ENERGY
    #   INITIATIVES, INC., a Nevada corporation (the “Maker”); and ASHER
    #   ENTERPRISES, INC., a Delaware corporatio"""
    if 'is executed' in lc_line and \
       'by' in lc_line and \
       'and' in lc_line:
        return 'True37'
    # agreement is made to ..., dated as of ... among
    if 'agreement' in lc_line and \
       'dated' in lc_line and \
       'among' in lc_line:
        return 'True38'
    # 'this certifies that, ... is entitled to,
    if 'is entitled to' in lc_line and \
       'certifies' in lc_line and \
       'purchase' in lc_line:
        return 'True39'
    if 'is made' in lc_line and \
       'following parties' in lc_line:
        return 'True40'
    return 'False41'


def find_first_party_lead(line):
    lc_line = line.lower()
    for st_pat in ST_PAT_LIST:
        idx = lc_line.find(st_pat)
        if idx != -1:
            return idx + len(st_pat)
    return -1

def find_party_separator(line):
    pat = re.compile(r'each lender party hereto \([^\)]+\),', re.IGNORECASE)
    return re.search(pat, line)

def find_party_separator2(line):
    pat = re.compile(r'(?<=\)) and ', re.IGNORECASE)
    return re.search(pat, line)

def find_and(line):
    pat = re.compile(r'\band\b', re.IGNORECASE)
    return re.search(pat, line)

def find_a_corp(line):
    pat = re.compile(r'[A-Z\.\, ]+ (AG|LTD|N\.\s*A|Limited|LIMITED)\.?, '
                     r'(a|as)\b[^\(]+\([^\)]+\)[\.,]?')
    return re.search(pat, line)


def extract_parties(line):
    index = find_first_party_lead(line)
    if index != -1:
        print('after lead [{}]'.format(line[index:]))

        party_separator_match = find_party_separator(line[index:])
        if party_separator_match:
            # index2 = party_separator_match.end() + index

            before_sep = line[index:index + party_separator_match.start()]
            after_sep = line[index + party_separator_match.end():]

            print('\nbefore party_sep [{}]'.format(before_sep))
            print('\nafter party_sep [{}]'.format(after_sep))

            mm1 = find_a_corp(before_sep)
            print('\nmm1 = [{}]'.format(before_sep[mm1.start():mm1.end()]))

            before_sep2 = before_sep[mm1.end():]
            mm2 = find_a_corp(before_sep2)
            print('\nmm2 = [{}]'.format(before_sep2[mm2.start():mm2.end()]))

            mm3 = find_a_corp(after_sep)
            print('\nmm3 = [{}]'.format(after_sep[mm3.start():mm3.end()]))

            after_sep2 = after_sep[mm3.end():]
            mm4 = find_a_corp(after_sep2)
            print('\nmm4 = [{}]'.format(after_sep2[mm4.start():mm4.end()]))

    return []


def find_a_corp2(line):
    pat = re.compile(r'([A-Z][a-zA-Z\.\, ]*)+ [^\(]+\([^\)]+\)[\.,]?')
    return re.search(pat, line)


def extract_name_parties(line):
    index = find_first_party_lead(line)
    if index != -1:
        print('after lead [{}]'.format(line[index:]))

        party_separator_match = find_party_separator2(line[index:])
        if party_separator_match:
            # index2 = party_separator_match.end() + index

            before_sep = line[index:index + party_separator_match.start()]
            after_sep = line[index + party_separator_match.end():]

            print('\nbefore party_sep [{}]'.format(before_sep))
            print('\nafter party_sep [{}]'.format(after_sep))

            mm1 = find_a_corp2(before_sep)
            print('\nmm1 = [{}]'.format(before_sep[mm1.start():mm1.end()]))

            mm3 = find_a_corp2(after_sep)
            print('\nmm3 = [{}]'.format(after_sep[mm3.start():mm3.end()]))

    return []


if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('files', metavar='FILE', nargs='*',
                        help='files to read, if empty, stdin is used')

    # pylint: disable=invalid-name
    args = parser.parse_args()

    # pylint: disable=line-too-long
    st = 'THIS FIFTH AMENDMENT TO CREDIT AGREEMENT, dated as of December 17, 2012 (this “Amendment”) to the Existing Credit Agreement (such capitalized term and other capitalized terms used in this preamble and the recitals below to have the meanings set forth in, or are defined by reference in, Article I below) is entered into by and among W.E.T. AUTOMOTIVE SYSTEMS, AG, a German stock corporation (the “German Borrower”), W.E.T. AUTOMOTIVE SYSTEMS LTD., a Canadian corporation (together with the German Borrower, the “Borrowers” and each, a “Borrower”), each lender party hereto (collectively, the “Lenders” and individually, a “Lender”), BANC OF AMERICA SECURITIES LIMITED, as administrative agent (in such capacity, the “Administrative Agent”) and BANK OF AMERICA, N.A., as Swing Line Lender and L/C Issuer (“Bank of America”).'


    # party_list = extract_parties(st)

    # pylint: disable=line-too-long
    st2 = 'THIS REVOLVING LINE OF CREDIT LOAN AGREEMENT (this “Agreement”) is made as of May 29, 2009, by and between Michael Reger having a business address at 777 Glade Road Suite 300, Boca Raton, Florida 33431("Lender") and GelTech Solutions, Inc., a Delaware Coloration (the "Borrower"), having a business address at 1460 Park Lane South Suite 1, Jupiter, Florida 33458 attention, Michael Cordani.'

    party_list = extract_name_parties(st2)


    st3 = 'This Executive Employment Agreement (this "Agreement") is made this 21st day of May, 2010 (the "Effective Date"), by and between MOLYCORP, INC., a Delaware corporation ("Employer") and John Burba ("Executive"). '

    party_list = extract_name_parties(st3)
