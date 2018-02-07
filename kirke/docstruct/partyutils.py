#!/usr/bin/env python

import argparse
import logging
import re

from typing import List, Match

from kirke.utils import engutils, strutils


DEBUG_MODE = False

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

MADE_BY_PAT = re.compile(r'\bmade.*by\b', re.IGNORECASE)
CONCLUDED_PAT = re.compile(r'\bhave concluded.*agreement\b', re.IGNORECASE)

THIS_AGREEMENT_PAT = re.compile(r'this.*agreement\b', re.IGNORECASE)

REGISTERED_PAT = re.compile(r'\bregistered\b', re.I)

ORG_SUFFIX_ST = (r' ('
                 r'branch|ag|company|co|corp|corporation|d\.\s*a\.\s*c|inc|incorporated|llc|'
                 r'gmbh|'
                 r'l\.\s*l\.\s*c|ulc|'
                 r'ltd|limited|lp|l\.\s*p|limited partnership|n\.\s*a|plc|'
                 r'pca|pty|holdings?|'
                 r'bank|trust|association|group|sas|s\.\s*a|sa|c\.\s*v|cv'
                 r')\b\.?')

# copied from kirke/ebrules/parties.py on 2/4/2016
ORG_SUFFIX_PAT = re.compile(ORG_SUFFIX_ST, re.I)
ORG_SUFFIX_END_PAT = re.compile(ORG_SUFFIX_ST + r'\s*$', re.I)

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


def is_party_line(line: str) -> bool:
    result = is_party_line_aux(line)

    # do some extra verfication
    # a party line must starts with 1) a) or i) 'l' is for bad OCR 1's
    if result:
        if re.match(r'\(\S\)', line):
            # (a) EAch three (3)
            parens1_mat_list = strutils.get_consecutive_one_char_parens_mats(line)

            if parens1_mat_list:
                return True

            parens1_mat_list = strutils.get_one_char_parens_mats(line)

            if len(parens1_mat_list) == 1:
                return bool(re.match(r'\(?\s*(1|a|i|l)\s*\)', line, re.I))
            else:
                # if has one than 1 parens1_mat, should have pass the consecutive test before
                return False
    return result

# pylint: disable=too-many-return-statements, too-many-branches
def is_party_line_aux(line: str) -> bool:

    # this is not a party line due to the words used
    # adding this will decrease f1 by 0.001.  Will figure out later.
    # if re.search(r'\bif\b', line, re.I) and re.search(r'\bwithout\b', line, re.I):
    #    return False

    if re.search(r'\b(engages?|made\s+available)\b', line, re.I):
        return False

    # 2/6/2018, uk/file3.txt, multiple parties got mentioned and registered, but not a party line
    alpha_words = strutils.get_alpha_words(line, is_lower=False)
    # this is match, not search, uk/file3.txt
    if re.match(r'\d+\-\d+', line) and strutils.is_all_upper_words(alpha_words):
        return False

    #returns true if bullet type, and a real line
    # if re.match(r'\(?[\div]+\)', line) and len(line) > 60:
    mat = re.match(r'\(?\s*(1|a|i|l)\s*\)\s*(.*)', line, re.I)
    if mat and len(line) > 60:
        # TODO, jshaw, 36820.txt  Rediculous way of formatting
        # need to pass line number in to disable this aggressive matching
        # will fix later.  Not happening in PDF docs?
        """
        print("I am hereeeeeeeee")
        suffix_st = mat.group(2)
        suffix_mat = re.match(r'\s*party \(?(.*)\b', suffix_st, re.I)
        if not suffix_mat:
            return True
        if suffix_mat and \
           not (suffix_mat.group(1).startswith('A') or
                suffix_mat.group(1).startswith('1')):
            return False
        """
        return True

    # Party A: xxx,
    # Party B:
    if re.match(r'Party \S+\s*:', line, re.I) and line[0].isupper():
        return True

    num_org_suffix = len(get_org_suffix_mat_list(line))
    if 'among' in line and ' dated ' in line and num_org_suffix > 2:
        return True

    # this is from a title line, not a party line
    if len(line) < 200 and ORG_SUFFIX_END_PAT.search(line) and line.strip()[-1] != '.':
        return False

    # Removed.  This turns out to be false for UK document multiple times.
    # multipled parties mentioned
    # if len(list(REGISTERED_PAT.finditer(line))) > 1:
    #    if strutils.is_all_upper_words(alpha_words):  #
    #        return False
    #    return True


    # 44139.txt, info is attached, yada yada
    # '\$\d' doesn't work, decrease F1 by 20%!  Too many
    # promissory or loan notes has partyline with '$'
    # if re.search(r'\b(is attached|partial)\b', line, re.I):
    #    return False

    # This is NOT true.  There are agreements that this is not true.
    # 39761.txt.  Around 2% lower.
    # # 'agreement, dated may 24, 2004', is NOT a party line
    # if re.search(r'\bdated\b', line, re.I) and \
    #    not re.search(r'\bis\s+dated\b', line, re.I):
    #    return False

    if re.search(r'\b(entered)\b', line, re.I) and \
       re.search(r'\b(by\s+and\s+between)\b', line, re.I):
        return True
    # TODO, jshaw, kkk
    # [tn=0, fp=1347], [fn=2877, tp=8034]], f1=0.7918
    # => [[tn=0, fp=1335], [fn=2877, tp=8034]] f1= 0.7923
    # so remove this line reduces false positives.
    if re.search(r'\b(hereby\s+enter(ed)?\s+into)\b', line, re.I):
        return True

    # added on 02/06/2018, jshaw
    if re.search(r'way\s+of\s+deed', line, re.I):
        return True
    # 'deed of release is made"
    if re.search(r'\b(deed\s+is\s+made|deed.*is\s+made)\b', line, re.I):
        return True
    # this is slight aggressive
    if re.search(r'^This.*(deed|guarantee).*dated\b', line, re.I):
        return True

    # uk doc, file2
    # This Agreement is made on 2017
    #        Between:
    # (1) ...
    if re.search(r'agreement\s+is\s+made\s+on', line, re.I):
        return True

    if line.startswith('T') and \
       re.match('(this|the).*contract.*is made on', line, re.I):
        return True
    if len(line) < 40:  # don't want to match line "BY AND BETWEEN" in title page
        return False
    if engutils.is_skip_template_line(line):
        return False
    if 'means' in line:  # in definition section of 'purchase agreement'
        return False
    mat = PARTY_PAT.search(line)
    if mat:
        return bool(mat)
    lc_line = line.lower()
    if 'between' in lc_line and engutils.has_date(lc_line):
        return True
    if 'made' in lc_line and engutils.has_date(lc_line) and 'agreement' in lc_line:
        return True
    if 'issued' in lc_line and engutils.has_date(lc_line) and 'agreement' in lc_line:
        return True
    if 'entered' in lc_line and engutils.has_date(lc_line) and 'agreement' in lc_line:
        return True
    # power of attorney
    if 'made on' in lc_line and engutils.has_date(lc_line) and 'power' in lc_line:
        return True
    if 'between' in lc_line and 'agreement' in lc_line:
        return True
    # assigns lease to
    if 'assign' in lc_line and 'lease to' in lc_line:
        return True
    if MADE_BY_PAT.search(line) and ('day' in lc_line or
                                     'date' in lc_line):
        return True
    if CONCLUDED_PAT.search(line):
        return True

    if THIS_AGREEMENT_PAT.search(line) and "amendment" in lc_line:
        return True

    # termination agreement
    if 'agree that' in lc_line and 'employment at' in lc_line:
        return True
    # 'Patent Security Agreement, dated as of December 1, 2009, by BUSCH
    # ENTERTAINMENT LLC (the “Grantor”), in favor of BANK OF AMERICA, N.A...'
    if 'date' in lc_line and ' by ' in lc_line and 'in favor of' in lc_line:
        return True
    if 'reach an agreement' in lc_line or \
       'the following terms' in lc_line or \
       'terms and condistions' in lc_line or \
       'enter into this contract' in lc_line:
        return True
    if 'hereinafter' in lc_line and 'agree' in lc_line:
        return True
    if 'confirm' in lc_line and 'agree' in lc_line:
        return True
    #"""In this Agreement (unless the context requires otherwise) the following words
    # shall have the following meanings"""
    if 'agree' in lc_line and 'follow' in lc_line and not "meaning" in lc_line:
        return True
    if 'follow' in lc_line and 'between' in lc_line:
        return True
    # for warrants, 'the Lenders from time to time party thereto,'
    if 'from time to time party thereto' in lc_line:
        return True

    #"""This Amendment No. 1 to the Convertible Promissory Note (this
    #   "Amendment") is executed as of October 17, 2011, by SOLAR ENERGY
    #   INITIATIVES, INC., a Nevada corporation (the “Maker”); and ASHER
    #   ENTERPRISES, INC., a Delaware corporatio"""
    if 'is executed' in lc_line and \
       'by' in lc_line and \
       'and' in lc_line:
        return True
    # agreement is made to ..., dated as of ... among
    if 'agreement' in lc_line and \
       'dated' in lc_line and \
       'among' in lc_line:
        return True
    # 'this certifies that, ... is entitled to,
    if 'is entitled to' in lc_line and \
       'certifies' in lc_line and \
       'purchase' in lc_line:
        return True
    if 'is made' in lc_line and \
    'following parties' in lc_line:
        return True
    return False


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
