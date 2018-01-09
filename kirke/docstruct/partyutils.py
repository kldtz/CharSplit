#!/usr/bin/env python

import argparse
import fileinput
import logging
import sys
import warnings
import re
import pprint

from kirke.utils import strutils
from kirke.utils import txtreader, engutils


DEBUG_MODE = False

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(levelname)s : %(message)s')

st_pat_list = ['is made and entered into by and between',
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
               'promises to pay to'
]

party_pat = re.compile(r'\b({})\b'.format('|'.join(st_pat_list)), re.IGNORECASE)

made_by_pat = re.compile(r'\bmade.*by\b', re.IGNORECASE)
concluded_pat = re.compile(r'\bhave concluded.*agreement\b', re.IGNORECASE)

this_agreement_pat = re.compile(r'this.*agreement\b', re.IGNORECASE)


def is_party_line(line):
    if re.match(r'\(?[\div]+\)', line):
        return True
    if len(line) < 40:  # don't want to match line "BY AND BETWEEN" in title page
        return False
    if engutils.is_skip_template_line(line):
        return False
    if 'means' in line:  # in definition section of 'purchase agreement'
        return False
    mat = party_pat.search(line)
    if mat:
        return mat
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
    if made_by_pat.search(line) and ('day' in lc_line or
                                     'date' in lc_line):
        return True
    if concluded_pat.search(line):
        return True

    if this_agreement_pat.search(line) and "amendment" in lc_line:
        return True

    # termination agreement
    if 'agree that' in lc_line and 'employment at' in lc_line:
        return True
    # 'Patent Security Agreement, dated as of December 1, 2009, by BUSCH ENTERTAINMENT LLC (the “Grantor”), in favor of BANK OF AMERICA, N.A...'
    if 'date' in lc_line and ' by ' in lc_line and 'in favor of' in lc_line:
        return True    
    if ('reach an agreement' in lc_line or
        'the following terms' in lc_line or
        'terms and condistions' in lc_line or
        'enter into this contract' in lc_line):
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
    if ('is executed' in lc_line and
        'by' in lc_line and
        'and' in lc_line):
        return True
    # agreement is made to ..., dated as of ... among
    if ('agreement' in lc_line and
        'dated' in lc_line and
        'among' in lc_line):
        return True
    # 'this certifies that, ... is entitled to,
    if ('is entitled to' in lc_line and
        'certifies' in lc_line and
        'purchase' in lc_line):
        return True
    if ('is made' in lc_line and
        'following parties' in lc_line):
        return True    
    return False

def find_first_party_lead(line):
    lc_line = line.lower()
    for st_pat in st_pat_list:
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
    pat = re.compile(r'[A-Z\.\, ]+ (AG|LTD|N\.\s*A|Limited|LIMITED)\.?, (a|as)\b[^\(]+\([^\)]+\)[\.,]?')
    return re.search(pat, line)


def extract_parties(line):
    index = find_first_party_lead(line)
    if index != -1:
        print('after lead [{}]'.format(line[index:]))

        party_separator_match = find_party_separator(line[index:])
        if party_separator_match:
            index2 = party_separator_match.end() + index

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
            index2 = party_separator_match.end() + index

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
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('files', metavar='FILE', nargs='*',
                        help='files to read, if empty, stdin is used')    

    args = parser.parse_args()


    st = 'THIS FIFTH AMENDMENT TO CREDIT AGREEMENT, dated as of December 17, 2012 (this “Amendment”) to the Existing Credit Agreement (such capitalized term and other capitalized terms used in this preamble and the recitals below to have the meanings set forth in, or are defined by reference in, Article I below) is entered into by and among W.E.T. AUTOMOTIVE SYSTEMS, AG, a German stock corporation (the “German Borrower”), W.E.T. AUTOMOTIVE SYSTEMS LTD., a Canadian corporation (together with the German Borrower, the “Borrowers” and each, a “Borrower”), each lender party hereto (collectively, the “Lenders” and individually, a “Lender”), BANC OF AMERICA SECURITIES LIMITED, as administrative agent (in such capacity, the “Administrative Agent”) and BANK OF AMERICA, N.A., as Swing Line Lender and L/C Issuer (“Bank of America”).'
    

    # party_list = extract_parties(st)


    st2 = 'THIS REVOLVING LINE OF CREDIT LOAN AGREEMENT (this “Agreement”) is made as of May 29, 2009, by and between Michael Reger having a business address at 777 Glade Road Suite 300, Boca Raton, Florida 33431("Lender") and GelTech Solutions, Inc., a Delaware Coloration (the "Borrower"), having a business address at 1460 Park Lane South Suite 1, Jupiter, Florida 33458 attention, Michael Cordani.'

    party_list = extract_name_parties(st2)
    

    st3 = 'This Executive Employment Agreement (this "Agreement") is made this 21st day of May, 2010 (the "Effective Date"), by and between MOLYCORP, INC., a Delaware corporation ("Employer") and John Burba ("Executive"). '

    party_list = extract_name_parties(st3)    
