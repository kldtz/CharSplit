#!/usr/bin/env python3

import argparse
import logging
from pprint import pprint
import sys
import warnings
import re

from nltk.tree import Tree

from kirke.utils import nlputils, strutils

from kirke.docstruct import partyutils
from kirke.ebrules import parties

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    # parser.add_argument('file', help='input file')

    args = parser.parse_args()


    # line = '1) aba bd (b) a2df'    
    # start, end, word = strutils.find_previous_word(line, 6)

    """
    line = 'Volkswagen Bank GmbH, a company incorporated under I.B.M. Corp.'
    mat_list = partyutils.get_org_suffix_mat_list(line)
    st_list = [line[mat.start():mat.end()] for mat in mat_list]
    print("org: {}".format(st_list))
    """

    """
    line = 'Volkswagen Bank GmbH, a company incorporated under'
    result = partyutils.find_non_lc_party_name(line)
    print("org: {}".format(result))
    """

    """
    line = 'Citibank Bank, N.A. is a bank incorporated under'
    result = partyutils.find_non_lc_party_name(line)
    print("org: {}".format(result))
    print("org2: [{}]".format(line[result[0][0]:result[0][1]]))
    """

    """
    line = 'HSBC Bank pic, a bank incorporated under'
    result = partyutils.find_non_lc_party_name(line)
    print("org: {}".format(result))
    """

    """
    line = 'HSBC Bank pic '
    result = partyutils.find_non_lc_party_name(line)
    print("org: {}".format(result))
    """

    """
    line = 'Business Marketing Services, Inc. One Broadway Street,'    
    result = partyutils.find_non_lc_party_name(line)
    print("org: {}".format(result))
    print("org2: [{}]".format(line[result[0][0]:result[0][1]]))


    line = 'Business Marketing Services, Inc, One Broadway Street,'    
    result = partyutils.find_non_lc_party_name(line)
    print("org: {}".format(result))
    print("org2: [{}]".format(line[result[0][0]:result[0][1]]))

    line = 'Business Marketing Services, Inc., One Broadway Street,'    
    result = partyutils.find_non_lc_party_name(line)
    print("org: {}".format(result))
    print("org2: [{}]".format(line[result[0][0]:result[0][1]]))

    line = 'Business Marketing Services, Inc'
    result = partyutils.find_non_lc_party_name(line)
    print("org: {}".format(result))
    print("org2: [{}]".format(line[result[0][0]:result[0][1]]))

    line = 'Business Marketing Services, Incor'
    result = partyutils.find_non_lc_party_name(line)
    print("org: {}".format(result))
    print("org2: [{}]".format(line[result[0][0]:result[0][1]]))

    line = 'Business Marketing Services, Inc,. One Broadway Street,'    
    result = partyutils.find_non_lc_party_name(line)
    print("org: {}".format(result))
    print("org2: [{}]".format(line[result[0][0]:result[0][1]]))
    """

    """
    line = 'I.B.M. and Dell Inc., are in a war, battle, and cold-war.'
    result = list(strutils.word_comma_tokenize(line))
    for i, se_word in enumerate(result):
        print('{}\t{}'.format(i, se_word))
    print(list(result))
    """

    """
    line = 'ABN AMRO BANK N.V., BANC OF AMERICA SECURITIES LIMITED, BARCLAYS CAPITAL,  CITIGROUP GLOBAL MARKETS LIMITED, COÖPERATIEVE CENTRALE RAIFFEISEN- BOERENLEENBANK B.A., DEUTSCHE BANK AG, LONDON BRANCH, HSBC BANK PLC, ING  BANK N.V., JPMORGAN CHASE BANK N.A., MORGAN STANLEY BANK INTERNATIONAL  LIMITED, ROYAL BANK OF CANADA and THE ROYAL BANK OF SCOTLAND PLC as  mandated lead arrangers (the "Mandated Lead Arrangers");'
    # line = 'ROYAL BANK OF CANADA and THE ROYAL BANK OF SCOTLAND PLC'
    result = partyutils.find_uppercase_party_name_list(line)
    for i, party_se in enumerate(result):
        print("party #{}: [{}]".format(i, line[party_se[0]:party_se[1]]))
    """

    """
    line = 'BNP PARIBAS, GOLDMAN SACHS BANK USA and SOCIÉTÉ GÉNÉRALE S.A., ACTING  THROUGH ITS AMSTERDAM BRANCH as lead arrangers (the “Lead Arrangers”);'
    result = partyutils.find_uppercase_party_name_list(line)
    for i, party_se in enumerate(result):
        print("party #{}: [{}]".format(i, line[party_se[0]:party_se[1]]))
    """

    """
    line = 'ABN AMRO BANK N.V., BANC OF AMERICA SECURITIES LIMITED, BARCLAYS CAPITAL,  CITIGROUP GLOBAL MARKETS LIMITED, DEUTSCHE BANK AG, ING BANK N.V.,  JPMORGAN CHASE BANK N.A., MORGAN STANLEY and ROYAL BANK OF CANADA as  bookrunners (the "Bookrunners");'
    result = partyutils.find_uppercase_party_name_list(line)
    for i, party_se in enumerate(result):
        print("party #{}: [{}]".format(i, line[party_se[0]:party_se[1]]))
    """

    """
    line = 'Johnson and Johnson, a bank incorporated under'
    result = partyutils.find_uppercase_party_name_list(line)
    for i, party_se in enumerate(result):
        print("party #{}: [{}]".format(i, line[party_se[0]:party_se[1]]))    
    """

    """
    line = 'Citibank Bank, n.a. is smaller'
    line = 'HSBC Bank Brasil S.A - Banco Multiple, having its registered office address at'
    result = partyutils.find_uppercase_party_name(line)
    print("result: {}".format(result))

    print("party_st: [{}]".format(line[result[0][0]:result[0][1]]))
    print("after_line: [{}]".format(line[result[1]:]))
    """
    
    """
    result = partyutils.find_uppercase_party_name_list(line)
    print("result: {}".format(result))
    for i, party_se in enumerate(result):
        print("party #{}: [{}]".format(i, line[party_se[0]:party_se[1]])) 
    """

    """
    line = 'Aviva International Insurance Limited, registered in England under no. 21487 and having  its registered office at St. Helen’s, 1 Undershaft, London EC3P 3DQ and Aviva Insurance  Limited, registered in Scotland under no. 2116 and having its registered office at Pitheavlis,  Perth PH2 ONH (together “the Guarantor”)'
    result = partyutils.find_uppercase_party_name_list(line)
    for i, party_se in enumerate(result):
        print("party #{}: [{}]".format(i, line[party_se[0]:party_se[1]]))
    """

    """
    line = '(2) Aviva International Insurance Limited, registered in England under no. 21487 and having  its registered office at St. Helen’s, 1 Undershaft, London EC3P 3DQ and Aviva Insurance  Limited, registered in Scotland under no. 2116 and having its registered office at Pitheavlis,  Perth PH2 ONH (together “the Guarantor”)'        
    parties, term = parties.party_line_group_to_party_list_term([(0, len(line), line, [])])
    for i, party_se in enumerate(parties):
        print("party #{}: [{}]".format(i, line[party_se[0]:party_se[1]]))
    print("term: [{}]".format(line[term[0]:term[1]]))
    """

    """
    line = 'This Non-Disclosure Agreement (“Agreement”), effective as of the last signature  date below, (“Effective Date”), is by and between Partner 4, LLC, a Virginia Corporation  having its headquarters located at 999 Parkview Drive, West Church, VA 22099, on  behalf of itself, its subsidiaries and Affiliates, (collectively, “P4”), and Box, Inc., a  Delaware Corporation having its headquarters located at 900 Jefferson Ave,  Redwood City, CA 94063, on behalf of itself and its subsidiaries and Affiliates  (collectively, “Supplier”).  The term “Affiliates” shall mean those entities controlled  by, which control or which are under common control with an identified named Party.  Such entity shall be deemed to be an Affiliate only so long as such control exists.  Such control means: (i) direct or indirect ownership or control (now or hereafter) of  more than fifty percent (50%)'
    print("first_line = [{}]".format(nlputils.first_sentence(line)))
    """

    """
    line = 'This Non-Disclosure Agreement (“Agreement”), effective as'
    print('\nline = [{}]'.format(line))
    print("first_line = [{}]".format(nlputils.first_sentence(line)))

    tokens = nlputils.tokenize(line)
    for i, token in enumerate(tokens):
        print("token #{}\t[{}]".format(i, token))

    line = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).'
    print('\nline = [{}]'.format(line))    
    # print("first_line = [{}]".format(nlputils.first_sentence(line)))
    print("first_line = [{}]".format(nlputils.first_sentence(line)))

    tokens = nlputils.tokenize(line)
    for i, token in enumerate(tokens):
        print("token #{}\t[{}]".format(i, token))

    print('\nline = [{}]'.format(line))            
    tokens = nlputils.span_tokenize(line)
    for i, span_token in enumerate(tokens):
        tstart, tend = span_token
        print("span_token #{}\t{}\t[{}]".format(i, span_token, line[tstart:tend]))


    print('\nline = [{}]'.format(line))        
    tokens = nlputils.word_punct_tokenize(line)
    for i, token in enumerate(tokens):
        print("wpunct_token #{}\t[{}]".format(i, token))


    print('\nline = [{}]'.format(line))        
    tokens = nlputils.jm_sent_tokenize(line)
    for i, span_token in enumerate(tokens):
        tstart, tend = span_token
        print("span_token #{}\t{}\t[{}]".format(i, span_token, line[tstart:tend]))
    

    line = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).  James'
    print('\nline = [{}]'.format(line))            
    tokens = nlputils.sent_tokenize(line)
    for i, span_token in enumerate(tokens):
        tstart, tend = span_token
        print("sentence #{}\t{}\t[{}]".format(i, span_token, line[tstart:tend]))


    line = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).  James'
    print('\nline = [{}]'.format(line))            
    tokens = nlputils.span_tokenize(line)
    for i, span_token in enumerate(tokens):
        tstart, tend = span_token
        print("non-sent span_token #{}\t{}\t[{}]".format(i, span_token, line[tstart:tend]))


    line = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).  James'
    print('\nline = [{}]'.format(line))            
    tokens = nlputils.text_span_tokenize(line)
    for i, span_token in enumerate(tokens):
        tstart, tend = span_token
        print("span_token #{}\t{}\t[{}]".format(i, span_token, line[tstart:tend]))                

    """

    # print('\npos')
    line = 'This Non-Disclosure Agreement (“Agreement”), effective as of the last signature  date below, (“Effective Date”), is by and between I.B.M. Corp., Partner 4, LLC, a Virginia Corporation  having its headquarters located at 999 Parkview Drive, West Church, VA 22099, on  behalf of itself, its subsidiaries and Affiliates, (collectively, “P4”), and Box, Inc., a  Delaware Corporation having its headquarters located at 900 Jefferson Ave,  Redwood City, CA 94063, on behalf of itself and its subsidiaries and Affiliates  (collectively, “Supplier”).'

    """
    tokens = nlputils.word_tokenize(line)
    tok_pos_list = nlputils.pos_tag(tokens)
    for i, tok_pos in enumerate(tok_pos_list):
        print("  pos #{}\t{}".format(i, tok_pos))
    """

    """
    xxx = nlputils.chunkize(line)
    print('chunkize =')
    print(xxx)

    for i, tree in enumerate(xxx):
        if isinstance(tree, Tree):
            print('tree #{}\t{}'.format(i, tree))
        else:
            print('pos_tag #{}\t{}'.format(i, tree))
    """

"""
nouns = nlputils.get_nouns(line)
for i, noun in enumerate(nouns):
    print("noun #{}\t{}".format(i, noun))
    
proper_nouns = nlputils.get_proper_nouns(line)
for i, pnoun in enumerate(proper_nouns):
    print("proper noun #{}\t{}".format(i, pnoun))
    
proper_nouns = nlputils.get_better_nouns(line)
for i, pnoun in enumerate(proper_nouns):
    print("better noun #{}\t{}".format(i, pnoun))

nlputils.extract_proper_names(line)

nlputils.find_nouns(line)

"""

phrase_sent = nlputils.PhrasedSent(line)

# phrase_sent.print_parsed()

orgs_term_list = phrase_sent.extract_orgs_term_list()

for i, orgs_term in enumerate(orgs_term_list):
    orgs, term = orgs_term
    print("orgs_term #{}:".format(i))
    print("    orgs:")
    for j, org in enumerate(orgs):
        print("      #{} {}".format(j, org))
    print("    term:")
    print("         {}".format(term))
    

    
"""            
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk import Tree
tagged_text = "[ The/DT cat/NN ] sat/VBD on/IN [ the/DT mat/NN ] [ the/DT dog/NN ] chewed/VBD ./."
gold_chunked_text = tagstr2tree(tagged_text)
unchunked_text = gold_chunked_text.flatten()
print("gold_chunked_text:")
print(gold_chunked_text)

print("\nunchunked_text")
print(unchunked_text)
"""

    
