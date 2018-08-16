#!/usr/bin/env python3

import argparse
import re

from typing import List, Match, Optional, Tuple

from kirke.utils import engutils, nlputils, strutils


IS_DEBUG_MODE = False
IS_DEBUG_PARTY_LINE = False


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
               'agreement.*is delivered by',
               'note.* is made by.*for.*benefit',
               'to the order of',
               'promises to pay to']

PARTY_PAT = re.compile(r'\b({})\b'.format('|'.join(ST_PAT_LIST)), re.IGNORECASE)

MADE_BY_PAT = re.compile(r'\bmade(.*)by\b', re.IGNORECASE)
CONCLUDED_PAT = re.compile(r'\bhave concluded.*agreement\b', re.IGNORECASE)

THIS_AGREEMENT_PAT = re.compile(r'this.*agreement\b', re.IGNORECASE)

REGISTERED_PAT = re.compile(r'\bregistered\b', re.I)

def is_made_by_check(line: str) -> bool:
    mat = MADE_BY_PAT.search(line)
    return bool(mat and len(mat.group(1)) < 20)


def is_valid_uppercase_party_name(line: str) -> bool:
    """Verify a name is a party name.

    'london branch' is invalid
    'acting xxx' is invalid
    """
    words = line.split()
    if words[-1].lower() == 'branch' and len(words) < 3:
        return False
    if words[0].lower() == 'acting':
        return False
    return True


# used by seline_attrs_to_tabled_party_list_terms()
def find_uppercase_party_name(line: str) \
-> Optional[Tuple[Tuple[int, int], int, bool]]:
    found_party_se_other = find_first_non_title_and_org(line)
    if found_party_se_other:
        (party_start, party_end), other_start = found_party_se_other
        party_st = line[party_start:party_end]
        # 'Johnson & Johnson Medikal Sanayi Ve Ticaret Limited Sirketi'
        if len(party_st) > 50:
            # find the end org
            org_mat_list = nlputils.get_org_suffix_mat_list(party_st)
            if org_mat_list:
                last_org_mat = org_mat_list[-1]
                party_end = party_start + last_org_mat.end()
                party_st = line[party_start:party_end]
                other_start = strutils.find_next_not_space_idx(line, party_end)
        elif len(list(re.finditer(r'\b(of|de|du)\b', party_st, re.I))) > 1:
            of_mat_list = list(re.finditer(r'\b(of|de|du)\b', party_st, re.I))
            # there should be only 1 "of", otherwise, likely an address
            last_of_mat = of_mat_list[1]
            party_end = party_start + last_of_mat.end()
            party_st = line[party_start:party_end]
            other_start = strutils.find_next_not_space_idx(line, party_end)

        return (party_start, party_end), other_start, is_valid_uppercase_party_name(party_st)
    return None


# used by seline_attrs_to_tabled_party_list_terms()
def find_uppercase_party_name_list(line: str) \
    -> List[Tuple[int, int]]:

    """Extract all party names from line.

    The party has to satisfy the following 3 criteria
       - mostly capitalized words, with some exception such as 'eBrevia', 'plc', or 'pic'
       - ',' can be used to separate parties
       - normally would have org_person suffix
       - more likely after 'and'
    """
    result = []
    offset = 0
    party_se_other_idx = find_uppercase_party_name(line)
    while party_se_other_idx:
        # print("party_se_other_idx: {}".format(party_se_other_idx))
        (pstart, pend), other_start, is_valid = party_se_other_idx
        # print("party_st = [{}]".format(line[offset + pstart:offset+pend]))
        if is_valid:
            result.append((offset + pstart, offset + pend))

        offset += other_start
        after_line = line[offset:]
        # print("after_line = [{}]".format(after_line))

        and_mat = re.match(r'and\s+', after_line)
        if and_mat:
            and_mat_len = len(and_mat.group())
            offset += and_mat_len
            after_line = after_line[and_mat_len:]
            # print("after_line2 = [{}]".format(after_line))

        party_se_other_idx = find_uppercase_party_name(after_line)
    return result


def find_not_title_idx(line: str) -> int:
    comma_idx = line.index(',')
    if comma_idx != -1:
        line = line[:comma_idx]
    se_word_list = list(nlputils.word_comma_tokenize(line))
    for unused_wstart, wend, word in se_word_list:
        if re.match('(of|de)\b', word, re.I):
            continue
        if not word[0].isupper():
            return wend
    return -1


# pylint: disable=too-many-locals, too-many-return-statements, too-many-branches, too-many-statements
def find_first_non_title_and_org(line: str) -> Optional[Tuple[Tuple[int, int], int]]:
    """Find the first non-title and non-org word.

    The string might have "and", such as "Johnson and Johnson Inc", or
    has digit, "Apartment 3 corp".  Needs to jump to the end of both.

    Return the start, end of company name, followed by the start of the rest of the line
    """
    prev_end = -1
    maybe_se_other_start = None
    other_word_idx, other_word = -1, ''
    se_word_list = list(nlputils.word_comma_tokenize(line))
    # print("se_word_list = {}".format(se_word_list))
    word_i = 0
    # pylint: disable=too-many-nested-blocks
    for start, end, word in se_word_list:
        if word == ',':
            prev_aft_end = -1
            word_j = word_i + 1

            if word_j < len(se_word_list):
                aft_start, unused_aft_end, aft_word = se_word_list[word_j]

                tmp_org_mat = nlputils.ORG_PERSON_SUFFIX_PAT.match(line[aft_start:])
                if tmp_org_mat:
                    prev_aft_end = aft_start + tmp_org_mat.end()
                    other_start = strutils.find_next_not_space_idx(line, prev_aft_end)
                    prev_end = prev_aft_end
                    maybe_se_other_start = 0, prev_end, other_start

                    other_word_idx = word_j
                    # now set the other_word_idx to the right one
                    for tmp2_wstart, unused_tmp2_wend, tmp2_word in se_word_list[word_j:]:
                        if tmp2_wstart > prev_aft_end:
                            other_word = tmp2_word
                            break
                        other_word_idx += 1
                    break
                # there is no org after comma
                maybe_se_other_start = 0, prev_end, aft_start
                other_word_idx, other_word = word_i+1, aft_word
            else:
                # exactly as found a lowercase word
                maybe_se_other_start = 0, prev_end, start
                other_word_idx, other_word = word_i, word
            # break one way or the other
            break
        elif word.islower() and not nlputils.is_org_suffix(word):
            # found a lowercase word
            maybe_se_other_start = 0, prev_end, start
            other_word_idx, other_word = word_i, word
            break
        # if this is an abbreviation with a period, we will
        # take the period
        if len(word) == 1 and \
           end < len(line) and \
           line[end] == '.':
            prev_end = end + 1
        else:
            prev_end = end

        word_i += 1

    # cannot find begin title
    if prev_end == -1:
        return None
    elif not maybe_se_other_start:  # the whole line has istitle()
        return (0, prev_end), prev_end

    unused_fx_start, unused_fx_end, other_start = maybe_se_other_start

    after_line = line[other_start:]

    # handle 'Johnson and Johnson'
    if strutils.is_digits(other_word) or \
       ((word_i < 2) and \
        other_word.lower() == 'and'):
        if other_word_idx + 1 < len(se_word_list):
            # the start of the first word after 'and'
            other_start = se_word_list[other_word_idx+1][0]

            prev_end = se_word_list[other_word_idx][1]  # the end of the 'and'
            for sc_start, sc_end, word in se_word_list[other_word_idx+1:]:
                if word == ',':
                    tmp_other_start = strutils.find_next_not_space_idx(line, sc_end)
                    return (0, prev_end), tmp_other_start
                elif word.islower() and not nlputils.is_org_suffix(word):
                    return (0, prev_end), sc_start
                prev_end = sc_end

            # reaching here means the whole line is istitle()
            return (0, prev_end), prev_end
        # there is no more words, return everything befefore 'and'
        return (0, prev_end), prev_end
    elif re.match('r\b(of|de)\b', other_word, re.I):
        # Royal Bank of Canada
        tmp_idx = find_not_title_idx(after_line)
        if tmp_idx != -1:
            prev_end = other_start + tmp_idx
            other_start = strutils.find_next_not_space_idx(line, prev_end+1)

    # if want to handle "Citibank, n.a.", can do it here
    # by regex matching
    elif nlputils.ORG_PERSON_SUFFIX_PAT.match(after_line):
        # do matching again, this will be rare, tolerate the cost
        mat = nlputils.ORG_PERSON_SUFFIX_PAT.match(after_line)
        # mat has to be True because the "elif" check earlier
        if mat:
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

    # only get the first sentence of the line, otherwise too much junk
    line = nlputils.first_sentence(line)

    result = is_party_line_aux(line)

    if IS_DEBUG_PARTY_LINE:
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

    num_all_upper_words = strutils.count_all_upper_words(line)
    if num_all_upper_words >= 40 and \
       num_all_upper_words / len(line.split()) > 0.8:
        return 'False1.102'

    if re.match(r'this\s+agreement\s+is\s+dated\b', line, re.I):
        return 'True0.1'

    if re.match(r'for\s+(the\s+)?value\s+received\b', line, re.I) and \
       re.search(r'\b(promise\w*\s+to\s+pay)\b', line, re.I):
        return 'True0.12'

    if len(line) > 5000:  # sometime the whole doc is a line
        return 'False1'

    if re.search(r'\bby\s+and\s+between\b', line, re.I) and \
       len(line) > 100:
        return 'True0.1242'

    # export-train/39752.txt
    if re.search(r'\bpropose(d|s)?\s+to\s+issue\b.*notes', line, re.I):
        return 'True0.1243'

    if re.search(r'\b(engages?|made\s+available|subordinate\s+to|all\s+liens)\b', line, re.I):
        return 'False2'

    # Now, Therefore
    if re.match(r'now\b', line, re.I):
        return 'False2.1.2'

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

    # must be before False4.1
    # This often only happend in the title, intentionally not handle
    # if re.match(r'by\s+and\s+between\b', line, re.I):
    #    return 'True28.1'

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
        # pylint: disable=fixme
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

    num_org_suffix = len(nlputils.get_org_suffix_mat_list(line))
    if 'among' in line and ' dated ' in line and num_org_suffix > 2:
        return 'True7'

    # this is from a title line, not a party line
    if len(line) < 200 and nlputils.ORG_PERSON_SUFFIX_END_PAT.search(line) and \
       line.strip()[-1] != '.':
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


    if re.search(r'\b(excluding|excludes?|closing)\b', line, re.I):
        return 'False1.0.13'

    if re.search(r'\b(entered)\b', line, re.I) and \
       re.search(r'\b(by\s+and\s+between)\b', line, re.I):
        return 'True9'

    if re.search(r'\b(agreement|contract)\b', line, re.I) and \
       re.search(r'\b(entered\s+into)\b', line, re.I):
        return 'True9.1'

    # 39871.txt
    # too many false positives
    if re.search(r'\bcorporation.*\band\b.*corporation.*\bagree[sd]?\b', line, re.I) and \
       not re.search(r'\b(therefore|subject.*conditions?|during)\b', line, re.I):
        return 'True9.1.2.2'

    # export-train/39856.txt
    if re.search(r'\bagree[sd]?\b.*\bto\b.*contract', line, re.I) or\
       re.search(r'\bagree[sd]?\b.*as\b.*follows', line, re.I):
        return 'True9.1.2.3'

    # if re.search(r'\band\b.*\bagree[sd]?\b', line, re.I) and \
    #    not re.search(r'\b(therefore|subject.*conditions?|during)\b', line, re.I):
    #     return 'True9.1.2'

    # mytest/doc1.txt fail on this
    # pylint: disable=pointless-string-statement
    """
    if re.search(r'\b(agreement|contract)\b', line, re.I) and \
       re.search(r'\b(dated)\b', line, re.I):
        return 'True9.2'
    """

    # pylint: disable=fixme
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
    if re.search(r'agreement\s+is\s+made\s+(on|as\s+of)', line, re.I):
        return 'True14'

    if line.startswith('T') and \
       re.match('(this|the).*(contract|lease|agreement).*is made', line, re.I):
        return 'True15'
    if len(line) < 40:  # don't want to match line "BY AND BETWEEN" in title page
        return 'False16'
    if engutils.is_skip_template_line(line):
        return 'False17'
    if 'means' in line:  # in definition section of 'purchase agreement'
        return 'False18'
    if re.match(r'now\b', line, re.I) or \
       re.search(r'\btherefore\b', line, re.I):
        return 'False18.3'

    mat = PARTY_PAT.search(line)
    if mat and \
       not re.search(r'\b(assign(s|ed)?|percent|rate)\b', line, re.I):
        # export-train/35736.txt
        # export-train/38849.txt for "restated, modified"
        return 'True8.8'  # bool(mat)

    lc_line = line.lower()
    if 'between' in lc_line and engutils.has_date(lc_line):
        return 'True19'
    if 'made' in lc_line and engutils.has_date(lc_line) and 'agreement' in lc_line:
        return 'True20'
    if 'issued' in lc_line and engutils.has_date(lc_line) and 'agreement' in lc_line:
        return 'True21'
    if re.search(r'\benter(ed|s)?\b', line, re.I) and \
       engutils.has_date(lc_line) and \
       'agreement' in lc_line:
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

    if THIS_AGREEMENT_PAT.search(line) and \
       "amendment" in lc_line and \
       not 'section' in lc_line:
        return 'True28'

    if re.search(r'\b(following\s+meanings?)\b', line, re.I):
        return 'False33.12'

    # termination agreement
    if 'agree that' in lc_line and 'employment at' in lc_line:
        return 'True29'
    # 'Patent Security Agreement, dated as of December 1, 2009, by BUSCH
    # ENTERTAINMENT LLC (the “Grantor”), in favor of BANK OF AMERICA, N.A...'
    if 'date' in lc_line and ' by ' in lc_line and 'in favor of' in lc_line:
        return 'True30'

    # export-train/35739.txt
    if 'section'  in lc_line or \
       'elsewhere'  in lc_line or \
       'meanings'  in lc_line or \
       lc_line.startswith('in addition'):
        return 'False33.124'

    if 'reach an agreement' in lc_line or \
       'the following terms' in lc_line or \
       'terms and conditions' in lc_line or \
       'enter into this contract' in lc_line:
        return 'True31'
    if 'hereinafter' in lc_line and \
       'agree' in lc_line and \
       not 'subject to' in lc_line:
        return 'True32'
    if 'confirm' in lc_line and \
       'agree' in lc_line and \
       not 'subject to' in lc_line:
        return 'True33'
    # NOW, THEREFORE, in consideration of the premises and the mutual covenants
    # contained in this Agreement,  the Parties hereto agree as follows:
    if re.search(r'\b(now|therefore|in cosideration)\b', line, re.I):
        return 'False33.1'
    #"""In this Agreement (unless the context requires otherwise) the following words
    # shall have the following meanings"""
    # if 'agree' in lc_line and 'follow' in lc_line and not "meaning" in lc_line:
    #    return 'True34'


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
    return 'False99'


# pylint: disable=invalid-name
def is_party_line_prefix_without_parties(line: str) -> bool:
    """Detect if a line have no party, we don't want to detect 'The Agreement ... as
       a party.

    This is whatever is after "between" and end of sentence.
    """
    if len(line) >= 100:  # it's not obvious that it has no party
        return False

    between_mat = re.search(r'\b(between|among)\b', line, re.I)
    if between_mat and between_mat.end() < len(line) - 10:
        return True

    org_suffix_list = nlputils.get_org_suffix_mat_list(line)
    num_org_suffix = len(org_suffix_list)

    # num_and = len(list(re.finditer(r'\band\b', line, re.I)))
    num_parens = len(list(re.finditer(r'\([^\)]+\)', line, re.I)))
    # need to skip first 'the agreement is'
    maybe_org_st_list = []
    for multi_cap_mat in re.finditer(r'\b[A-Z]\w+( [A-Z]\w+)+\b', line, re.I):
        multi_cap_st = line[multi_cap_mat.start():multi_cap_mat.end()]
        if re.search(r'\bdate\b', multi_cap_st, re.I) or \
           re.search(r'\b(agreement|lease|contract)\b', multi_cap_st, re.I):
            pass
        else:
            maybe_org_st_list.append(multi_cap_st)

    if num_parens >= 2:
        return False
    if num_org_suffix == 0 and \
       len(maybe_org_st_list) <= 1:
        return True

    return False


# Note: I have seen up to 13 companies
def match_list_prefix(line: str) -> Optional[Match[str]]:
    """This return the match of whatever is AFTER the prefix as group(1)."""
    # the first ' is due to some OCR error, mytest/doc16.txt
    return re.match(r"'?\(?\s*[\divx]+\s*\)\s*(.*)", line, re.I)


# I have seen up to 13 companies
def match_party_list_prefix(line: str) -> Optional[Match[str]]:
    """This return the match of whatever is AFTER the prefix as group(1)."""
    num_mat = re.match(r"'?\(?\s*[\divx]+\s*\)\s*(.*)", line, re.I)
    if num_mat:
        return num_mat

    # now try "Party A:"
    mat = re.match(r'Party\s*\S+\s*:\s*(.*)', line, re.I)
    return mat


# pylint: disable=invalid-name
def is_party_list_prefix_with_validation(line: str) -> bool:
    if match_party_list_prefix(line):
        if re.search(r'\b(engages?|product)\b', line, re.I):
            return False
        return True
    return False

def is_party_list_with_end_between(line: str) -> bool:
    words = line.lower().split()
    last_8_words = words[-8:]
    # print("last_8_words = {}".format(last_8_words))
    if ('between' in last_8_words or 'among' in last_8_words) and \
       words[-1] == 'and':
        return True
    elif words[-1] in set(['between', 'among']):
        return True
    return False


def is_all_english_title_case(line: str) -> bool:
    words = strutils.get_alpha_words(line, is_lower=False)
    if not words:
        return False
    for word in words:
        # check if preposition, lc
        if not (word.istitle() or
                word.isupper() or
                word in set(['of', 'the', 'as', 'from', 'to',
                             'between', 'among', 'de', # 'of' in Spanish
                             'and', 'or'])):
            # print("failed tt: [{}]".format(word))
            return False
    return True



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

    # party_list = extract_name_parties(st2)


    st3 = 'This Executive Employment Agreement (this "Agreement") is made this 21st day of May, 2010 (the "Effective Date"), by and between MOLYCORP, INC., a Delaware corporation ("Employer") and John Burba ("Executive"). '

    # party_list = extract_name_parties(st3)
