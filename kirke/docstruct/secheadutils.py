#!/usr/bin/env python3
# pylint: disable=too-many-lines

import argparse
import logging
import re
# pylint: disable=unused-import
from typing import List, Optional, Tuple

from nltk.tokenize import wordpunct_tokenize

from kirke.docstruct import lxlineinfo, footerutils, addrutils
from kirke.utils import strutils, stopwordutils

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEBUG_MODE = False

SUBHEAD_PREFIX_PAT = re.compile(r'^[\s§]*((Section\s*)?\d+(\s*\.\d+)+\.?\b|'
                                r'\(?[a-zA-Z0-9]+\)|[a-zA-Z0-9]+\.|\(\d+)\s*(.*)$',
                                re.IGNORECASE)

# 10. (a)
SUBHEAD_PREFIX_PAT2 = re.compile(r'^[\s§]*(\d+\.?\s*\([a-z0-9]\))\s*(.*)$')

SUBHEAD_SUFFIX_PAT = re.compile(r'^(.*)\s\s+(\d+(\s*\.\s*\d+)+)\s*$')

# should not check for eoln because bad OCR or other extra text
# 'TABLE OF CONTENTS OFFICE LEASE'
TOC_PREFIX_PAT = re.compile(r'^\s*(table\s*of\s*contents?|contents?)\s*:?', re.IGNORECASE)
TOC_PREFIX_2_PAT = re.compile(r'^(.+)\.{5}')

# SECHEAD_PREFIX_PAT = re.compile(r'^((Section )?\d+\.?)\s+(.*)$')
SECHEAD_PREFIX_PAT = re.compile(r'^[\s§]*((Section\s*|Article\s*|Appendix\s*)?\d+(\.\d*)?|'
                                r'Recitals:?|article\s*[ivxm\d\.]+|'
                                r'(EXHIBIT|Exmllit)\s*“?\s*([A-Z]+\s*[\“\”]?|\d+(\.\d*)?))\s*(.*)$',
                                re.IGNORECASE)

SECHEAD_DIGIT_PAT = re.compile(r'\b(\d+|[A-Z]|[VIXM]+)\.$')


class SectionHead:

    # pylint: disable=too-many-arguments, too-many-instance-attributes, too-few-public-methods
    def __init__(self,
                 sec_cat: str,
                 start: int,
                 end: int,
                 pagenum: int,
                 sec_prefix: str,
                 sec_title: str,
                 sec_text: str,
                 head_lineinfos) -> None:
        self.category = sec_cat   # "sechead", "subsec"
        self.start = start
        self.end = end
        self.pagenum = pagenum
        self.prefix = sec_prefix
        self.title = sec_title
        self.text = sec_text
        self.head_lineinfo_list = head_lineinfos
        self.lineinfo_list = []  # type: List

    def append_lineinfo(self, lineinfo):
        self.lineinfo_list.append(lineinfo)

    def __repr__(self):
        return "SectionHead('%s', page= %d, %d, %d, '%s', '%s', '%s')" % \
            (self.category, self.pagenum, self.start, self.end, self.prefix, self.title,
             self.text)


def is_stopword(st):
    return st.lower() in ['.', ',', '/', ';', ':', '"', '\'', '-', '–', '*.',
                          '&', '’', '.”', '“', ';”',
                          'by', 'for', 'of', 's', 'at',
                          'to', 'in', 'on', 'with', 'without', 'or', 'and', 'a', 'the',
                          'this', 'that', 'from', 'up']


subsec_prefix_word_set = strutils.load_lc_str_set('dict/subsec.prefix.dict')
subsec_xxx_word_set = strutils.load_lc_str_set('dict/subsec.xxx.dict')

sechead_prefix_word_set = strutils.load_lc_str_set('dict/sechead.prefix.dict')
sechead_xxx_word_set = strutils.load_lc_str_set('dict/sechead.xxx.dict')

sechead_invalid_words = strutils.load_lc_str_set('dict/sechead.invalid.dict')

sechead_invalid_heading_set = strutils.load_lc_str_set('dict/sechead.invalid.line')


ALPHA_ONLY_PAT = re.compile(r'^[a-zA-Z]+$')
def is_alpha_word(xst):
    return ALPHA_ONLY_PAT.match(xst)

def contains_invalid_sechead_word(word_list):
    for word in word_list:
        if strutils.is_number(word) and float(word) > 200:
            return True
        elif word in sechead_invalid_words:
            return True
    return False


def is_word_overlap(word_list, word_set, perc=0.6):
    if not word_list:  # empty word list is not overlap
        return False
    word_overlap_count = 0
    for word in word_list:
        if word in word_set:
            word_overlap_count += 1
    return word_overlap_count / len(word_list) >= perc

paren_word_pat = re.compile(r'^\s*\(\s*\S\S?\s*\)(.*)$')

def is_header_cap(line):
    #if "Seller's Representations and Warranties" in line:
    #    print("hello")
    #if "ARTICLE2" in line:
    #    print("hello")
    #if "Term and Termination" in line:
    #    print("hello")

    mat = paren_word_pat.match(line)
    if mat:
        if mat.group(1):
            line = mat.group(1)

    words = [word for word in wordpunct_tokenize(line)]
    alphas = [word for word in words if is_alpha_word(word)]
    not_stopwords = [word for word in alphas if not is_stopword(word.lower())]

    num_stop = len(alphas) - len(not_stopwords)
    # 'Reference to and Effect on the Loan Documents.'
    if num_stop >= 6:
        return False
    # 'to the Company; or'
    if alphas and len(not_stopwords) / len(alphas) < 0.4:
        return False
    num_title = 0
    for word in alphas:
        if word[0].isupper():
            num_title += 1

    if not not_stopwords:
        return False
    return num_title / len(not_stopwords) > 0.6   # 2 out of 3 words


def is_all_single_chars(line):
    # don't take the last word because sometimes it has punctuations
    words = [word for word in line.split()[:-1]]
    if len(words) < 5:
        return False
    for word in words:
        if len(word) > 1:
            return False
    return True


def compact_line(line):
    return re.sub(r'\s+', '', line)


def norm_lcword(line):
    # print("norm_lcword({})".format(line))
    lc_words = [word.lower() for word in wordpunct_tokenize(line)]
    return [lc_word for lc_word in lc_words if not is_stopword(lc_word)]


# ARTICLES ARTICLE 1
def remove_duplicated_first_words(line: str) -> str:
    words = line.split()
    if len(words) >= 2:
        if are_the_same_word(words[0], words[1]):
            return ' '.join(words[1:])
    return line

# ARTICLES ARTICLE 1 -- BASIC TERMS
def are_the_same_word(word1: str, word2: str) -> bool:
    return (word1 == word2 or
            word1[:-1] == word2 or
            word1 == word2[-1])



WORD_WITH_PERIOD_PAT = re.compile(r'(\S+[\.\:]\S*|\S+\s\s\s+(\S))')


def split_subsection_head3(line) -> int:
    linelen = len(line)
    mat = re.search(WORD_WITH_PERIOD_PAT, line)
    # make sure it is at the beginning of the string
    # TODO, jshaw, is_relaxed_number() should handle '1.14b' and other valid
    # sechead numbers
    if mat and (strutils.is_header_number(mat.group()) or
                strutils.is_digit_core(mat.group()[0])) and mat.start() < 30:
        matched_line = line[mat.end():]
        if '.' not in matched_line and \
           ':' not in matched_line and \
           stopwordutils.is_title_non_stopwords(matched_line):
            return -1
        mat2 = re.search(WORD_WITH_PERIOD_PAT, line[mat.end():])
        if mat2:
            # "Distributor     shall xxx"
            if mat2.group(2) and not mat2.group(2)[0].isupper():
                return -1
            period_index = mat.end() + mat2.end()
            # print("period_index %d" % (period_index, ))
            if (mat2.group().endswith('.') or \
                mat2.group().endswith(':') or \
                mat2.group().endswith('   ')) and period_index < 70:
                if period_index < linelen:
                    space_idx = period_index
                    while space_idx < linelen and \
                          strutils.is_space(line[space_idx]):
                        space_idx += 1
                    return space_idx
    return -1


def is_invalid_heading(line: str) -> bool:
    words = line.split()
    # may contain the prefix 6.2
    if len(words) >= 2 and \
       ' '.join(words[1:]).lower() in sechead_invalid_heading_set:
        return True
    return line.lower() in sechead_invalid_heading_set


def is_maybe_sechead_title(line):
    norm_line_words = norm_lcword(line)  # to catch "523 East Weddel" as an address
    # avoid 'patent license agreement', which is in sechead.xxx.xxx
    # When such line is treated as a sechead, it cannot be
    # considered as a 'title' anymore.  So must avoid this.
    if norm_line_words[-1] in set(['agreement', 'contract']):
        return False
    if is_word_overlap(norm_line_words, sechead_xxx_word_set) and \
       not is_invalid_heading(line) and \
       not contains_invalid_sechead_word(norm_line_words):
        return True
    return False


def is_invalid_sechead(unused_sechead_type: str,
                       prefix: str,
                       head: str,
                       unused_split_idx: int):
    # toc
    if '...' in head:
        return True
    if prefix == 'a':   # 'a', 'Force Majeure Event.'
        return True
    words = head.split()
    # 'At the termination of the Transmission Force Majeure Event, the '
    if len(words) >= 8:
        if strutils.is_word_all_lc(words[-1]):
            return True
        # 'xxx shall:'
        if words[-1][-1] == ':' and strutils.is_word_all_lc(words[-1][:-1]):
            return True
    # 'Agreement'
    if (not prefix) and head in set(['Agreement', 'Agreement.']):
        return True
    return False


def extract_sechead(line: str,
                    *,
                    prev_line: str = '',
                    prev_line_idx: int = -1,
                    is_combine_line: bool = True) \
                    -> Optional[Tuple[str, str, str, int]]:
    shead_tuple = extract_sechead_aux(line,
                                      prev_line=prev_line,
                                      prev_line_idx=prev_line_idx,
                                      is_combine_line=is_combine_line)
    if shead_tuple and not is_invalid_sechead(*shead_tuple):
        return shead_tuple
    return None


# assuming prev_line, if set, is the sec
# returns tuple-4, (sechead|sechead-comb, prefix+num, head, split_idx)
# pylint: disable=too-many-locals, too-many-return-statements, too-many-branches, too-many-statements
def extract_sechead_aux(line: str,
                        prev_line: str = '',
                        prev_line_idx: int = -1,
                        is_combine_line: bool = True) \
                        -> Optional[Tuple[str, str, str, int]]:
    if not line:
        return None

    if not is_combine_line:

        # 3 Months
        if is_invalid_heading(line):
            return None

        split_idx = split_subsection_head3(line)
        if split_idx != -1:
            # print("split2: [{}]".format(line[:split_idx]))
            line = line[:split_idx]

        # print("exxx line= [{}]".format(line))

        # print("\txxx\tline\t[{}...]".format(line[:40]))
        prefix, num, head, end_idx = parse_sechead_remove_lastnum(line)

        if not (prefix or head):
            return None
        prefix = ' '.join([prefix, num]).strip()
        return ('sechead', prefix, head, split_idx)


    if not prev_line:
        last_extract_sechead_v4_line = ''
    elif prev_line_idx == -1:
        last_extract_sechead_v4_line = prev_line
    else:
        last_extract_sechead_v4_line = prev_line[:prev_line_idx]

    comb_prefix, comb_num, comb_head, comb_split_idx = '', '', '', -1
    prefix, num, head, split_idx = '', '', '', -1

    lc_line = line.lower()
    if last_extract_sechead_v4_line and \
       not (lc_line.startswith('section') and
            len(last_extract_sechead_v4_line) > 20):

        # for the following 2 cases, not include the prev_line
        # Articles Article 1
        # 3 2.3.2 Section Head.
        lc_words = lc_line[:50].split()
        # we only want the first word
        lc_prev_words = last_extract_sechead_v4_line[:50].lower().split()

        # if (footerutils.classify_line_page_number(last_extract_sechead_v4_line) and
        #    strutils.is_digit_st(line[0])):
        if footerutils.classify_line_page_number(last_extract_sechead_v4_line):
            combined_line = ''
        # Articles Article 1 XXX YYY
        elif len(lc_prev_words) == 1 and are_the_same_word(lc_prev_words[0], lc_words[0]):
            combined_line = line
        # Article 1\nArticle 1 xxx
        # no need to do comb_line
        elif lc_line.startswith(last_extract_sechead_v4_line.lower()):
            combined_line = ''
        else:
            combined_line = ' '.join([last_extract_sechead_v4_line, line])

        if combined_line:
            # print("\txxx\tcombined_line\t[{}...]".format(combined_line[:40]))

            comb_split_idx = split_subsection_head3(combined_line)
            if comb_split_idx != -1:
                # print("split2: [{}]".format(line[:split_idx]))
                combined_line = combined_line[:comb_split_idx]
            if combined_line.strip() != last_extract_sechead_v4_line.strip():
                comb_prefix, comb_num, comb_head, end_idx = \
                    parse_sechead_remove_lastnum(combined_line)

                # if the result is only from prev_line
                # print("lx343: {}".format(len(last_extract_sechead_v4_line)))
                # print("lx343x: [{}]".format(last_extract_sechead_v4_line))
                #print("lx344: {}".format(len(combined_line[:end_idx].strip())))
                # print("lx345: [{}]".format(combined_line[:end_idx]))
                if end_idx >= 0 and \
                   len(combined_line[:end_idx].strip()) <= len(last_extract_sechead_v4_line):
                    comb_prefix, comb_num, comb_head, end_idx = '', '', '', -1


    split_idx = split_subsection_head3(line)
    if split_idx != -1:
        # print("split2: [{}]".format(line[:split_idx]))
        line = line[:split_idx]

    # print("\txxx\tline\t[{}...]".format(line[:40]))
    prefix, num, head, end_idx = parse_sechead_remove_lastnum(line)

    comb_prefix = ' '.join([comb_prefix, comb_num]).strip()
    prefix = ' '.join([prefix, num]).strip()

    if DEBUG_MODE and (comb_prefix or comb_head or prefix or head):
        print("\n\t\tcomb_prefix, comb_head = [{}]\t[{}]".format(comb_prefix, comb_head))
        print("\t\tprefix, head = [{}]\t[{}]".format(prefix, head))
        print("\t\tlast_line: [{}]".format(last_extract_sechead_v4_line[:60]))
        print("\t\tline: [{}]".format(line[:60]))


    # pylint: disable=too-many-boolean-expressions
    if not (comb_prefix or comb_head or prefix or head):
        return None
    # 'Artilce II\nServices',  'Service' didn't match head
    elif comb_prefix and comb_head and not (prefix or head):
        if comb_split_idx >= 0 and comb_split_idx > len(last_extract_sechead_v4_line):
            comb_split_idx -= (len(last_extract_sechead_v4_line) + 1)  # 1 for eoln
        return ('sechead-comb', comb_prefix, comb_head, comb_split_idx)
    # 'Schedule\n2.1'
    elif comb_prefix and not (comb_head or prefix or head):
        return ('sechead-comb', comb_prefix, comb_head, comb_split_idx)
    # Article V\n Assignment, # 'assignment' matched head
    elif comb_prefix and comb_head and not prefix and comb_head == head:
        return ('sechead-comb', comb_prefix, comb_head, comb_split_idx)
    # '1.2 Other Defintion' + 'Term'
    elif comb_prefix and comb_head and not prefix and head in comb_head:
        return ('sechead-comb', comb_prefix, comb_head, split_idx)
    # exhibit 10.2
    # 'agreement', 'recitals'
    elif (not (comb_prefix or comb_head) and head and not prefix) or \
         (not (comb_prefix or comb_head) and prefix and not head) or \
         (not (comb_prefix or comb_head) and (prefix or head)):
        return ('sechead', prefix, head, split_idx)
    # '1.\nRecitals'
    elif comb_prefix and prefix and comb_prefix != prefix and prefix in comb_prefix:
        return ('sechead-comb', comb_prefix, comb_head, -1)
    # 1. Background. xxxx\n2. Description of Service.
    elif comb_prefix and prefix and comb_prefix != prefix:
        return ('sechead', prefix, head, split_idx)
    # this is a little messy, questionable
    # 'Agreement\n1. Recitals.'
    # got comb_prefix=[], comb_head=[Agreement 1. Recitals.], prefix=[Recitals 1.], head = [.]
    elif prefix and comb_head and (not comb_prefix) and len(comb_head) >= len(prefix) + len(head):
        return ('sechead-comb', '', comb_head, -1)
    # 'Defintions\nIn this Agreement'
    # 'Backgroun\nA."
    elif comb_head and not (comb_prefix or prefix or head):
        return ('sechead-comb', '', comb_head, -1)
    # 'Project&\nDevelopment'
    elif head and comb_head and not (comb_prefix or prefix) and head in comb_head:
        return ('sechead-comb', '', comb_head, -1)
    # '1.\nEngagement. xxx'
    elif comb_prefix and comb_head and head and not prefix and comb_head in head:
        return ('sechead-comb', '', comb_head, -1)
    # 'EXHIBITS\nExhibit A'
    elif comb_prefix and not comb_head and prefix and not head and comb_prefix == prefix:
        return ('sechead', prefix, '', -1)
    elif prefix and head and comb_prefix == prefix and comb_head == head:
        return ('sechead', prefix, head, -1)
    # 'Article 1\nArticle 1'
    # comb_prefix = 'Article 1', comb_head= 'Article', prefix = 'Article', head=''
    elif comb_prefix == prefix and prefix and not head and comb_head in prefix:
        return ('sechead', prefix, '', -1)
    # 'Scheduel A to Exhibit "F"\nSchedule A'
    elif comb_prefix == prefix and comb_head and not head:
        return ('sechead', prefix, '', -1)
    # 1 Desk Chair - purple\n1 office Chair -gray'
    # bascially a list of items
    elif (comb_prefix and strutils.is_all_digits(comb_prefix) and
          prefix and strutils.is_all_digits(prefix)):
        pass
    # (1) Desk Chair - purple\n(1) office Chair -gray'
    elif (comb_prefix and strutils.is_parens_all_digits(comb_prefix) and
          prefix and strutils.is_parens_all_digits(prefix)):
        pass
    else:
        pass
        # TODO, jshaw
        # This sould be enabled.  Disabled for now
        # because in production
        # logger.warning('bbbad sechead extraction:')
        # logger.warning('\tprev_line[:60]: [{}]'.format(last_extract_sechead_v4_line[:60]))
        # logger.warning('\tcurr_line[:60]: [{}]'.format(line[:60]))
        # logger.warning('\tcomb_prefix, comb_head = [{}]\t[{}]'.format(comb_prefix, comb_head))
        # logger.warning('\tprefix, head = [{}]\t[{}]'.format(prefix, head))

    return None


# sck, maybe this is not used anymore
def parse_sec_head(line: str, debug_mode: bool = False) \
    -> Tuple[Optional[str], str, str]:
    """
    return (sechead|subsechead|None, prefix, rest)
    In future, we might want to return prefix_num, the exact section number.
    """

    lc_line = line.lower()
    if reject_sechead(lc_line) or \
       not is_header_cap(line) or \
       addrutils.is_address_line(line):
        return None, '', line

    mat = TOC_PREFIX_PAT.search(line)
    if mat:
        return ('toc', '', mat.group(1))

    mat = TOC_PREFIX_2_PAT.search(line)
    if mat:
        return ('toc', '', mat.group(1))

    # handle the case
    # ARTICLES ARTICLE 1 -- BASIC TERMS
    line = remove_duplicated_first_words(line)
    # print("line = {}".format(line))

    mat = SUBHEAD_PREFIX_PAT.match(line)
    mat2 = SUBHEAD_PREFIX_PAT2.match(line)
    if mat2:
        if debug_mode:
            print("matching mat2, subhead_prefix_pat2")
        return ('subsection', mat2.group(1), mat2.group(2))
    # pylint: disable=too-many-nested-blocks
    elif mat:
        # check for just sechead
        secmat = SECHEAD_DIGIT_PAT.match(mat.group(1))
        if secmat:
            if debug_mode:
                print("matching mat, subhead_prefix_pat, sechead_digit_pat")
            return ("sechead", mat.group(1), mat.group(4))
        else:
            if debug_mode:
                print("matching mat, subhead_prefix_pat, NOT sechead_digit_pat")
            return ('subsection', mat.group(1), mat.group(4))
    else:
        # try subhead suffix
        mat = SUBHEAD_SUFFIX_PAT.match(line)

        if mat:
            if debug_mode:
                print("matching mat, subhead_suffix_pat")
            return ('subsection', mat.group(2), mat.group(1))
        else:

            # try sechead prefix
            mat = SECHEAD_PREFIX_PAT.match(line)
            if mat:
                if debug_mode:
                    print("matching mat, NOT subhead_suffix_pat, subhead_prefix_pat")
                norm_words = norm_lcword(mat.group(7))
                norm_line_words = norm_lcword(line)  # to catch "523 East Weddel" as an address

                if is_word_overlap(norm_words, sechead_xxx_word_set) and \
                   ' '.join(norm_words).lower() not in sechead_invalid_heading_set and \
                   not contains_invalid_sechead_word(norm_line_words):
                    return ("sechead", mat.group(1), mat.group(7))
                # 12000 Westheimer Rd, address
                return None, '', line
                # return ("sechead", mat.group(1), mat.group(7))
            else:
                if debug_mode:
                    print("NOT matching mat, no prefix")

                # OK, now no prefix
                # if len(line) < 60:
                if is_header_cap(line):  # this is now redundant
                    norm_words = norm_lcword(line)
                    if is_word_overlap(norm_words, sechead_xxx_word_set):
                        return ("sechead", "", line)
                    elif is_word_overlap(norm_words, subsec_xxx_word_set):
                        return ("subsection", "", line)

                    # handling "W  I  T  N  E  S  S  E  T  H:"
                    if is_all_single_chars(line):
                        comp_line = compact_line(line)
                        # print("cline = {}".format(comp_line))

                        norm_words = norm_lcword(comp_line)
                        if is_word_overlap(norm_words, sechead_xxx_word_set):
                            return ("sechead", "", line)
                        elif is_word_overlap(norm_words, subsec_xxx_word_set):
                            return ("subsection", "", line)


    return None, "", line


def parse_sechead_remove_lastnum(line: str,
                                 debug_mode: bool = False) \
                                 -> Tuple[str, str, str, int]:
    """
    return (sechead|subsechead|None, prefix, rest)
    In future, we might want to return prefix_num, the exact section number.
    """

    prefix_st, num_st, head_st, end_idx = parse_line_aux(line, debug_mode)

    if prefix_st or num_st or head_st:
        new_head_st, new_len = remove_last_num(head_st)
        if new_len >= 0:
            return prefix_st, num_st, new_head_st, end_idx - (len(head_st) - new_len)
        return prefix_st, num_st, head_st, end_idx

    return '', '', '', -1



def classify_sec_head(filename):
    with open(filename, 'rt') as fin:
        for line in fin:
            line = line.strip().replace('/N', ' ')
            print("line: [{}]".format(line))

            cat, head_line = line.split("\t")

            if 'textsub' in cat:
                gold_label = 'subsection'
            else:
                gold_label = 'sechead'

            (guess_label, prefix, head_text) = parse_sec_head(head_line)

            if gold_label == guess_label:
                print("good\t{}\t[{}]\t[{}]\t[{}]".format(guess_label, prefix, head_text, line))
            elif guess_label == 'subsection':
                print("bad1\t{}\t[{}]\t[{}]\t[{}]".format(guess_label, prefix, head_text, line))
            elif guess_label == 'sechead':
                print("bad2\t{}\t[{}]\t[{}]\t[{}]".format(guess_label, prefix, head_text, line))
            else:   # guess_label == None
                print("bad3\t{}\t[{}]\t[{}]\t[{}]".format(guess_label, prefix, head_text, line))


TOP_SEC_PREFIX_PAT = re.compile(r'^(\d+)\.\d+$')
TOP_SEC_PREFIX_PAT2 = re.compile(r'^(\d+)\.?$')
TOP_SEC_PREFIX_PAT3 = re.compile(r'^([ivxm]+)\.?$')
TOP_SEC_PREFIX_PAT4 = re.compile(r'^([A-Z])\.$')


TOP_SEC_PREFIX_SEC_PAT = re.compile(r'^(section|article)\s*(\d+)\.?$', re.IGNORECASE)
TOP_SEC_PREFIX_SEC2_PAT = re.compile(r'^(section|article)\s*(\d+)\.\d+\.?$', re.IGNORECASE)
TOP_SEC_PREFIX_EXH_PAT = re.compile(r'^(exhibit|exmllit)\s*(\S+)\.?$', re.IGNORECASE)



prev_top_sechead_num = 1

def verify_sechead_prefix(line):
    # pylint: disable=global-statement
    global prev_top_sechead_num
    if not line:   # assume it's 'WITNESSETH'
        return True, prev_top_sechead_num

    lc_line = line.lower()
    mat = TOP_SEC_PREFIX_SEC_PAT.match(lc_line) or TOP_SEC_PREFIX_SEC2_PAT.match(lc_line)
    if mat:
        cur_top_sechead_num = int(mat.group(2))
        prev_top_sechead_num = cur_top_sechead_num
        return True, prev_top_sechead_num

    mat = TOP_SEC_PREFIX_EXH_PAT.match(lc_line)
    if mat:
        prev_top_sechead_num = 1
        return True, prev_top_sechead_num

    """
    if (lc_line.startswith('article') or
        lc_line.startswith('exhibit') or
        lc_line.startswith('section')):
        if lc_line.startswith('exhibit'):  # only for exhibit, we reset this
            prev_top_sechead_num = 1
        return True, prev_top_sechead_num
        """

    mat = TOP_SEC_PREFIX_PAT.match(lc_line)
    if mat:
        cur_top_sechead_num = int(mat.group(1))
        diff_top_sechead_num = cur_top_sechead_num - prev_top_sechead_num
        if diff_top_sechead_num < 8:
            prev_top_sechead_num = cur_top_sechead_num
            return True, prev_top_sechead_num

    mat = TOP_SEC_PREFIX_PAT2.match(lc_line)
    if mat:   # we have "34." or "230", which are way off
        cur_top_sechead_num = int(mat.group(1))
        diff_top_sechead_num = cur_top_sechead_num - prev_top_sechead_num
        if diff_top_sechead_num < 8:
            prev_top_sechead_num = cur_top_sechead_num
            return True, prev_top_sechead_num

    mat = TOP_SEC_PREFIX_PAT3.match(lc_line)
    if mat:
        return True, prev_top_sechead_num

    # "U.S.", "P.O. Box"
    if lc_line.startswith('u') or \
       lc_line.startswith('p'):
        return False, prev_top_sechead_num

    # this is line, not lc_line
    mat = TOP_SEC_PREFIX_PAT4.match(line)
    if mat:
        return True, prev_top_sechead_num

    return False, prev_top_sechead_num




HEADER_NUM_PAT = re.compile(r'(\d+)(\.\d+)+')

"""

def find_section_header(lineinfo_list, skip_lineinfo_set):
    sechead_results = []
    prevYStart = -1
    prevPageNum = -1
    for lineinfo in lineinfo_list:
        if not lineinfo in skip_lineinfo_set:
            is_header_num = HEADER_NUM_PAT.match(lineinfo.words[0])
            if is_header_num:
                if prevPageNum != lineinfo.page:
                    prevYStart = -1
                # print("yes sec header: {}".format(lineinfo.text))
                ydiff = lineinfo.yStart - prevYStart
                #print("pageNum= {}, prevYStart = {}, yStart= {}, ydiff={}".format(lineinfo.page,
                #                                                                  prevYStart,
                #                                                                  lineinfo.yStart,
                #                                                                  ydiff))
                if ydiff > 18.0:
                    lineinfo.category = 'sechead'
                    sechead_results.append(lineinfo)
            else:
                #print("maybe sec header: {}".format(lineinfo.text))
                pass
        else:
            # print("skip as sec header: {}".format(lineinfo.text))
            pass
        prevYStart = lineinfo.yStart
        prevPageNum = lineinfo.page
    return sechead_results


def find_section_header2(lineinfo_list, skip_lineinfo_set):
    sechead_results = []
    prevYStart = -1
    prevPageNum = -1
    for lineinfo in lineinfo_list:
        if not lineinfo in skip_lineinfo_set:
            guess_label, prefix, head_text = secheadutils.parse_sec_head(lineinfo.text)
            if guess_label:
                if prevPageNum != lineinfo.page:
                    prevYStart = -1
                # print("yes sec header: {}".format(lineinfo.text))
                ydiff = lineinfo.yStart - prevYStart
                #print("pageNum= {}, prevYStart = {}, yStart= {}, ydiff={}".format(lineinfo.page,
                #                                                                  prevYStart,
                #                                                                  lineinfo.yStart,
                #                                                                  ydiff))
                if ydiff > 18.0:
                    lineinfo.category = guess_label
                    sechead_results.append(lineinfo)
            else:
                #print("maybe sec header: {}".format(lineinfo.text))
                pass
        else:
            # print("skip as sec header: {}".format(lineinfo.text))
            pass
        prevYStart = lineinfo.yStart
        prevPageNum = lineinfo.page
    return sechead_results
"""

"""
# this take "WHEREAS" from beginning of a line, chop it off
# and also merge "ARTICLE 1", "Definitons and Rules of Interpretation"
# into sechead
def xxxfind_section_header(lineinfo_list, skip_lineinfo_set):
    sechead_results = []
    sechead_lineinfo_results = []
    prevYStart = -1
    prevPageNum = -1

    linfo_index = 0
    max_linfo_index = len(lineinfo_list)
    while linfo_index < max_linfo_index:
        lineinfo = lineinfo_list[linfo_index]

        if not lineinfo in skip_lineinfo_set:

            # if lineinfo.text.lower().startswith('article 17'):
            #    print("helllo234")

            #if lineinfo.text == 'EXHIBIT J ':
            #    print("helllo234")
            #if lineinfo.start >= 187595:
            #    print("helllo234")

            if prevPageNum != lineinfo.page:
                prevYStart = -1
            # print("yes sec header: {}".format(lineinfo.text))
            ydiff = lineinfo.yStart - prevYStart

            # it's possible that paragraphs might be out of order "Article 17\nIndemnity",
            # versus "Section 17.01 xxx"
            if lineinfo.is_close_prev_line:
                linfo_index += 1
                prevYStart = lineinfo.yStart
                prevPageNum = lineinfo.page
                continue

            if linfo_index+1 < max_linfo_index:
                next_lineinfo =  lineinfo_list[linfo_index+1]

            if (lineinfo.is_center() and linfo_index+1 < max_linfo_index and
                lineinfo_list[linfo_index+1].is_center()):
                if lineinfo_list[linfo_index+1] not in skip_lineinfo_set:
                    maybe_text = lineinfo.text + '  ' + lineinfo_list[linfo_index+1].text
                    guess_label, prefix, head_text = parse_sec_head(maybe_text)
                    # we don't want '(a)'
                    if guess_label and '(' not in prefix:
                        sechead = SectionHead("sechead",
                                              lineinfo.start,
                                              lineinfo_list[linfo_index + 1].end,
                                              prefix,
                                              head_text,
                                              maybe_text,
                                              [lineinfo, lineinfo_list[linfo_index + 1]])
                        # print("helllo2222 {}".format(sechead))
                        sechead_results.append(sechead)
                        sechead_lineinfo_results.append(lineinfo)
                        sechead_lineinfo_results.append(lineinfo_list[linfo_index + 1])
                        linfo_index += 1  # we already used up one extra
                #else:
                #    print("skipping hhhello: %s" %
                #          lineinfo.text + lineinfo_list[linfo_index+1].text)
            else:   # at this point, we know it is not close to previous line
                guess_label, prefix, head_text = parse_sec_head(lineinfo.text)
                if guess_label and '(' not in prefix:
                    lineinfo.category = guess_label
                    sechead = SectionHead(guess_label,
                                          lineinfo.start,
                                          lineinfo.end,
                                          prefix,
                                          head_text,
                                          lineinfo.text,
                                          [lineinfo])
                    sechead_results.append(sechead)
                    sechead_lineinfo_results.append(lineinfo)
        linfo_index += 1
        prevYStart = lineinfo.yStart
        prevPageNum = lineinfo.page
    return sechead_results, sechead_lineinfo_results
"""

def getLinfoYXstart(lineinfo):
    return lineinfo.yStart, lineinfo.xStart

def is_startswith_exhibit(line):
    lc_text = line.lower()
    return (lc_text.startswith('exhibit') or
            lc_text.startswith('exmllit'))

def mycmp2(x, y):
    if x < y:
        return -1
    if y < x:
        return 1
    return 0


def y_comparator(linfo1, linfo2):
    abs_y_diff = abs(linfo1.yStart - linfo2.yStart)
    if abs_y_diff < lxlineinfo.MAX_Y_DIFF_AS_SAME:
        return mycmp2(linfo1.xStart, linfo2.xStart)
    return mycmp2((linfo1.yStart, linfo1.xStart), (linfo2.yStart, linfo2.xStart))


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'
    # pylint: disable=too-few-public-methods
    class K:
        # pylint: disable=unused-argument
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K


# pylint: disable=too-many-nested-blocks
def find_paged_section_header(paged_lineinfo_list,
                              skip_lineinfo_set):
    sechead_results = []  # type: List
    sechead_lineinfo_results = []  # type: List

    cur_sechead = None
    for page_num, paged_lineinfos in enumerate(paged_lineinfo_list, 1):
        # if page_num == 44:
        #   print("hellolllll")
        paged_lineinfo_list = list(paged_lineinfos)

        # ybased_lineinfo_list = sorted(paged_lineinfo_list, key=getLinfoYXstart)
        ybased_lineinfo_list = sorted(paged_lineinfo_list, key=cmp_to_key(y_comparator))
        linfo_index = 0
        max_linfo_index = len(ybased_lineinfo_list)

        while linfo_index < max_linfo_index:
            lineinfo = ybased_lineinfo_list[linfo_index]

            if lineinfo not in skip_lineinfo_set:

                # if lineinfo.text.lower().startswith('article 17'):
                #    print("helllo234")

                # if lineinfo.text == 'CONTRACT RATE':
                #    print("helllo234")
                #if lineinfo.start >= 237477:
                #    print("helllo234")

                # it's possible that paragraphs might be out of order
                # "Article 17\nIndemnity", versus "Section 17.01 xxx"
                if lineinfo.is_close_prev_line:
                    linfo_index += 1
                    if cur_sechead:
                        cur_sechead.append_lineinfo(lineinfo)
                    continue

                # mainly for debugging purpose
                # if linfo_index+1 < max_linfo_index:
                #     next_lineinfo = ybased_lineinfo_list[linfo_index+1]

                if is_startswith_exhibit(lineinfo.text) and \
                   linfo_index+1 < max_linfo_index and \
                   ybased_lineinfo_list[linfo_index + 1] not in skip_lineinfo_set:

                    maybe_text = lineinfo.text + '  ' + ybased_lineinfo_list[linfo_index + 1].text
                    guess_label, prefix, head_text = parse_sec_head(maybe_text)
                    is_top_sechead, unused_top_sechead_num = verify_sechead_prefix(prefix)
                    # we don't want '(a)'
                    if guess_label and '(' not in prefix and is_top_sechead:
                        if is_startswith_exhibit(prefix):
                            guess_label = 'exhibit'
                        else:
                            guess_label = 'sechead'
                        lineinfo.category = guess_label
                        ybased_lineinfo_list[linfo_index + 1].category = guess_label
                        cur_sechead = SectionHead(guess_label,
                                                  lineinfo.start,
                                                  ybased_lineinfo_list[linfo_index + 1].end,
                                                  page_num,
                                                  prefix,
                                                  head_text,
                                                  maybe_text,
                                                  [lineinfo, ybased_lineinfo_list[linfo_index + 1]])
                        # print("helllo2222 {}".format(sechead))
                        sechead_results.append(cur_sechead)
                        sechead_lineinfo_results.append(lineinfo)
                        sechead_lineinfo_results.append(ybased_lineinfo_list[linfo_index + 1])
                        linfo_index += 1  # we already used up one extra
                    else:
                        if cur_sechead:
                            cur_sechead.append_lineinfo(lineinfo)
                elif lineinfo.is_center() and \
                     linfo_index+1 < max_linfo_index and \
                     ybased_lineinfo_list[linfo_index+1].is_center() and \
                     ybased_lineinfo_list[linfo_index + 1] not in skip_lineinfo_set:

                    maybe_text = lineinfo.text + '  ' + ybased_lineinfo_list[linfo_index + 1].text
                    guess_label, prefix, head_text = parse_sec_head(maybe_text)
                    is_top_sechead, unused_top_sechead_num = verify_sechead_prefix(prefix)
                    # we don't want '(a)'
                    if guess_label and '(' not in prefix and  is_top_sechead:
                        if is_startswith_exhibit(prefix):
                            guess_label = 'exhibit'
                        else:
                            guess_label = 'sechead'
                        lineinfo.category = guess_label
                        ybased_lineinfo_list[linfo_index + 1].category = guess_label
                        cur_sechead = SectionHead(guess_label,
                                                  lineinfo.start,
                                                  ybased_lineinfo_list[linfo_index + 1].end,
                                                  page_num,
                                                  prefix,
                                                  head_text,
                                                  maybe_text,
                                                  [lineinfo, ybased_lineinfo_list[linfo_index + 1]])
                        # print("helllo2222 {}".format(sechead))
                        sechead_results.append(cur_sechead)
                        sechead_lineinfo_results.append(lineinfo)
                        sechead_lineinfo_results.append(ybased_lineinfo_list[linfo_index + 1])
                        linfo_index += 1  # we already used up one extra
                    else:
                        if cur_sechead:
                            cur_sechead.append_lineinfo(lineinfo)

                            #else:
                    #    print("skipping hhhello: %s" %
                    #          lineinfo.text + ybased_lineinfo_list[linfo_index+1].text)

                else:   # at this point, we know it is not close to previous line

                    guess_label, prefix, head_text = parse_sec_head(lineinfo.text)
                    # for "toc", verify_sechead_prefix will fail
                    is_top_sechead, unused_top_sechead_num = verify_sechead_prefix(prefix)
                    if guess_label == 'toc':
                        cur_sechead = SectionHead(guess_label,
                                                  lineinfo.start,
                                                  lineinfo.end,
                                                  page_num,
                                                  prefix,
                                                  head_text,
                                                  lineinfo.text,
                                                  [lineinfo])
                        sechead_results.append(cur_sechead)
                        sechead_lineinfo_results.append(lineinfo)
                    elif guess_label and '(' not in prefix and is_top_sechead:
                        if is_startswith_exhibit(prefix):
                            guess_label = 'exhibit'
                        else:
                            guess_label = 'sechead'
                        lineinfo.category = guess_label
                        cur_sechead = SectionHead(guess_label,
                                                  lineinfo.start,
                                                  lineinfo.end,
                                                  page_num,
                                                  prefix,
                                                  head_text,
                                                  lineinfo.text,
                                                  [lineinfo])
                        sechead_results.append(cur_sechead)
                        sechead_lineinfo_results.append(lineinfo)
                    else:
                        if cur_sechead:
                            cur_sechead.append_lineinfo(lineinfo)
            linfo_index += 1
    return sechead_results, sechead_lineinfo_results


### rewrite of above code

# Section 1.1 XXX
SECHEAD_WORDS = ['section', r's\-', 'article', r'a\-', 'appendix', 'annex',
                 'annexes', 'recitals?', 'exhibit', 'attachment', 'exmllit',
                 r'W\s*I\s*T\s*N\s*E\s*S\s*S\s*E\s*T\s*H', 'whereas',
                 'witnesses', 'schedule',
                ]

# removed 'table of contents?' because it doesn't match the last s???
# removed 'definitions?', maybe similar to above
"""
                 'miscellaneous',
                 'agreement',
                 'notices',
                 'representations and warranties',
                 'confidential',
                 'witness',
                 'covenants?',
                 'governing law',
                 'term',
                 'events of default',
                 'premises',
                 'default',
                 'company',
                 'principal',
                 'other',
                 'remedies',
                 'force majeure',
                 'parties',
                 'waiver',
                 'no default',
                 'project',
                 'no waiver',
                 'conditions',
                 'assignment',
                 'representation',
                 'payments?',
                 'background',
                 'additional provisions',
                 'plant construction company',
                 'compliance with laws?',
                 'compliance with laws and agreements',
                 'commercial lease agreement',
                 'licensee',
                 'use',
                 'license agreement',
                 'loan agreement',
                 'license agreement',
                 'restrictive agreements?',
                 'ownership',
                 'construction',
                 'liquidity events of default and sepcial termination',
]
"""

# NUM_ROMAN_PAT = r'(([\(\“\”]?([\.\d]+|[ivxm\d\.]+|[a-z])[\)\“\”]?))+'
NUM_ROMAN_PAT = r'(([\(\“\”]?([\.\d]+|[ivxm\d]+\b)[\)\“\”\.]?|[\(\“\”\.]?[a-z][\s\.\)\“\”:]|' \
                r'\b[a-z]\b|\b[a-z]\-\d+\b))'
V2_SECHEAD_PAT = r'[\s§]*({})\b\s*[\;\:\.\–\-\—]*\s*{}\s*([\;\:\.\–\-\—\s]*)(.*)$' \
                 .format('|'.join(SECHEAD_WORDS), NUM_ROMAN_PAT)

SECHEAD_NUM_BREAK_CHAR = r'[\;\:\s]'
SECHEAD_NOT_NUM_BREAK_CHARS = r'([^\;\:\–\—\s]+)\s*[\;\:\.\–\—\s]*'

V4_SECHEAD_PAT = r'[\s§]*({})\b\s*[\;\:\.\–\-\—]*\s*{}(.*)$'.format('|'.join(SECHEAD_WORDS),
                                                                    SECHEAD_NOT_NUM_BREAK_CHARS)
V4_SECHEAD_PAT2 = r'[\s§]*{}(.*)$'.format(SECHEAD_NOT_NUM_BREAK_CHARS)

V4_SECHEAD_PAT3 = r'[\s§]*{}({})\b(.*)$'.format(SECHEAD_NOT_NUM_BREAK_CHARS,
                                                '|'.join(SECHEAD_WORDS))

# this is in table of content
# 11 Section 3.1
V2_NUM_SECHEAD_PAT = r'[\s§]*\d+\s*({})\s*[\;\:\.\–\-\—]*\s*{}\s*([\;\:\.\–\-\—\s]*)(.*)$' \
                     .format('|'.join(SECHEAD_WORDS),
                             NUM_ROMAN_PAT)

V2_SECHEAD_PAT2 = r'[\s§]*{}\s*([\;\:\.\–\-\—\s]*)(.*)$'.format(NUM_ROMAN_PAT)

# (5| - 5 - |--) 5.1 XXX
V2_SECHEAD_EXTRA_NUM_PAT2 = r'[\s§]*\d+\s+{}\s*([\;\:\.\–\-\—\s]*)(.*)$'.format(NUM_ROMAN_PAT)

# V2_SECHEAD_PAGE_PAT2 = r'\s*(\-+\s*\d+\s*\-+|\-+)(.*)$'.format(NUM_ROMAN_PAT)
V2_SECHEAD_PAGE_PAT2 = r'\s*(\-+\s*\d+\s*\-+|\-+)(.*)$'


V2_SECHEAD_PAT8 = r'[\s§]*({})\b\s*$'.format('|'.join(SECHEAD_WORDS))


# print('v2_sechead_pat = {}'.format(V2_SECHEAD_PAT))
## print('v2_sechead_pat2 = {}'.format(V2_SECHEAD_PAT2))
#print('v2_sechead_extra_num_pat2 = {}'.format(V2_SECHEAD_EXTRA_NUM_PAT2))
#print('v2_sechead_pat8 = {}'.format(V2_SECHEAD_PAT8))

# print("pat = {}".format(V2_SECHEAD_PAT))
pat = re.compile(V2_SECHEAD_PAT, re.IGNORECASE)
extra_num_sec_pat = re.compile(V2_NUM_SECHEAD_PAT, re.IGNORECASE)

# to handle 'C.3', basically anything with ".\d"
sec_head_pat = re.compile(r'({}\s*|(.*)\.\d(.*))$'.format(NUM_ROMAN_PAT), re.IGNORECASE)

witness_pat = re.compile(r'(w i t n e s s|w i t n e s s e t h)\s*[\:\.]?', re.IGNORECASE)
pat4 = re.compile(V4_SECHEAD_PAT, re.IGNORECASE)
pat2v4 = re.compile(V4_SECHEAD_PAT2, re.IGNORECASE)
pat3v4 = re.compile(V4_SECHEAD_PAT3, re.IGNORECASE)

pat2 = re.compile(V2_SECHEAD_PAT2, re.IGNORECASE)
extra_num_pat2 = re.compile(V2_SECHEAD_EXTRA_NUM_PAT2, re.IGNORECASE)
extra_page_pat2 = re.compile(V2_SECHEAD_PAGE_PAT2)

pat8 = re.compile(V2_SECHEAD_PAT8, re.IGNORECASE)

PAGENUM_PAT3 = re.compile(r'^\s*[a-z]\-\d+', re.IGNORECASE)


if DEBUG_MODE:
    print("v4_sechead_pat2 = {}".format(V4_SECHEAD_PAT2))
    print('pat8 = {}'.format(V2_SECHEAD_PAT8))
    # print("pat2 = {}".format(V2_SECHEAD_PAT2))
    print("sec_head_pat: {}".format(NUM_ROMAN_PAT))


def is_valid_sechead_number(word):
    mat = sec_head_pat.match(word)
    return mat

invalid_sechead_words = set(['follow', 'follows', 'by:', 'page', 'pages',
                             'gals.', 'gal', 'gallons', 'lbs.', 'lbs', 'lb', 'pounds', 'pound',
                             # addresses
                             'floor', 'street', 'center',
                             'attention', 'attn',
                             'esq.', 'esq', 'psc', 'jr.', 'jr', 'sr.', 'sr',
                             'llc', 'l.l.c.', 'inc.', 'corp.', 'corp', 'inc', 'l.l.c'])

# pylint: disable=too-many-branches
def reject_sechead(lc_line):
    lc_line = lc_line.strip()

    if lc_line in set(['company:', 'consultant:', 'by:']):
        return True

    # .22 lbs
    if lc_line.startswith('.'):
        return True

    if lc_line.startswith('$'):
        return True

    if lc_line.startswith('va-') or lc_line.startswith('wa-'):
        return True

    # by now, the line should be chopped with heading only (for subsections)
    # 'article 32 language, effectivenss of contract and miscellaneous provisions'
    if len(lc_line) > 90:
        return True

    if lc_line.endswith(','):
        return True

    # xxx inc.
    if lc_line.endswith(' inc.'):
        return True

    if '___' in lc_line:
        return True

    #mat = TOC_PREFIX_PAT.match(lc_line)
    #if mat:
    #    return True

    if lc_line.startswith('account') or lc_line.startswith('aba '):
        return True

    if '%' in lc_line:
        return True

    if lc_line.startswith('at the premise'):
        return True

    # By: Witness:
    if lc_line.startswith('by'):
        return True

    if lc_line.startswith('c/o'):
        return True

    mat = PAGENUM_PAT3.match(lc_line)
    if mat:
        return True

    # ??  Maybe just bad HTML
    if lc_line.startswith('of'):
        return True

    if lc_line.startswith('party'):
        return True

    if lc_line.startswith('/s'):
        return True

    # T =Tenant
    if '=' in lc_line:
        return True

    # Broker’s License No.:
    if 'no.' in lc_line:
        return True

    if strutils.has_likely_phone_number(lc_line):
        return True

    if strutils.count_numbers(lc_line) >= 4:
        return True

    # re.split(r'[\s\,\'\"]+', lc_line)  # lc_line.split()
    words = strutils.split_words(lc_line)
    if words:
        # no handling 'A G R E E M E N T'
        # 'D E S I G N & D E V E L O P M E N T'
        if strutils.is_all_length_1_words(words):
            return True
        if strutils.is_number(words[0]) and float(words[0]) > 100:
            return True
        if strutils.is_dashed_big_number_st(words[0]):
            return True

        if words[0] == 'i' and words[1] == 'am':
            return True

        if words[-1] in set(['and', 'or', 'to']):
            return True
        for word in words:
            if word in invalid_sechead_words:
                return True
            if word.startswith('$'):
                return True

        # '6. I.', which is really '6.1.'
        # 'background a.'
        if len(words) == 2 and \
           not strutils.is_all_alphas_dot(words[0]) and \
           len(words[1]) == 2 and words[1][-1] == '.':
            return True

        if len(words) >= 2:
            # a nevada corporation
            if words[0] == 'a':
                for word in words[1:]:
                    if word in set(['corporation', 'incorporated', 'corp.', 'inc.']):
                        return True
            if words[0] == 'o':
                return True

        # 2 May 2012
        for word in words[:3]:
            if strutils.is_all_digits(word) and (int(word) > 1900 and int(word) <= 3000):
                return True
        # often in title, ok to not detect them as section heading
        # '(e) regulation s.'
        if words[-1] in set(['for', 'of', 'u.s.']):
            return True
    return False


LAST_NUM_PAT = re.compile(r'^(.*)\s+\d+\s*$')

def remove_last_num(line):
    mat = LAST_NUM_PAT.match(line)
    if mat:
        return mat.group(1), mat.end(1)
    return line, -1

def parse_line_v3(line, debug_mode=False):
    prefix_st, num_st, head_st, end_idx = parse_line_aux(line, debug_mode)

    # sometime a heading might end with a number, probably TOC
    if head_st:
        new_head_st, new_len = remove_last_num(head_st)
        if new_len >= 0:
            return prefix_st, num_st, new_head_st, end_idx - (len(head_st) - new_len)
        return prefix_st, num_st, head_st, end_idx

    # not found, simply return the *_aux() result
    return prefix_st, num_st, head_st, end_idx


# returns prefix_st, num_st, head_st, end_idx
def parse_line_aux(line, debug_mode=False):
    lc_line = line.lower()

    mat = TOC_PREFIX_PAT.match(line)
    if mat:
        return 'toc', '', mat.group(1), -1

    mat = TOC_PREFIX_2_PAT.search(line)
    if mat:
        return 'toc', '', mat.group(1), -1

    if reject_sechead(lc_line) or not is_header_cap(line):
        return '', '', '', -1

    # 11 Section 1.1 XXX, probably in TOC
    """
    mat = extra_num_sec_pat.match(line)
    if mat:
        if mat.group(2):
            last_char = mat.group(2)[-1]
            space_after_num = mat.group(5)
            if (last_char.lower() in set(['i', 'x', 'v']) and not space_after_num and
                not strutils.is_roman_number(mat.group(2))):
                prefix_st = mat.group(1)
                num_st = mat.group(2)[:-1]
                head_st = mat.group(2)[-1] + mat.group(6)
                end_idx = mat.end(6)
                print("**prefix2=[{}]\tnum=[{}]\thead=[{}]".format(mat.group(1), num_st, head_st))
            else:
                prefix_st = mat.group(1)
                num_st = mat.group(2)
                head_st = mat.group(6)
                end_idx = mat.end(6)
                print("prefix=[{}]\tnum=[{}]\tspc=[{}]\thead=[{}]".format(mat.group(1),
                                                                          mat.group(2),
                                                                          mat.group(5),
                                                                          mat.group(6)))
        # goto next line
        return prefix_st, num_st, head_st, end_idx
    """

    # Section 1.1 XXX
    """
    mat = pat2.match(line)
    if mat:
        print("33line: {}".format(line))
        print("group2: {}".format(mat.group(2)))

        if mat.group(2):
            last_char = mat.group(2)[-1]
            space_after_num = mat.group(5)
            if (last_char.lower() in set(['i', 'x', 'v']) and not space_after_num and
                not strutils.is_roman_number(mat.group(2))):
                prefix_st = mat.group(1)
                num_st = mat.group(2)[:-1]
                head_st = mat.group(2)[-1] + mat.group(6)
                if debug_mode:
                    print("**prefix2=[{}]\tnum=[{}]\thead=[{}]".format(mat.group(1),
                                                                       num_st, head_st))
            else:
                prefix_st = mat.group(1)
                num_st = mat.group(2)
                head_st = mat.group(6)
                if debug_mode:
                    print("prefix=[{}]\tnum=[{}]\tspc=[{}]\thead=[{}]".format(mat.group(1),
                                                                              mat.group(2),
                                                                              mat.group(5),
                                                                              mat.group(6)))
        return prefix_st, num_st, head_st
"""
    mat = witness_pat.match(line)
    if mat:
        head_st = line.replace(' ', '')
        return '', '', head_st, -1

    mat = pat4.match(line)
    #if mat:
    #    for i, mygroup in enumerate(mat.groups()):
    #        print("xgroup[{}] = [{}]".format(i, mygroup))
    if mat and is_valid_sechead_number(mat.group(2)):
        prefix_st = mat.group(1)
        num_st = mat.group(2)
        head_st = mat.group(3)
        end_idx = mat.end(3)
        return prefix_st, num_st, head_st, end_idx

    # H Exhibit, must before pat2v4.match below
    mat = pat3v4.match(line)
    #if mat:
    #    for i, mygroup in enumerate(mat.groups()):
    #        print("ygroup[{}] = [{}]".format(i, mygroup))
    if mat and is_valid_sechead_number(mat.group(1)):
        num_st = mat.group(1)
        prefix = mat.group(2)
        head_st = mat.group(3)
        end_idx = mat.end(3)
        if debug_mode:
            print("num=[{}]\tprefix=[{}]\thead=[{}]".format(mat.group(1),
                                                            mat.group(2),
                                                            mat.group(3)))
        return prefix, num_st, head_st, end_idx

    # print("jjjj2v4 here: [{}]".format(line))
    mat = pat2v4.match(line)
    if mat and is_valid_sechead_number(mat.group(1)):
        num_st = mat.group(1)
        head_st = mat.group(2)
        end_idx = mat.end(2)
        if debug_mode:
            print("prefix=[{}]\thead=[{}]".format(mat.group(1), mat.group(2)))
        return '', num_st, head_st, end_idx

    # remove this for now, sunday night
    """
    mat = extra_page_pat2.match(line)
    if mat:
        if debug_mode:
            print("prefix4=[{}]\tspc=[{}]\thead=[{}]".format('', '', mat.group(2)))
        return '', '', mat.group(2), mat.end(2)


    mat = extra_num_pat2.match(line)
    if mat:
        prefix_st = mat.group(1)
        head_st = mat.group(5)
        if debug_mode:
            print("prefix5=[{}]\tspc=[{}]\thead=[{}]".format(mat.group(1),
                                                             mat.group(4), mat.group(5)))
        return prefix_st, '', head_st, mat.end(5)
"""

    """
    mat = pat2.match(line)
    if mat:
        prefix_st = mat.group(1)
        head_st = mat.group(5)
        if debug_mode:
            print("prefix6=[{}]\tspc=[{}]\thead=[{}]".format(mat.group(1),
                                                             mat.group(4), mat.group(5)))
        return prefix_st, '', head_st
"""
    # print("jjjj8 here: [{}]".format(line))
    mat = pat8.match(line)
    if mat:
        prefix_st = mat.group()
        return prefix_st, '', '', mat.end()

    # print("jjjjj_maybe here: [{}]".format(line))
    if is_maybe_sechead_title(line):
        head_st = line
        return '', '', head_st, len(line)

    # print("jjjjjout here: [{}]".format(line))
    if debug_mode:
        print("not match: [{}]".format(line))
    return '', '', '', -1


def st_sechead_str(xst):
    xst = re.sub(r'[\[\]”“"\.:\-\(\)–;,…/]', ' ', xst)
    xst = re.sub(r"['’][sS]\b", '', xst)
    xst = re.sub(r"[sS]['’]", 's', xst)
    xst = re.sub(r"\s+", ' ', xst)

    words = stopwordutils.str_to_tokens(xst, mode=0, is_remove_punct=True)
    words = stopwordutils.tokens_remove_stopwords(words)

    return ' '.join(words)


line_sechead_prefix_pat_only = re.compile(r'^\s*\(?([\d\.]+|[a-zA-Z])\)?\s*$')


def is_line_sechead_prefix_only(line: str):
    return line_sechead_prefix_pat_only.match(line)


# a) xxx
# 1.2) xxx
SECHEAD_PREFIX_PAT1 = r'(\(?([\d\.]+|[a-zA-Z])\))\s*\w'

# exhibit xxx
# exmlxxx xxx
# article xxx
# exhibitc    # because of ocr error
# SECHEAD_PREFIX_PAT2 = r'(exhibit|exmllit|exml|section|article)\s*\w'
SECHEAD_PREFIX_PAT2 = r'(exhibit|exmllit|exml|section|article)'

# 1.2 xxx
# a. xxx
# 1. xxx
SECHEAD_PREFIX_PAT3 = r'((\d+\.|\d+\.\d+|\d+\.\d+\.\d+|\d+\.\d+\.\d+\.\d+)\.?|[a-zA-Z]\. )\s*\w'

SECHEAD_PREFIX_LIST = (SECHEAD_PREFIX_PAT1,
                       SECHEAD_PREFIX_PAT2,
                       SECHEAD_PREFIX_PAT3)

line_sechead_prefix_pat = re.compile(r'^\s*({})'.format('|'.join(SECHEAD_PREFIX_LIST)), re.I)

def is_line_sechead_prefix(line: str):
    return line_sechead_prefix_pat.match(line)


# cannot have 1. xxx, must have at least 2 digit sequences
# 1.2 xx
SECHEAD_PREFIX_STRICT_PAT3 = r'((\d+\.\d+|\d+\.\d+\.\d+|\d+\.\d+\.\d+\.\d+)\.?|[a-zA-Z]\. )\s*\w'

line_sechead_strict_prefix_pat = re.compile(r'^\s*{}'.format(SECHEAD_PREFIX_STRICT_PAT3), re.I)

def is_line_sechead_strict_prefix(line: str):
    return line_sechead_strict_prefix_pat.match(line)


def main():
    parser = argparse.ArgumentParser(description='Parse a document into a document structure.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('doc', help='a file to be annotated', default='sechead.list.txt.sorted')

    # args = parser.parse_args()
    doc_fn = 'sechead.list.txt.sorted'

    classify_sec_head(doc_fn)
    # x = parse_sec_head("W  I  T  N  E  S  S  E  T  H:")
    # x = parse_sec_head("License of Patent Pending Applications")
    # x = parse_sec_head("THIS AGREEMENT WITNESSES THAT")
    # x = parse_sec_head("R E C I T A L S")
    # x = parse_sec_head("No Obligation to Prosecute or Maintain the Patents and Trademarks.")
    # x = parse_sec_head("10. (a)  Special Agreement.")
    # print("x = {}".format(x))

    # page_num_list = adoc.get_page_numbers()
    # atext = adoc.get_text()
    # for i, page_num in enumerate(page_num_list):
    #    print("page num #{}: {}".format(i, page_num))

    # docreader.format_document(adoc, sentV2_list)
    logger.info('Done.')


if __name__ == '__main__':
    main()
