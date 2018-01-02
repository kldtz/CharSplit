from collections import Counter
import itertools
import re
from typing import List, Match, Set

from kirke.docstruct import lxlineinfo
from kirke.utils import strutils


# [\dl] with "l" is for "1", but ocr sometimes mistaken l
PAGENUM_PAT = re.compile(r'^\s*(\bpages?)?\(\s*?\-*(\d+|[ivxm]+|[A-Z]-[\dl]+)\-*\s*\)?\s*$', re.IGNORECASE)


PAGENUM_SIMPLE1_PAT = re.compile(r'^\s*(\-*\s*\d+\s*\-*)\s*$')

# ( i )
# PAGENUM_PAT2 = re.compile(r'^\s*(\bpages?)?\-*(\d+|[ivxm]+|[A-Z]-[\dl]+)\-*\s*$', re.IGNORECASE)

# too permissive, matched (Signature page follows)
PAGENUM_PAT2 = re.compile(r'^\s*(.*)(\bpage )\-*(\d+|[ivxm]+|[A-Z]-[\dl]+)\-*\s*$', re.IGNORECASE)

# "a-l", but don't want '234-233'
PAGENUM_PAT3 = re.compile(r'^\s*(\S+)\-[\dl]+\s*$')

# "-iv-"
PAGENUM_PAT4 = re.compile(r'^\s*\-?[ivxm]+\-?\s*$', re.IGNORECASE)

# "page 3 of 4"
PAGENUM_PAT5 = re.compile(r'^\s*pages?\s*\d+\s*of\s*\d+\s*$', re.IGNORECASE)

# borrowed from secheadutils.py
SECHEAD_PREFIX_PAT = re.compile(r'(article|appendix|section|exhibit|recital)\s*\d+\s*', re.IGNORECASE)

def get_first_last_n_elements(alist: List, n=1) -> List:
    # only add new lineinfo, sometimes a page can have less than 3 lines
    result = []
    seen_set = set([])  # type: Set
    for elmt in itertools.chain(alist[:n],
                                alist[-n:]):
        if elmt not in seen_set:
            result.append(elmt)
            seen_set.add(elmt)
    return result


def get_first_last_n_prev_infos(alist: List, lineinfo_list) -> List:
    # only add new lineinfo, sometimes a page can have less than 3 lines
    sent_seq_list = [linfo.sid for linfo in alist]

    prev_sent_list = []
    for sent_seq in sent_seq_list:
        if sent_seq - 1 > 0:
            prev_sent_list.append(alist[sent_seq -1])
        else:
            prev_sent_list.append(None)

    return prev_sent_list


def classify_line_page_number(line: str):
    if PAGENUM_SIMPLE1_PAT.match(line):
        # print("pagenumber x1: {}".format(line))
        return True
    # if is_center_lineinfo(lineinfo):
    # print("LINE is CENTERED")
    if PAGENUM_PAT.match(line):
        # print("pagenumber x2: {}".format(line))
        return True

    # Exhibit K -Page 2
    if PAGENUM_PAT2.match(line):
        # print("pagenumber x3: {}".format(line))
        return True

    mat = PAGENUM_PAT3.match(line)
    if mat:
        words = strutils.split_words(line.replace('l', '1'))
        num_digit = 0
        for word in words:
            if strutils.is_all_digits(word):
                if int(word) > 20:
                    return False
                num_digit += 1
        if num_digit > 2:
            return False
        # print("pagenumber x3: {}".format(line))
        return True

    if PAGENUM_PAT4.match(line):
        # print("pagenumber x3: {}".format(line))
        return True

    # page 4 of 5
    if PAGENUM_PAT5.match(line):
        # print("pagenumber x3: {}".format(line))
        return True    
    
    return False

def is_sechead(line: str) -> Match[str]:
    return SECHEAD_PREFIX_PAT.match(line)

"""
tmpst = 'Exhibit K -Page 2'
print("classify_line_page_number({}) = {}".format(tmpst, classify_line_page_number(tmpst)))
tmpst = 'Page 2'
print("classify_line_page_number({}) = {}".format(tmpst, classify_line_page_number(tmpst)))
tmpst = '( ii )'
print("classify_line_page_number({}) = {}".format(tmpst, classify_line_page_number(tmpst)))
tmpst = 'article14'
print("is_seaction_head({}) = {}".format(tmpst, is_sechead(tmpst)))
"""


def find_pagenum_footer(page_lineinfos_list):
    num_pages = len(page_lineinfos_list)

    pagenum_results = []
    word_counter = Counter()
    page_num_candidates = []
    
    # only take the first 3 and last 3 sentence of a page
    for page_lineinfo_list in page_lineinfos_list:
        # only add new lineinfo, sometimes a page can have less than 3 lines
        first_last_n_infos = get_first_last_n_elements(page_lineinfo_list, 3)
        # first_last_n_prev_infos = get_first_last_n_prev_infos(first_last_n_infos, page_lineinfo_list)

        xpage_pagenum_results = []
        for lineinfo in first_last_n_infos:
            if lxlineinfo.is_short_sent_by_length(lineinfo.length):
                # print('short line #{} len={}: {}'.format(lineinfo.sid, lineinfo.length, lineinfo.text))
                # for word in lineinfo.words:
                #    print("    word {}".format(word))
                word_counter.update(lineinfo.words)
                page_num_candidates.append(lineinfo)

                if classify_line_page_number(lineinfo.text):
                    xpage_pagenum_results.append(lineinfo)
        if len(xpage_pagenum_results) > 1:
            xpage_pagenum = xpage_pagenum_results[-1]
        elif len(xpage_pagenum_results) == 1:
            xpage_pagenum = xpage_pagenum_results[0]
        else:
            xpage_pagenum = None

        if xpage_pagenum:
            xpage_pagenum.category = 'pagenum'
            pagenum_results.append(xpage_pagenum)

# can remove this once more comfortable            
        """
def find_pagenum_footer(page_lineinfos_list):
    num_pages = len(page_lineinfos_list)

    pagenum_results = []
    lineinfo_list = []
    # only take the first 3 and last 3 sentence of a page
    for page_lineinfo_list in page_lineinfos_list:
        # only add new lineinfo, sometimes a page can have less than 3 lines
        lineinfo_list.extend(get_first_last_n_elements(page_lineinfo_list, 3))
                
    word_counter = Counter()
    page_num_candidates = []
    for lineinfo in lineinfo_list:
        if lxlineinfo.is_short_sent_by_length(lineinfo.length):
            # print('short line #{} len={}: {}'.format(lineinfo.sid, lineinfo.length, lineinfo.text))
            # for word in lineinfo.words:
            #    print("    word {}".format(word))
            word_counter.update(lineinfo.words)
            page_num_candidates.append(lineinfo)

            if classify_line_page_number(lineinfo.text):
                lineinfo.category = 'pagenum'
                pagenum_results.append(lineinfo)
"""
    pagenum_lineinfo_set = set(pagenum_results)

    MIN_FOOTER_FREQ = num_pages / 2
    freq_word_10 = []
    for word, freq in word_counter.most_common(10):
        # print("({}), freq={}".format(word, freq))
        # jshaw, TODO, weird, need to check for word of size 0???
        if (word and not strutils.is_punct(word[0]) and
            word not in ['not', 'an'] and
            freq >= MIN_FOOTER_FREQ):
            freq_word_10.append(word)
    #print("num_pages", num_pages)
    #print("top 10")
    #print(freq_word_10)
    #print(word_counter.most_common(10))

    freq_word_10_set = set(freq_word_10)

    footer_results = []
    for lineinfo in page_num_candidates:

        # footer and pagenumber are separate
        if lineinfo in pagenum_lineinfo_set:
            continue
        
        count_freq_word = 0
        lineinfo_char_count = 0
        char_overlap_count = 0
        num_both_alphanum_word = 0
        num_alpha_word = 0
        for word in lineinfo.words:
            lineinfo_char_count += len(word)
            if word in freq_word_10_set:
                count_freq_word += 1
                char_overlap_count += len(word)
            if (strutils.is_both_alpha_and_num(word) or
                strutils.has_digit(word) or
                strutils.is_all_punct(word)):
                num_both_alphanum_word += 1
            if strutils.is_alpha_word(word):
                num_alpha_word += 1

        # '7%', part of a table
        if (strutils.is_num_perc(lineinfo.text) or
            strutils.is_num_period(lineinfo.text) or
            strutils.is_dollar_num(lineinfo.text)):
            continue
        if is_sechead(lineinfo.text):
            continue        
        
        # perc = count_freq_word / lineinfo.length
        perc = 0
        if lineinfo_char_count:  # to avoid division by zero
            perc = char_overlap_count / lineinfo_char_count
        # print("checking lineinfo #{}, perc= {}, {}".format(lineinfo.sid, perc, lineinfo.text))
        if perc >= 0.30:
            # print("maybe page num ({:.4f}%): {}".format(perc, lineinfo.text))
            # pnum_lineinfost_list.append(lineinfo.text)
            # result.append(EbPageNumber(lineinfo.start, lineinfo.end, "-1", lineinfo.text))
            lineinfo.category = 'footer'
            footer_results.append(lineinfo)
        # "1. Contract", "assignee.
        if num_alpha_word > 0:
            continue
        # "584342v2"
        # make sure it is not lineinfo.words is not empty
        if lineinfo.words and num_both_alphanum_word / len(lineinfo.words) > 0.3:
            lineinfo.category = 'footer'
            footer_results.append(lineinfo)

    # check if the footers found are valid footers
    tmp_footer_results = [] 
    for linfo in footer_results:
        if '%' in linfo.text:  # remove any "22%", probably from tables
            continue
        tmp_footer_results.append(linfo)
    footer_results = tmp_footer_results
    # print('footer_st_tmpx1_list = {}'.format(set([lineinfo.text for lineinfo in footer_results])))
    # print('len(page_lineinfos_list) = {}'.format(len(page_lineinfos_list)))
    if len(footer_results) / len(page_lineinfos_list) < 0.1:
        for linfo in footer_results:
            linfo.category = None
        footer_results = []

    for page_linfos in page_lineinfos_list:
        if len(page_linfos) > 2:
            last_linfo = page_linfos[-1]
            next_last_linfo = page_linfos[-2]

            if abs(next_last_linfo.yStart - last_linfo.yStart) < lxlineinfo.MAX_Y_DIFF_AS_SAME:

                # print("last_linfo: {}".format(last_linfo))
                # print("next_last_linfo: {}".format(next_last_linfo))

                if next_last_linfo.category == 'pagenum' and last_linfo.is_english == False:
                    last_linfo.category = 'footer'
                    footer_results.append(last_linfo)
                elif last_linfo.category == 'pagenum' and next_last_linfo.is_english == False:
                    next_last_linfo.category = 'footer'
                    footer_results.append(next_last_linfo)                
        
    # there is no guarantee that the footer will be the last 3 line because pdf transformer
    # can reorder lines, especially lines related to address or signature might interfere.
    footer_st_set = set([lineinfo.text for lineinfo in footer_results])
    footer_lineinfo_set = set(footer_results)
    # print('footer_st_list = {}'.format(footer_st_set))
    for page_lineinfo_list in page_lineinfos_list:    
        for lineinfo in page_lineinfo_list:
            if lineinfo not in footer_lineinfo_set:
                if lineinfo.text in footer_st_set:  # add new one
                    # print("xxxxxxxxxxxxxxxxxx  adding new footer: {}".format(lineinfo))
                    lineinfo.category = 'footer'
                    footer_results.append(lineinfo)

    # probably should sort footer_results here
    footer_results = sorted(footer_results)
            
    #if are_words_numbers([lineinfo.words[0] for lineinfo in footer_lineinfo_list], 0.6):
    #    for pnum, lineinfo in zip(result, footer_lineinfo_list):
    #        pnum.page_number = lineinfo.words[0]
    #        footer_results.append(lineinfo)
    return pagenum_results, footer_results
