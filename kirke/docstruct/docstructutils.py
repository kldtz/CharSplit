import re

from kirke.utils import strutils
from kirke.docstruct import secheadutils

def is_line_centered(line, xStart, xEnd, is_relax_check=False):
    
    if len(line) > 65:
        return False
    if strutils.is_all_caps_space(line) and len(line) > 52:
        return False
    
    right_diff = 612 - xEnd   # (0, 0, 595, 842);
    left_diff = xStart

    # print("left_diff = {}, right_diff= {}, diff = {}".
    # format(left_diff, right_diff, abs(right_diff - left_diff)))
    # print("text = {}".format(self.text))
    if left_diff > 100 and abs(right_diff - left_diff) < 18:
        return True

    # there are some short lines that are not really centered
    if (xEnd - xStart < 100 and
        left_diff > 100 and abs(right_diff - left_diff) < 80):
        return True

    # there are some short lines that are not really centered
    if (xEnd - xStart < 100 and
        left_diff > 100 and abs(right_diff - left_diff) < 80):
        return True

    if (xEnd - xStart < 340 and
        left_diff > 100 and abs(right_diff - left_diff) < 50):
        return True

    return False



# should not check for eoln because bad OCR or other extra text
# 'TABLE OF CONTENTS OFFICE LEASE'
TOC_PREFIX_PAT = re.compile(r'^\s*(table\s*of\s*contents?|contents?)\s*:?', re.IGNORECASE)
TOC_PREFIX_2_PAT = re.compile(r'^(.+)\.{5}')


def is_line_toc(line):
    mat = TOC_PREFIX_PAT.search(line)
    if mat:
        return True

    mat = TOC_PREFIX_2_PAT.search(line)
    if mat:
        return True
    return False



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


def is_line_page_num(line: str, is_centered: bool):
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


def is_line_footer(line: str,
                   page_line_num: int,
                   num_line_in_page: int,
                   lbk: float,
                   page_num_index: int,
                   is_english: bool,
                   align: str,
                   yStart: float):
    if yStart < 700.0:
        return False, -1.0
    score = 0
    if yStart >= 725.0:
        score += 0.4
    if num_line_in_page - page_line_num <= 2:
        score += 0.5
    if not is_english:
        score += 0.2
    if len(line) < 30:
        score += 0.2
    if lbk >= 2.0:
        score += 0.2
    if page_num_index != -1 and page_line_num >= page_num_index:
        score += 0.8
    # print('is_footer.score = {}'.format(score))
    return score >= 1, score


# returns tuple-4, (sechead|sechead-comb, prefix+num, head, split_idx)
# def extract_sechead_v4(line: str,

def is_invalid_sechead(sechead, prefix, head, split_idx):
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


def extract_line_sechead(line: str, prev_line: str):
    # sechead, prefix, head, split_idx = secheadutils.extract_sechead_v4(line, is_combine_line=False)
    sechead, prefix, head, split_idx = secheadutils.extract_sechead_v4(line, is_combine_line=True)
    if sechead:
        if not is_invalid_sechead(sechead, prefix, head, split_idx):
            return sechead, prefix, head, split_idx
    return False

