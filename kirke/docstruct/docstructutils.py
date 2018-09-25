import re
# pylint: disable=unused-import
from typing import Dict, List, Match, Optional, Tuple


from kirke.docstruct import docutils, linepos, secheadutils
from kirke.docstruct.docutils import PLineAttrs
from kirke.utils import corenlpsent, engutils, mathutils, stopwordutils, strutils

# from kirke.ebrules import addresses

# TODO, jshaw
# The header strings in here definitely is somewhat cheating, for AT&T
# But there is no easy to distinguish between form-based header,
# such as
#     AT&T VPN SERVICE
#     PRICING ADDENDUM
# and
# other normal section headings, such as "Article" and more obvious centered
# title.  Even a page title.


def is_line_title(line: str) -> bool:
    words = stopwordutils.get_nonstopwords_nolc_gt_len1(line)
    has_alpha_word = False
    for word in words:
        has_alpha_word = True
        if not word[0].isupper():
            return False
    return has_alpha_word


GLOBAL_PAGE_HEADER_SET = strutils.load_lc_str_set('resources/pageheader.txt')


# seems to be only called by pdfoffsets.py
# pylint: disable=invalid-name, too-many-return-statements
def is_line_centered(line: str,
                     xStart,  # is this int or float?
                     xEnd):

    if len(line) > 65:
        return False
    if strutils.is_all_caps_space(line) and len(line) > 52:
        return False
    # cendered but all lowercase doesn't mean it is a heading
    if strutils.is_all_lower(line) and len(line) > 52:
        return False

    right_diff = 612 - xEnd   # (0, 0, 595, 842);
    left_diff = xStart

    # print("left_diff = {}, right_diff= {}, diff = {}".
    # format(left_diff, right_diff, abs(right_diff - left_diff)))
    # print("text = {}".format(self.text))
    if left_diff > 100 and abs(right_diff - left_diff) < 18:
        return True

    # there are some short lines that are not really centered
    if xEnd - xStart < 100 and \
       left_diff > 100 and abs(right_diff - left_diff) < 80:
        return True

    # there are some short lines that are not really centered
    if xEnd - xStart < 100 and \
       left_diff > 100 and abs(right_diff - left_diff) < 80:
        return True

    if xEnd - xStart < 340 and \
       left_diff > 100 and abs(right_diff - left_diff) < 50:
        return True

    return False



# should not check for eoln because bad OCR or other extra text
# 'TABLE OF CONTENTS OFFICE LEASE'
TOC_HEADING_PAT = re.compile(r'^\s*(table\s*of\s*contents?|contents?)\s*:?$', re.IGNORECASE)
# 5 periods
TOC_PREFIX_2_PAT = re.compile(r'(\.\.\.\.\.|\. \. \. \. \. )')

def is_line_toc_heading(line: str) -> bool:
    mat = TOC_HEADING_PAT.search(line)
    return bool(mat and mat.end() < 40)


def is_line_toc(line: str) -> bool:
    # sometimes a signature line might look like toc
    # By: .,.....
    if is_line_signature_prefix(line):
        return False

    mat = TOC_HEADING_PAT.search(line)
    if mat:
        return True

    mat = TOC_PREFIX_2_PAT.search(line)
    if mat:
        # first verify that the number of words is less than 12
        no_multi_period_line = re.sub(r'\.\s*[\.\s]+', ' ', line)
        num_words = no_multi_period_line.split()
        if len(num_words) >= 12:
            # check if there are 3 or more of the multi-period frags.
            # some TOC lines might have been stuck together on the
            # same line or block

            # if there is OCR error on unknown stuff, it might form
            # multi-period frags.  If known as a sentence, don't bother.
            multi_period_mat = re.search(TOC_PREFIX_2_PAT, line)
            if multi_period_mat and multi_period_mat.start() > 70:
                return False
            return True

        # need to verify it again for the rest of the line
        line_left = line[mat.end(1):]
        # print('line_left = [{}]'.format(line_left))
        return bool(TOC_PREFIX_2_PAT.search(line_left))
    return False



# [\dl] with "l" is for "1", but ocr sometimes mistaken l
PAGENUM_PAT = re.compile(r'^\s*(\bpages?)?\(\s*?\-*(\d+|[ivxm]+|[A-Z]-[\dl]+)\-*\s*\)?\s*$',
                         re.IGNORECASE)

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


# line_num_in_page is 1-based
# pylint: disable=too-many-return-statements, too-many-branches, too-many-arguments
def is_line_page_num(line: str,
                     line_num_in_page: int = 1,
                     num_line_in_page: int = 20,
                     line_break: float = 6.0,
                     yStart: float = 700.0,
                     unused_is_centered: bool = False):
    if line_break > 5.0 or yStart >= 675.0:  # seen yStart==687.6 as page number
        pass
    elif line_num_in_page > 2 and line_num_in_page <= num_line_in_page - 2:
        return False

    # no sechead in page number, if it is obvious sechead
    if secheadutils.is_line_sechead_strict_prefix(line):
        return False

    # 'page' in toc header
    if line.lower() == 'page' or PAGENUM_SIMPLE1_PAT.match(line):
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


IGNORE_LINE_LIST = [r'at&t and customer confidential information',
                    r'asap!',
                    r'page\s*\d+\s*of\s*\d+',
                    # this is the first|last page of
                    r'this is the \S+\s*page of.*']

ATT_PAGE_NUM_PAT = re.compile(r'^\s*({})\s*$'.format('|'.join(IGNORE_LINE_LIST)),
                              re.I)




def is_line_footer_by_content(line: str) -> bool:
    return bool(ATT_PAGE_NUM_PAT.match(line))


NOT_FOOTER_PAT = re.compile(r'^Note:', re.I)

def is_line_not_footer_aux(line: str) -> bool:
    words = stopwordutils.get_nonstopwords_nolc_gt_len1(line)
    num_lc_word = 0
    for word in words:
        if not word[0].isupper():
            num_lc_word += 1
    return num_lc_word >= 5



def is_line_not_footer_by_content(line: str) -> bool:
    if NOT_FOOTER_PAT.match(line):
        return True
    return is_line_not_footer_aux(line)


def is_line_footer(line: str,
                   page_line_num: int,
                   num_line_in_page: int,
                   lbk: float,
                   page_num_index: int,
                   is_english: bool,
                   is_centered: bool,
                   # TODO, remove, not used.  Mentioned in pdftxtparser.py
                   unused_align: str,
                   yStart: float):

    if yStart < 700.0:
        return False, -1.0
    if is_line_not_footer_by_content(line):
        return False, -1.0
    if is_line_footer_by_content(line):
        return True, 1.0

    # if it is a part of the last block
    # if lbk < 1.1:
    #    return False, -1

    is_debug_footer = False

    if is_debug_footer:
        print("is_line_footer({}, {})".format(line, page_line_num))

    score = 0.0
    if yStart >= 725.0:
        score += 0.4
    if is_debug_footer:
        print("score = {}, after yStart".format(score))
    if num_line_in_page - page_line_num <= 2:
        score += 0.5
    if is_debug_footer:
        print("score = {}, after num_line_in_page".format(score))
    if is_english:
        score -= 0.8
    else:
        score += 0.2
    if is_debug_footer:
        print("score = {}, after is_english".format(score))
    if len(line) < 30:
        score += 0.2
    if is_debug_footer:
        print("score = {}, after len(line)".format(score))
    if lbk >= 2.0:
        score += 0.2
    if is_debug_footer:
        print("score = {}, after lbk".format(score))
    if page_num_index != -1 and page_line_num >= page_num_index:
        score += 0.8
    if is_debug_footer:
        print("score = {}, after page_num_index = {}, page_line_num = {}".format(score,
                                                                                 page_num_index,
                                                                                 page_line_num))

    if 'confidential information' in line.lower() and is_centered:
        score += 0.8
    if is_debug_footer:
        print("score = {}, confid".format(score))

    # no sechead in footer, if it is obvious sechead
    if secheadutils.is_line_sechead_strict_prefix(line):
        score -= 20
    if is_debug_footer:
        print('is_footer.score = {}'.format(score))
    return score >= 1, score


HEADER_PAT = re.compile(r'(execution copy|anx343534anything)', re.I)

# no re.I
# TODO, jshaw, these should really be sechead, not headers.
# OK for now.
HEADER_PARTIAL_PAT = re.compile(r'(State and Local Sales and Use Tax|'
                                r'Exempt Use Certificate|State Department of)')

def is_line_header(line: str,
                   yStart: float,
                   line_num: int,
                   is_english: bool,
                   is_centered: bool,
                   align: str,
                   # num_line_in_block, int,
                   num_line_in_page: int,
                   header_set=None):

    # for domain specific headers
    if header_set and line.lower().strip() in header_set:
        return True

    # this is a normal sentences
    if is_english:
        if is_line_title(line):
            pass
        elif 'LF' in align:
            return False

    score = 0.0
    if HEADER_PAT.match(line) and yStart < 140:
        score += 0.9
    elif ((HEADER_PAT.match(line) or
           HEADER_PARTIAL_PAT.search(line)) and
          yStart < 140):
        score += 1.0
    elif yStart < 80.0:
        score += 0.7

    if not is_english or len(line) < 30:
        score += 0.2

    # don't use is_line_address(), too costly
    if secheadutils.is_line_sechead_prefix(line) or \
       is_line_address_prefix(line) or \
       is_line_signature_prefix(line):
        score -= 10.0

    if 'RT' in align or 'CN' in align:
        score += 0.3
    elif is_centered:   # sometimes, 'exhibit a' can be mistaken for header
        # a negative feature
        score -= 0.3

    #if num_line_in_block == 1:
    #    score += 0.2
    #elif num_line_in_block <= 2:
    #    score += 0.1

    # first or last line
    if line_num == 1 or line_num == num_line_in_page:
        score += 0.3
    elif line_num == 2 or line_num == num_line_in_page - 1:
        score += 0.2
    elif line_num < 4:
        score += 0.2

    # print("score = {}, is_line_header({})".format(score, line))
    return score >= 1.0


def is_invalid_sechead(unused_sechead,
                       prefix: str,
                       head: str,
                       unused_split_idx: int):
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


def extract_line_sechead(line: str, unused_prev_line: Optional[str] = None) \
    -> Optional[Tuple[str, str, str, int]]:
    sechead, prefix, head, split_idx = secheadutils.extract_sechead_v4(line,
                                                                       is_combine_line=True)
    if sechead:
        if not is_invalid_sechead(sechead, prefix, head, split_idx):
            return sechead, prefix, head, split_idx
    return None


SIGNATURE_PREFIX_PAT = re.compile(r'(By|Name|Title)(.*?):')
# SIGNATURE_PREFIX_PAT = re.compile(r'(By|Name|Title)\s*:')

def is_line_signature_prefix(line: str):
    mat = SIGNATURE_PREFIX_PAT.match(line)
    # print("mat = [{}]".format(mat.group(2)))
    # to handle 'Name.~:,', 'By_:', or 'Title_:'
    if mat and len(mat.group(2)) <= 3:
        return True
    return False


ADDRESS_PREFIX_PAT = re.compile(r'(attention|attn|c/o|email:|'
                                r'facsimi|fax|phone|tel|p[ \.]+o[ \.]+box)',
                                re.I)

def is_line_address_prefix(line: str):
    mat = ADDRESS_PREFIX_PAT.match(line)
    return mat

def is_line_address(line: str, is_english=False, is_sechead=False):
    if is_english or is_sechead:
        return False
    # each call takes 7 or 8 msec.  Too slow when
    # a file has 60000 lines, such as a lot of tables with
    # short non-english lines
    # return addresses.classify(line) >= 0.5
    return is_line_address_prefix(line)


PHONE_PAT = re.compile(r'(tel\S*|phone|fax|facsim\S*|'
                       r'phone num\S*)?[\s:]*(\(?\d\d\d\)?[\s\-]*\d\d\d[\s\-]*\d\d\d\d)',
                       re.I)


def is_line_phone_number(line):
    return PHONE_PAT.match(line)


EMAIL_PAT = re.compile(r'(email*)?[\s:]*\S+@\S+\.\S+', re.I)

def is_line_email(line):
    return EMAIL_PAT.match(line)


def is_line_english(line: str) -> bool:
    return engutils.classify_english_sentence(line)


def is_block_all_not_english(linex_list):   # List[LineWithAttrs]):
    is_all_not_english = False  # in case there is no element
    for linex in linex_list:
        if linex.is_english:
            return False
        is_all_not_english = True
    return is_all_not_english


def line_list_to_block_list(linex_list):
    if not linex_list:
        return []
    cur_block = [linex_list[0]]
    prev_block_num = linex_list[0].block_num
    tmp_block_list = [cur_block]  # list of list of line
    for linex in linex_list[1:]:
        block_num = linex.block_num
        if block_num != prev_block_num:
            cur_block = [linex]
            tmp_block_list.append(cur_block)
        else:
            cur_block.append(linex)
        prev_block_num = block_num
    return tmp_block_list


# this is in-place
# pylint: disable=too-many-locals
def update_ebsents_with_sechead(ebsent_list: List[corenlpsent.EbSentence],
                                paras_with_attrs: List[Tuple[List[Tuple[linepos.LnPos,
                                                                        linepos.LnPos]],
                                                             PLineAttrs]]) \
                                -> None:
    if not ebsent_list:  # if there is no data
        return

    para_i, len_paras = 0, len(paras_with_attrs)
    ebsent_i, len_ebsents = 0, len(ebsent_list)
    ebsent = ebsent_list[ebsent_i]
    ebsent_start, ebsent_end = ebsent.start, ebsent.end

    while para_i < len_paras and ebsent_i < len_ebsents:
        span_se_list, pline_attrs = paras_with_attrs[para_i]
        (unused_para_from_start, unused_para_from_end), \
            (para_to_start, para_to_end) = \
                docutils.span_frto_list_to_fromto(span_se_list)

        if para_to_start == para_to_end:  # empty line, move on
            para_i += 1
            continue

        sechead_attr = pline_attrs.sechead
        if sechead_attr:
            # print("attrs: {}".format(attrs[0]))
            unused_sechead_type, unused_sh_prefix_num, sh_header, unused_sh_idx = \
                sechead_attr
        else:
            sh_header = ''
        # ttx_span_se_list, ttx_pline_attrs = paras_with_attrs[para_i]
        # print("para #{}: ({}, {})".format(para_i, ttx_span_se_list,
        #                                   # this is to make the output look the same as
        #                                   # before 2018-08-25, can be removed in the future
        #                                   '[' + str(ttx_pline_attrs)[8:] + ']'))
        while ebsent_start <= para_to_end:
            if mathutils.start_end_overlap((ebsent_start, ebsent_end),
                                           (para_to_start, para_to_end)):
                # print("\tebsent set sechead ({}, {}): {}". \
                #       format(ebsent_start, ebsent_end, sh_header))
                if sh_header:
                    ebsent.set_sechead(' '. \
                        join(stopwordutils. \
                        tokens_remove_stopwords([word.lower()
                                                 for word in re.findall(r'\w+',
                                                                        sh_header)],
                                                is_lower=True)))
                # else, don't even set it
            ebsent_i += 1
            if ebsent_i < len_ebsents:
                ebsent = ebsent_list[ebsent_i]
                ebsent_start, ebsent_end = ebsent.start, ebsent.end
            else:
                ebsent_start = para_to_end + 1  # end the loop
        para_i += 1
    #ebsent_i = 0
    #while ebsent_i < len_ebsents:
    #    print("sent #{}: {}".format(ebsent_i, ebsent_list[ebsent_i]))
    #    ebsent_i += 1


def text_from_para_with_attrs(doc_text: str,
                              nlp_paras_with_attrs: List[Tuple[List[Tuple[linepos.LnPos,
                                                                          linepos.LnPos]],
                                                               PLineAttrs]]) -> str:
    para_st_list = []  # type: List[str]
    for nlp_para_with_attrs in nlp_paras_with_attrs:

        # print("para_with_attrs: {}".format(para_with_attrs))
        lnpos_pair_list, unused_attrs = nlp_para_with_attrs
        for from_lnpos, unused_to_lnpos in lnpos_pair_list:
            from_start, from_end, unused_from_line_num = from_lnpos.to_tuple()
            para_st_list.append(doc_text[from_start:from_end])

        # para_st_list.append(' '.join(para_st_list))
    nlp_text = '\n'.join(para_st_list)
    return nlp_text
