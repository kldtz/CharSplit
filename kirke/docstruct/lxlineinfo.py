from collections import Counter
import copy
from functools import total_ordering
import re
# pylint: disable=unused-import
from typing import Any, Dict, List, Optional, Tuple

from kirke.docstruct import tokenizer
from kirke.utils import strutils, engutils, stopwordutils

# if 2 y's are diff by less than this value, they are in the same line
# Mainly used for table detection
MAX_Y_DIFF_AS_SAME = 1.0

MAX_X_DIFF_AS_SAME_LINE = 6.0

SHORT_SENT_WORD_THRESHOLD = 40

# X_L_CENTER = 300
# X_R_CENTER = 310

# LineInfo = namedtuple('LineInfo', ['start', 'end', 'sid', 'xStart', 'xEnd', 'yStart',
#                                   'para', 'page', 'text', 'length', 'words'])

def is_short_sent_by_length(xlen: int) -> bool:
    return xlen < SHORT_SENT_WORD_THRESHOLD

# is_relax_check is only True when on first page
# pylint: disable=too-many-return-statements
def is_line_centered(line: str, x_start: float, x_end: float,
                     # pylint: disable=unused-argument
                     is_relax_check=False) -> bool:

    if len(line) > 65:
        return False
    if strutils.is_all_caps_space(line) and len(line) > 52:
        return False

    right_diff = 612 - x_end   # (0, 0, 595, 842);
    left_diff = x_start

    # print("left_diff = {}, right_diff= {}, diff = {}".
    # format(left_diff, right_diff, abs(right_diff - left_diff)))
    # print("text = {}".format(self.text))
    if left_diff > 100 and abs(right_diff - left_diff) < 18:
        return True

    # there are some short lines that are not really centered
    if x_end - x_start < 100 and \
       left_diff > 100 and \
       abs(right_diff - left_diff) < 80:
        return True

    # there are some short lines that are not really centered
    if x_end - x_start < 100 and \
       left_diff > 100 and \
       abs(right_diff - left_diff) < 80:
        return True

    if x_end - x_start < 340 and \
       left_diff > 100 and \
       abs(right_diff - left_diff) < 50:
        return True

    return False


# currently is_relax_check is not used
def calc_align_label(line: str, x_start: float, x_end: float,
                     # pylint: disable=unused-argument
                     is_relax_check=False) -> str:
    if is_line_centered(line, x_start, x_end):
        return "CN"
    if x_start < 90:
        return "LF1"
    if x_start < 115:
        return "LF2"
    if x_start < 130:
        return "LF3"
    if x_start > 450:
        return "RT"
    return ""

@total_ordering
# pylint: disable=too-many-instance-attributes
class LineInfo:

    # pylint: disable=too-many-arguments
    def __init__(self, start: int, end: int, sid: int, x_start: float, x_end: float, y_start: float,
                 para: int, page: int, text: str, length: int,
                 words: List[str], non_punct_words: List[str]) -> None:
        self.start = start
        self.end = end
        self.sid = sid
        # pylint: disable=invalid-name
        self.xStart = x_start
        # pylint: disable=invalid-name
        self.xEnd = x_end
        # pylint: disable=invalid-name
        self.yStart = y_start
        self.para = para
        self.page = page
        self.text = text
        self.length = length
        self.words = words
        self.non_punct_words = non_punct_words
        self.category = None  # type: Optional[str]
        self.align_label = calc_align_label(text, x_start, x_end)
        # False means it is EITHER unknown or is not True
        self.is_close_prev_line = False
        self.is_english = engutils.classify_english_sentence(text)

    def __eq__(self, other) -> bool:
        #if hasattr(other, 'start') and hasattr(other, 'end'):
        return (self.start, self.end).__eq__((other.start, other.end))

    def __lt__(self, other) -> Any:
        #if hasattr(other, 'start') and hasattr(other, 'end'):
        return (self.start, self.end).__lt__((other.start, other.end))

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __repr__(self) -> str:
        # self.words is not printed, intentionally
        return str(('start={}'.format(self.start),
                    'end={}'.format(self.end),
                    'sid={}'.format(self.sid),
                    'xStart={:.2f}'.format(self.xStart),
                    'xEnd={:.2f}'.format(self.xEnd),
                    'yStart={:.2f}'.format(self.yStart),
                    'para={}'.format(self.para),
                    'page={}'.format(self.page),
                    'english={}'.format(self.is_english),
                    self.text,
                    'len={}'.format(self.length),
                    # 'words={}'.format(self.words),
                    'is_close_prev_line={}'.format(self.is_close_prev_line),
                    'category={}'.format(self.category),
                    'align_label={}'.format(self.align_label)))

    def tostr2(self, text: str) -> str:
        not_en_mark = ''
        if self.is_english:
            tags = ['EN']
        else:
            tags = ['NOT_EN']
            not_en_mark = '>>>'
        if self.category:
            tags.append('CAT-{}'.format(self.category))
        if self.is_close_prev_line:
            tags.append("CLSE")
        else:
            tags.append("FAR")
        if self.align_label:
            tags.append(self.align_label)
        tags.append('LN-{}'.format(self.length))
        # tags.append('PG-{}'.format(self.page))

        tags_st = ' '.join(tags)

        # self.words is not printed, intentionally
        return '%-25s\t%s\t%s' % (tags_st, not_en_mark, text[self.start:self.end])


    def is_center(self) -> bool:
        return self.align_label == 'CN'

    # pylint: disable=invalid-name
    def is_LF(self) -> bool:
        return bool(self.align_label) and self.align_label.startswith('LF')

    # pylint: disable=invalid-name
    def is_LF1(self) -> bool:
        return self.align_label == 'LF1'

    # pylint: disable=invalid-name
    def is_LF2(self) -> bool:
        return self.align_label == 'LF2'

    # pylint: disable=invalid-name
    def is_LF3(self) -> bool:
        return self.align_label == 'LF3'

    def update_text(self, text: str) -> None:
        self.text = text
        self.length = len(text)
        self.words = tokenizer.word_tokenize(self.text)
        self.non_punct_words = [word for word in self.words if not strutils.is_punct(word)]
        self.is_english = engutils.classify_english_sentence(self.text)


    # TODO, jshaw, hack
    #def to_aligned(self, is_relax_check=False):
    #    if self.is_center(is_relax_check=is_relax_check):
    #        return "CN"
    #    tmp_aligned = xstart_to_aligned(self.xStart)
    # return tmp_aligned


    # TODO, jshaw, hack
    #def is_aligned_right(self):
    #    return self.xStart > 300




"""
TypedLineInfo = namedtuple('TypedLineInfo', ['start', 'end', 'sid',
                                             'para', 'page', 'text', 'length', 'category',
                                             'isCenter', 'isAlignedLeft', 'isAlignedRight',
                                             'isInTable'])

def toTypedLineInfo(lineinfo, category):
    return TypedLineInfo(lineinfo.start,
                         lineinfo.end,
                         lineinfo.sid,
                         lineinfo.para,
                         lineinfo.page,
                         lineinfo.text,
                         lineinfo.length,
                         category,
                         lineinfo.is_center(),
                         lineinfo.is_aligned_left(),
                         lineinfo.is_aligned_right(),
                         False)
"""

def mark_close_adjacent_lines(lineinfo_list: List[LineInfo]) -> None:
    """Set lineinfo.is_close_prev_line if their Y are less than 22 points apart.

    Args:
        lineinfo_list: the lineinfo of a document

    Returns:
        In-place modified lineinfo list.
    """

    # first figure what's the average yStart difference
    prev_ystart = lineinfo_list[0].yStart
    ydiff_counter = Counter()  # type: Counter
    for lineinfo in lineinfo_list[1:]:
        ydiff_int = int(lineinfo.yStart - prev_ystart)
        ydiff_counter[ydiff_int] += 1
        prev_ystart = lineinfo.yStart

    if not ydiff_counter.most_common(1):
        return
    # print("ydiff_count.top = {}".format(ydiff_counter.most_common(1)))
    max_isclose_ydiff = ydiff_counter.most_common(1)[0][0] * 1.5
    # print("max_isclose_ydiff = {}".format(max_isclose_ydiff))

    prev_ystart = lineinfo_list[0].yStart
    for lineinfo in lineinfo_list[1:]:
        ydiff = lineinfo.yStart - prev_ystart
        # used 17.5
        # if ydiff > 0 and ydiff <= 22 and not lineinfo.is_center():
        # TODO, jshaw, not sure why lineinfo.is_center() is used before
        if ydiff > 0 and ydiff <= max_isclose_ydiff:
            lineinfo.is_close_prev_line = True
        prev_ystart = lineinfo.yStart

"""
def text2page_lineinfos_list(text: str, line_offsets: List[Dict]) -> List[LineInfo]:
    lineinfo_list = []
    # line_st_list = []
    for i, line_offset in enumerate(line_offsets):
        start = line_offset['start']
        end = line_offset['end']
        line = text[start:end]

#        if not line.strip():  # blank line, or line with only spaces
#            # They have no use, unless for formatting reason
#            # Skip them for now since they complicates things.
#            continue

        # it's possible that a line will have no words
        words = tokenizer.word_tokenize(line)
        non_punct_words = [word for word in words if not strutils.is_punct(word)]

        # InfoInfo contains several interesting inferred information
        #    - is_close_prev_line in y-axis
        #    - is that line a english sentnece
        #    - aligned label
        lineinfo = LineInfo(start, end,
                            i,
                            line_offset['xStart'], line_offset['xEnd'],
                            line_offset['yStart'],
                            line_offset['para'], line_offset['page'],
                            line,
                            len(line),
                            words,
                            non_punct_words)
        lineinfo_list.append(lineinfo)

        # line_st_list.append(text[lineinfo.start:lineinfo.end])

    # print('len(lineinfo_list) = {}'.format(len(lineinfo_list)))
    merged_lineinfo_list = merge_samey_adjacent_lineinfos(lineinfo_list, text)
    # print('len(merged_lineinfo_list) = {}'.format(len(merged_lineinfo_list)))

    # It takes 7 seconds to compute this for large PDF files.  Not worth the effort
    # time_0 = time()
    # jenks = jenksutils.Jenks([lineinfo.xStart for lineinfo in merged_lineinfo_list])
    # for lineinfo in merged_lineinfo_list:
    #    lineinfo.align_label = jenks.classify(lineinfo.xStart)
    # print("done in %0.3fs" % (time() - time_0))

    # mark lines next to each other that are very close to be in-para
    mark_close_adjacent_lines(merged_lineinfo_list)

    with open('hello1.txt', 'wt') as out1:
        for linfo in merged_lineinfo_list:
            # print("{}\t{}\n".format(linfo.sid, linfo.text), file=out1)
            print("{}\n".format(linfo.text), file=out1)

    sechead_split_linfo_list = split_sechead_lineinfos(merged_lineinfo_list)
    # copy it back so rest of the code doesn't change
    merged_lineinfo_list = sechead_split_linfo_list

    with open('hello2.txt', 'wt') as out1:
        for linfo in merged_lineinfo_list:
            # print("{}\t{}\n".format(linfo.sid, linfo.text), file=out1)
            print("{}\n".format(linfo.text), file=out1)

    prev_page_num = -1
    curpage_lineinfo_list = None
    page_lineinfos_list = []  # list of list of lineinfos
    for lineinfo in merged_lineinfo_list:
        if lineinfo.page != prev_page_num:
            curpage_lineinfo_list = []
            page_lineinfos_list.append(curpage_lineinfo_list)
        curpage_lineinfo_list.append(lineinfo)
        prev_page_num = lineinfo.page

        # print("page: {}, para: {}, ({}, {}): [{}]".format(line_offset['page'],
        #                                                  line_offset['para'],
        #                                                  start, end,
        #                                                  text[start:end]))

    return merged_lineinfo_list, page_lineinfos_list
"""

# SEC_HEAD_PAT = re.compile(r"^(\d+\.)\s+(([\w\-]+;?\s*)+[\.:])\s*(.*)$")
# SEC_HEAD_CHECK_PAT = re.compile(r"^(\d+\.)\s+((\w[\w\-]+;?\s*)+[\.:])(.*)$")
SEC_HEAD_CHECK_PAT = re.compile(r"^(\d+\.)\s+([^\.:]+[\.:])(.*)$")
SEC_HEAD_PAT = re.compile(r"^(\d+\.)\s+(([a-zA-Z\-]+;?\s*)+[\.:])(.*)$")

SEC_HEAD_FULL_LINE_PAT = re.compile(r"^(\d+\.)\s+(.+)$")


def find_sechead_prefix_part_line(line):
    # first check if full line is a heading, let other deal with it
    mat = SEC_HEAD_FULL_LINE_PAT.match(line)
    if mat:
        sec_words_st = mat.group(2)
        sec_words = stopwordutils.tokens_remove_stopwords(sec_words_st.split())
        # TODO, jshaw, "re:" is NOT handled correctly in partial match except in the line below
        # need to treate "re:" as a regular word instead of relying on the punctuation
        # "9. OPA Obligations re: Other Contracts"
        sec_words = [word for word in sec_words if word not in set(['re:'])]
        if len(sec_words) <= 5 and strutils.is_all_title_words(sec_words):
            # this is full section head per line
            # print("hhhhhhhhhhhhhhhhhhhhh: {}".format(line))              # skip
            return None

    # the above regex, 2nd one is VERY slow if the check is not enforced
    mat = SEC_HEAD_CHECK_PAT.match(line)
    if not mat:
        return None
    if mat and not mat.group(3).strip():  # matched, but empty rest text
        return None

    sec_words_st = mat.group(2)[:-1]
    sec_words = stopwordutils.tokens_remove_stopwords(sec_words_st.split())
    if not (len(sec_words) <= 5 and strutils.is_all_title_words(sec_words)):
        return None

    # TODO, jshaw, something is wrong with the regex that it is slow.
    # Some backtrack is triggered
    mat = SEC_HEAD_PAT.match(line)
    if mat:
        # sec_num = mat.group(1)
        sec_words_st = mat.group(2)[:-1]
        # rest_group = mat.group(4)

        sec_words = stopwordutils.tokens_remove_stopwords(sec_words_st.split())
        if len(sec_words) <= 5 and strutils.is_all_title_words(sec_words):
            # return mat.group(1), mat.group(2), mat.group(4)
            return mat
    return None


# this is destructive
def split_sechead_lineinfos(lineinfo_list: List[LineInfo]) -> List[LineInfo]:
    split_sid = 0
    result = []
    for linfo in lineinfo_list:

        # print("linfo_id = {}".format(linfo.sid))
        if not linfo.is_close_prev_line:
            # print("find_sechead_prefix_part_line = [{}]".format(linfo.text))
            mat = find_sechead_prefix_part_line(linfo.text)
            if mat and mat.group(4):   # must be text in rest of non-sechead
                acopy = copy.copy(linfo)  # shallow copy
                acopy.sid = split_sid
                split_sid += 1
                sechead_start = mat.start(1)
                sechead_end = mat.end(2)
                sechead_text = linfo.text[sechead_start:sechead_end]
                acopy.start = linfo.start + sechead_start
                acopy.end = linfo.start + sechead_end
                acopy.update_text(sechead_text)
                acopy.category = 'sechead'
                # modify lininfo
                result.append(acopy)

                # at this point, xStart, xEnd, yStart are meaningless after lineinfo transformation
                # aligned label is also incorrect

                rest_text = linfo.text[mat.start(4):]
                linfo = copy.copy(linfo)
                linfo.start = linfo.start + mat.start(4)
                linfo.update_text(rest_text)
                linfo.sid = split_sid
                result.append(linfo)
            else:
                linfo.sid = split_sid
                result.append(linfo)
        else:
            linfo.sid = split_sid
            result.append(linfo)

        split_sid += 1  # always increase by 1

    return result


"""
def merge_samey_adjacent_lineinfos(lineinfo_list: List[LineInfo], text: str) -> List[LineInfo]:
    " ""Merge adjacent lineinfos if their Y are very similar, and their
       X are less than 5.

    Args:
       lineinfo_list:
       text:

    Returns:
       List[LineInfo]: list of merged lineinfos.
    " ""
    debug_mode = False
    prev_end, prev_y_start, prev_paragraph_num, prev_page_num = -1, -1, -1, -1
    prev_x_end = -1.0
    prev_line_info = None  # type: Optional[LineInfo]
    result = []

    merged_sid = 0
    for i, linfo in enumerate(lineinfo_list):
	# if they are of same Y (in 2D) and x (in offset space) is adjacent
        if debug_mode and prev_line_info:
            print("line #{}, page= {}, prev.XEnd = {}, curStart = {}, diff = {}".format(linfo.sid, linfo.page,
                                                                                        prev_line_info.xEnd, linfo.xStart,
                                                                                        (linfo.xStart - prev_x_end)))

        #if linfo.sid == 1357:
        #    print("hello6434")
        y_diff = abs(linfo.yStart - prev_y_start) if prev_y_start != -1 else 100.0
        if (linfo.page == prev_page_num and linfo.para == prev_paragraph_num and
                y_diff < MAX_Y_DIFF_AS_SAME and linfo.start == prev_end + 1 and
            linfo.xStart - prev_x_end <= MAX_X_DIFF_AS_SAME_LINE):
            " ""
            if debug_mode:
                prev_line_st = ''
                if i > 0:
                    prev_line_st = line_st_list[i - 1]
                line_st = line_st_list[i]
                # print('  i =', i, file=sys.stderr)
                # print('  prevLine : [{}] ='.format(prevLineSt), file=sys.stderr)
                # print('   curLine : [{}] ='.format(lineSt), file=sys.stderr)
                print('  i =', i)
                print('  prevLine : [{}]'.format(prev_line_st))
                print('   curLine : [{}]'.format(line_st))
            " ""
            prev_line_info.end = linfo.end
            prev_line_info.xEnd = linfo.xEnd
            prev_line_info.update_text(text[prev_line_info.start:prev_line_info.end])
        else:
            acopy = copy.copy(linfo)  # shallow copy
            acopy.sid = merged_sid
            merged_sid += 1
            result.append(acopy)
            prev_line_info = acopy

        prev_end = linfo.end
        prev_y_start = linfo.yStart
        prev_x_end = linfo.xEnd
        prev_page_num = linfo.page
        prev_paragraph_num = linfo.para
    return result
"""

def find_list_start_end(lineinfo_list):
    min_start = lineinfo_list[0].start
    max_end = lineinfo_list[0].end
    for lineinfo in lineinfo_list[1:]:
        if lineinfo.start < min_start:
            min_start = lineinfo.start
        if lineinfo.end > max_end:
            max_end = lineinfo.end
    return min_start, max_end


def is_word_overlap(group_linfos_list, page_lineinfo_list, skip_lineinfo_list, perc):
    group_word_count = 0
    for group_linfos in group_linfos_list:
        for linfo in group_linfos:
            if linfo not in skip_lineinfo_list:
                group_word_count += len(linfo.words)

    page_word_count = 0
    for linfo in page_lineinfo_list:
        if linfo not in skip_lineinfo_list:
            page_word_count += len(linfo.words)

    word_overlap_perc = group_word_count / page_word_count
    return word_overlap_perc >= perc



def is_lineinfos_word_overlap(small_linfo_list, big_linfo_list, perc):
    small_set = set(small_linfo_list)
    big_set = set(big_linfo_list)
    a_has_b_not = 0
    b_has_a_not = 0
    both_have = 0
    for linfo in small_linfo_list:
        if linfo in big_set:
            both_have += len(linfo.words)
        else:
            a_has_b_not += len(linfo.words)

    for linfo in big_linfo_list:
        if linfo not in small_set:
            b_has_a_not += len(linfo.words)

    word_overlap_perc = both_have / (both_have + a_has_b_not + b_has_a_not)
    return word_overlap_perc >= perc


def is_item_prefix(lineinfo):
    if not lineinfo.words:  # if it has no words
        return False
    # first check is for 'a)', or '10)'
    if len(lineinfo.words[0]) <= 2 and \
       len(lineinfo.words) >= 2 and \
        lineinfo.words[1] in [')', '.']:
        return True
    if len(lineinfo.words[0]) <= 2 and \
       len(lineinfo.words) >= 3 and \
       (lineinfo.words[1] in [')', '.'] or lineinfo.words[2] in [')', '.']):
        return True
    return False

def is_itemized_list(lineinfo_list):
    count_item_prefix = 0
    # count_lc_start = 0
    for lineinfo in lineinfo_list:
        if is_item_prefix(lineinfo):
            count_item_prefix += 1
        #if strutils.is_lc(lineinfo.words[0]):
        #    count_lc_start += 1

    return count_item_prefix >= 4
