import re
import string
from typing import Dict, List, Optional, Set, Tuple

from kirke.utils import regexutils, txtreader

# IS_DEBUG = True
# IS_DEBUG_MODE = True

IS_DEBUG = False
IS_DEBUG_MODE = False

DATA_DIR = './dict/titles/'


"""Process strings as lines."""


ALNUM = set(string.ascii_letters).union(string.digits)
NON_TITLE_LABELS = [r'exhibit[^A-Za-z0-9]+\d+(?:\.\d+)*',
                    r'execution[^A-Za-z0-9]+copy']
LABEL_REGEXES = [re.compile(label) for label in NON_TITLE_LABELS]


"""Process strings as titles."""

def regex_of(category):
    with open(DATA_DIR + category + '.list') as fin2:
        # pylint: disable=invalid-name
        terms = [t.lower() for t in fin2.read().split('\n') if t.strip()]
    return re.compile(r'\b({})\b'.format('|'.join(terms)))


# These appeared in utils/regexutils.py also
TAG_REGEXES0 = [(re.compile(r'\d'), 1),
                (regex_of('cardinals'), 4),
                (regex_of('ordinals'), 6),
                (re.compile(r'\d+(?:st|nd|rd|th)'), 6),
                (regex_of('months'), 7),
                (regex_of('states'), 9)]

TAG_REGEXES = [(regex, str(tag) * tag) for (regex, tag) in TAG_REGEXES0]


def tag(line: str) -> str:
    # replace any non-alphanum
    line = re.sub(r'[\-\/\(\)“”",—]', ' ', line.lower())
    for (regex, xtag) in TAG_REGEXES:
        line = regex.sub(xtag, line)
    return line


def process_as_title(line):
    """Must lower before tag (1st) & tag before remove whitespace (New York)"""
    return ''.join(tag(alnum_strip(line.lower())).split())


# TODO, jshaw, should add all new vocab to both train and test titles
# we should be using the whole data set instead of the training
# too time pressed to modify/update train data and test data now.
def load_train_title_list() -> List[str]:
    # st_list = txtreader.load_str_list(DATA_DIR + 'past_titles_train.list')
    st_list = txtreader.load_str_list(DATA_DIR + 'jclean.train.title.list')
    seen_set = set([])  # type: Set[str]
    for line in st_list:
        line = line.strip().lower()
        if not line in seen_set:
            seen_set.add(line)

    st_list = txtreader.load_str_list(DATA_DIR + 'uk_titles_train.list')
    for line in st_list:
        line = line.strip().lower()
        if not line in seen_set:
            seen_set.add(line)
    return sorted(seen_set)

TRAIN_TITLE_LIST = load_train_title_list()

# TODO, jshaw, should remove BOM
# for the below file, need to remove the BOM at the end of the phrases toward the end of
# the file.  Should automate this
"""
Remember to swap comment line above.  Select past_titles_train.list instead
with open('dict/titles/jclean.train.title.list', 'w') as fout:
    for title in TRAIN_TITLE_LIST:
        print(title.strip(), file=fout)

# this is only for viewing, not used
with open('dict/titles/jclean2.train.title.list', 'w') as fout:
    for title in TRAIN_TITLE_LIST:
        print(tag(title), file=fout)
"""

# pylint: disable=invalid-name
train_title_wordset_list = []
for train_title in TRAIN_TITLE_LIST:
    norm_title_st = tag(train_title)
    line_wordset_x1 = set(norm_title_st.split())
    train_title_wordset_list.append(line_wordset_x1)

def alnum_strip(line):
    non_alnum = set(line) - ALNUM
    return line.strip(str(non_alnum)) if non_alnum else line


def remove_label_regexes(line):
    for label_regex in LABEL_REGEXES:
        line = label_regex.sub('', line)
    return line


def process_as_line(astr: str):
    if astr:
        # print("process_as_line({})".format(astr))
        x = alnum_strip(remove_label_regexes(alnum_strip(astr.lower())))
        # print("   return: {}".format(x))
        return x
    return astr


"""Get and process past titles to match against."""


with open(DATA_DIR + 'past_titles_train.list') as fin:
    # pylint: disable=invalid-name
    TITLES = {process_as_title(t) for t in fin.read().split('\n') if t.strip()}

with open(DATA_DIR + 'uk_titles_train.list') as fin:
    # pylint: disable=invalid-name
    UK_TITLES = {process_as_title(t) for t in fin.read().split('\n') if t.strip()}

# print("titles:")
# print(titles)
TITLES.update(UK_TITLES)


# pylint: disable=too-many-return-statements
def is_ok_title_filter(line: str, norm_line: str = '') -> bool:
    """Remove obvious problematic titles.
    """
    words = line.split()
    if len(words) > 8:
        return False
    if re.search(r'^(as |in )', line, re.I):
        return False
    if re.search(r'\bdated?\b', line, re.I):
        return False
    if re.match(r'section\b', line, re.I):
        return False
    # this probably is never reached because of extract_line_v2()'s
    # regexutils.remove_non_alpha_num(line_st)
    # if re.search(r'\.\.\.', line, re.I):
    #    return False
    if re.search(r'^\s*\d+.*\d+\s*$', line, re.I):   # '3 abc 4', toc
        return False

    # a heading
    # 1. xxxx
    # 1       Amendment
    if re.match(r'\s*[\divx]+\.\s+', line) or \
       re.match(r'[\divx]+\.?\s\s\s+', line):
        return False

    # a toc
    # 'xxx         13'
    if re.search(r'\s\s\s+[\divx]+\s*$', line) and len(line) > 60:
        return False

    norm_words = norm_line.split()
    # a date
    if len(norm_words) == 3 and \
       re.search(r'\b(1|11)\b', norm_line) and \
       re.search(r'\b(1111)\b', norm_line):
        return False

    """
    aaa = set(['1', '11', '1111'])
    print("aaa = {}".format(aaa))

    if len(norm_word) >= 3:
        aaa = set(norm_words[:3])
        print("aaa = {}".format(aaa))
    """

    # probably a date, followed by something else
    if len(norm_words) >= 3 and \
       len(set(norm_words[:3]).intersection(set(['1', '11', '1111']))) >= 2:
        return False
    if 'dear' in norm_words:
        return False
    return True


# the goal here is to find lines that might have titles
# pylint: disable=too-many-locals
def extract_lines_v2(paras_attr_list) -> Tuple[List[Dict],
                                               List[Tuple[int, int]],
                                               List[float]]:
    lines = []  # type: List[Dict]
    start_end_list = []
    adjust_score_list = []  # type: List[float]

    offset = 0
    is_found_party_line, unused_is_found_first_eng_para = False, False
    is_found_toc = False
    max_maybe_title_lines = 75

    num_toc_lines = 0
    # num_maybe_title_lines = -1
    num_lines_before_first_eng_para = 0
    for i, (line_st, para_attrs) in enumerate(paras_attr_list):
        if IS_DEBUG:
            attrs_st = '|'.join([str(attr) for attr in para_attrs])
            print("extract_line_v2()\t{}".format('\t'.join([attrs_st,
                                                            '[{}]'.format(line_st)])))

        line_st_len = len(line_st)

        if i >= max_maybe_title_lines or \
           'toc7' in para_attrs or \
           'toc' in para_attrs:
            is_found_toc = True
            num_toc_lines += 1

            if num_toc_lines > 12:
                break
            # there can be titles after incorrectly classified toc line
            # break
        if 'party_line' in para_attrs:
            is_found_party_line = True
            # num_maybe_title_line = i  # title cannot be in party line
            # break
        if 'first_eng_para' in para_attrs:
            num_lines_before_first_eng_para = len(lines)

        if is_found_party_line and i > 20:
            # break only are are really sure
            break

        # 'skip_as_template' not in para_attrs:
        orig_line_st = line_st
        line_st = regexutils.remove_non_alpha_num(line_st)
        is_toc = 'toc' in para_attrs or 'toc7' in para_attrs

        if line_st and not is_toc:
            title = 1 if 'title' in para_attrs else 0
            maybe_title = 1 if 'maybe_title' in para_attrs else 0
            # end_char is exclusive-end for last line (-1 = entire line)
            line = {'line': line_st, 'start': i, 'end': i, 'end_char': -1,
                    'title': title, 'maybe_title': maybe_title, 'is_toc': is_toc}
            lines.append(line)

            if IS_DEBUG_MODE:
                print('j3 line: [{}], attr = {}'.format(line_st, para_attrs))

            # lines and adjust_score_list must be synchronized
            if 'sechead' in para_attrs:
                adjust_score_list.append(-0.9)
            elif '...' in orig_line_st:  # check for toc
                adjust_score_list.append(-0.9)
            else:
                adjust_score_list.append(0)

        # lines and start_end_list are NOT synchronized
        # start_end_list must have the same size as i
        start_end_list.append((offset, offset + line_st_len))
        offset += line_st_len + 1

    if is_found_party_line or is_found_toc:
        return lines, start_end_list, adjust_score_list

    return lines[:num_lines_before_first_eng_para], start_end_list, adjust_score_list


def jaccard(word_list1: Set[str], word_list2: Set[str]) -> Tuple[float, Set[str], Set[str]]:
    intersect1 = word_list1.intersection(word_list2)
    union1 = word_list1.union(word_list2)

    return len(intersect1) / len(union1), intersect1, union1


# num_line is not used now
# pylint: disable=unused-argument
def calc_jaccard_title_list(line_st: str,
                            norm_line_st: str,
                            num_lines: int,
                            line_wordset: Set[str],
                            title_wordset_list: List[Set[str]]) -> Tuple[float, int, int, Set[str]]:

    adjust_score = 0.0
    # if certain line is bad, we want to eliminate it as a candidate
    if not is_ok_title_filter(line_st, norm_line_st):
        adjust_score = -0.6

    orig_score_x = calc_jaccard_title_list_aux(line_wordset, title_wordset_list)
    orig_score, num_intersect, num_union, title_wordset = orig_score_x

    score = orig_score + adjust_score

    if IS_DEBUG_MODE:
        print("calc_jaccard_title_list({}) = {}, orig_score = {}, {}".format(norm_line_st,
                                                                             score,
                                                                             orig_score,
                                                                             title_wordset))
    return score, num_intersect, num_union, title_wordset


def calc_jaccard_title_list_aux(line_wordset: Set[str],
                                title_wordset_list: List[Set[str]]) \
                                -> Tuple[float, int, int, Set[str]]:
    score_list = []  # type: List[Tuple[float, int, int, Set[str]]]
    for title_wordset in title_wordset_list:
        score, intersect1, union1 = jaccard(line_wordset, title_wordset)
        # just matching one word is not that impressive, such as lease
        if len(intersect1) == 1 and len(union1) == 1:
            score = 0.6
        # we don't want line, 'company', matches 'company note' in training
        if len(line_wordset) == 1 and score == 0.5:
            score = 0.33
        score_list.append((score, len(intersect1), len(union1), title_wordset))

    result = sorted(score_list, reverse=True)
    return result[0]

# mis-OCRed "English Deed OE Charge" vs "Deed Of Charge"
# intersect = 2, union = 5, or jaccard = 0.4
# We want this to be more than 1/3 or 0.33
MIN_JACCARD = 0.34


# pylint: disable=too-few-public-methods
class Title:

    def __init__(self, start: int, end: int, score: float, intersect: int) -> None:
        self.start = start
        self.end = end
        self.score = score
        self.intersect = intersect


# pylint: disable=too-many-branches, too-many-statements
def extract_offsets(paras_attr_list, unused_paras_text: str) -> Tuple[Optional[int],
                                                                      Optional[int]]:
    # pylint: disable=global-statement
    global train_title_wordset_list

    # Grab lines from the file
    linex_list, start_end_list, adjust_score_list = extract_lines_v2(paras_attr_list)
    if not linex_list:
        return None, None

    if IS_DEBUG:
        for linex, start_end in zip(linex_list, start_end_list):
            print("{}\t{}".format(start_end, linex))

    if IS_DEBUG_MODE:
        for linex in linex_list:
            print()
            print("jj({}, {})\t{}".format(linex['start'], linex['end'], linex))

            if linex:
                norm_ling = tag(linex['line'])
                print("line = [{}]".format(linex['line']))
                print("      \t{}".format(norm_ling))

    # Placeholder title. offsets: start, end, end_char (exclusive)
    # title = {'offsets': (-1, -1, -1), 'score': -1, 'ratio': -1, 'intersect': -1}
    title = Title(start=-1, end=-1, score=-1.0, intersect=-1)

    title_candidate_list = []  # type: List[Tuple[str, int, int, int]]

    found_title_try_limited = -1
    line_wordset_list = []
    score_list = []
    for i, (linex, adjust_score) in enumerate(zip(linex_list, adjust_score_list)):

        if found_title_try_limited < 0:
            pass
        elif found_title_try_limited > 0:
            found_title_try_limited -= 1
        elif found_title_try_limited == 0:
            break

        # print('adjscore = {}, linex = [{}]'.format(adjust_score, linex['line']))
        if adjust_score < 0:  # skip sechead
            continue

        line_st = linex['line']
        norm_line_st = tag(line_st)
        line_wordset = set(norm_line_st.split())
        line_wordset_list.append(line_wordset)

        title_candidate_list.append((line_st, linex['start'], linex['end'], 1))

        score, num_intersect, num_union, best_title = \
            calc_jaccard_title_list(line_st,
                                    norm_line_st,
                                    num_lines=1,
                                    line_wordset=line_wordset,
                                    title_wordset_list=train_title_wordset_list)

        score_list.append((score, num_intersect, num_union,
                           best_title, linex['start'], linex['end']))

        if score == title.score and num_intersect > title.intersect:
            title = Title(start=linex['start'],
                          end=linex['end'],
                          score=score,
                          intersect=num_intersect)
            if score > MIN_JACCARD:
                found_title_try_limited = 5
        elif score > title.score:
            title = Title(start=linex['start'],
                          end=linex['end'],
                          score=score,
                          intersect=num_intersect)
            if score > MIN_JACCARD:
                found_title_try_limited = 5

        # try to add the next line
        if i + 1 < len(linex_list):
            next_linex = linex_list[i+1]
            next_line_st = next_linex['line']
            # the line must be next to each other
            if next_linex['start'] != linex['start'] + 1:
                continue
            two_lines = line_st + ' ' + next_line_st

            title_candidate_list.append((two_lines, linex['start'], next_linex['end'], 2))

            norm_line_st = tag(two_lines)
            line_wordset = set(norm_line_st.split())
            line_wordset_list.append(line_wordset)

            score, num_intersect, num_union, best_title = \
                calc_jaccard_title_list(two_lines,
                                        norm_line_st,
                                        num_lines=2,
                                        line_wordset=line_wordset,
                                        title_wordset_list=train_title_wordset_list)

            score_list.append((score, num_intersect, num_union, best_title,
                               linex['start'], next_linex['end']))

            if score == title.score and num_intersect > title.intersect:
                title = Title(start=linex['start'],
                              end=next_linex['end'],
                              score=score,
                              intersect=num_intersect)
                if score > MIN_JACCARD:
                    found_title_try_limited = 5
            elif score > title.score:
                title = Title(start=linex['start'],
                              end=next_linex['end'],
                              score=score,
                              intersect=num_intersect)
                if score > MIN_JACCARD:
                    found_title_try_limited = 5


    if IS_DEBUG_MODE:
        print('\n\n')
        for score, num_intersect, num_union, best_title, tstart, tend in sorted(score_list,
                                                                                reverse=True):
            print('score {}, itx={}, un={}({}, {}), best_title = [{}]'.format(score,
                                                                              num_intersect,
                                                                              num_union,
                                                                              tstart, tend,
                                                                              best_title))

    if title.score > MIN_JACCARD:
        line_start = title.start
        line_end = title.end

        if IS_DEBUG_MODE:
            print("line_start = {}".format(line_start))
            print("se ===== {}".format(start_end_list[line_start]))
        start_offset = start_end_list[line_start][0]
        end_offset = start_end_list[line_end][1]

        return start_offset, end_offset

    # ok, no title found, try a set of heuristics
    # This only works for lines
    for i, (title_candidate, lx_start, lx_end, num_line) in enumerate(title_candidate_list):
        if IS_DEBUG_MODE:
            print("  tt cand #{} nline={} : [{}]".format(i, num_line, title_candidate))
        if num_line == 1:
            if is_ok_title_filter(title_candidate):
                if re.search(r'\b(agreement|letter|contract)\s*$', title_candidate):
                    start_offset = start_end_list[lx_start][0]
                    end_offset = start_end_list[lx_end][1]
                    return start_offset, end_offset

    return None, None


# pylint: disable=too-many-branches, too-many-statements
def extract_offsets_not_line(paras_attr_list, paras_text: str) -> Tuple[Optional[int],
                                                                        Optional[int]]:
    """Extract title based on regex.

    Example: 'This consuting agreement (the "agreement")...
    """
    offset = 0
    for line_st, para_attrs in paras_attr_list:
        line_st_len = len(line_st)

        if 'party_line' in para_attrs:
            # must start from ebgin of a sentence
            mat = re.match(r'((\w+)(\s+\w+)+)\s+\(the [“"”]agreement[“"”]\)', line_st, re.I)
            if mat and len(mat.group(3).split()) < 5:
                # maybe_title_st = mat.group(1)
                first_word = mat.group(2)
                span_start = offset + mat.start(1)
                if first_word in set(['the', 'this']):
                    span_start = offset + mat.start(3)
                span_end = offset + mat.end(1)
                return span_start, span_end

        offset += line_st_len + 1
    return None, None


def extract_nl_offsets(nl_text: str) -> Tuple[Optional[int],
                                              Optional[int]]:
    """Extract based on NL offsets.

       Because sometimes nl_text is empty, i.e., HTML documents, we also try
       paras_text on some regex also.
    """
    # pylint: disable=global-statement
    global train_title_wordset_list

    se_lines = list(txtreader.text_to_lines_with_offsets(nl_text))

    # Placeholder title. offsets: start, end, end_char (exclusive)
    # title = {'offsets': (-1, -1, -1), 'score': -1, 'ratio': -1, 'intersect': -1}
    title = Title(start=-1, end=-1, score=-1.0, intersect=-1)

    title_candidate_list = []  # type: List[Tuple[str, int, int, int]]

    found_title_try_limited = -1
    line_wordset_list = []
    score_list = []
    for i, se_line in enumerate(se_lines):
        start, end, line = se_line

        if found_title_try_limited < 0:
            pass
        elif found_title_try_limited > 0:
            found_title_try_limited -= 1
        elif found_title_try_limited == 0:
            break

        if len(line) > 300:  # skip long line
            continue

        line_st = line
        norm_line_st = tag(line_st.lower())
        line_wordset = set(norm_line_st.split())
        line_wordset_list.append(line_wordset)

        title_candidate_list.append((line_st, start, end, 1))

        score, num_intersect, num_union, best_title = \
            calc_jaccard_title_list(line_st,
                                    norm_line_st,
                                    num_lines=1,
                                    line_wordset=line_wordset,
                                    title_wordset_list=train_title_wordset_list)

        score_list.append((score, num_intersect, num_union, best_title, start, end))

        if score > MIN_JACCARD and \
           ((score == title.score and num_intersect > title.intersect) or \
            score > title.score):

            title = Title(start=start,
                          end=end,
                          score=score,
                          intersect=num_intersect)
            found_title_try_limited = 5

        # now try 2 lines
        # try to add the next line
        if i + 1 < len(se_lines):
            unused_next_start, next_end, next_line_st = se_lines[i+1]

            two_lines = line_st + ' ' + next_line_st

            title_candidate_list.append((two_lines, start, next_end, 2))

            norm_line_st = tag(two_lines)
            line_wordset = set(norm_line_st.split())
            line_wordset_list.append(line_wordset)

            score, num_intersect, num_union, best_title = \
                calc_jaccard_title_list(two_lines,
                                        norm_line_st,
                                        num_lines=2,
                                        line_wordset=line_wordset,
                                        title_wordset_list=train_title_wordset_list)

            score_list.append((score, num_intersect, num_union, best_title,
                               start, next_end))

            if score > MIN_JACCARD and \
               ((score == title.score and num_intersect > title.intersect) or \
                score > title.score):
                title = Title(start=start,
                              end=next_end,
                              score=score,
                              intersect=num_intersect)
                found_title_try_limited = 5

    if IS_DEBUG_MODE:
        print('\n\n')
        for score, num_intersect, num_union, best_title, tstart, tend in sorted(score_list,
                                                                                reverse=True):
            print('score {}, itx={}, un={}({}, {}), best_title = [{}]'.format(score,
                                                                              num_intersect,
                                                                              num_union,
                                                                              tstart, tend,
                                                                              best_title))

    if title.score > MIN_JACCARD:
        span_start, span_end = title.start, title.end
        span_text = nl_text[span_start:span_end]
        # clean up spaces at the end
        end_space_mat = re.search(r'\s+$', span_text)
        if end_space_mat:
            span_end -= len(end_space_mat.group())
        return span_start, span_end

    # ok, no title found, try a set of heuristics
    # This only works for lines
    for i, (title_candidate, lx_start, lx_end, num_line) in enumerate(title_candidate_list):
        if IS_DEBUG_MODE:
            print("  tt cand #{} nline={} : [{}]".format(i, num_line, title_candidate))
        if num_line == 1:
            if is_ok_title_filter(title_candidate):
                if re.search(r'\b(agreement|letter|contract)\s*$', title_candidate):
                    span_start, span_end = lx_start, lx_end
                    span_text = nl_text[span_start:span_end]
                    # clean up spaces at the end
                    end_space_mat = re.search(r'\s+$', span_text)
                    if end_space_mat:
                        span_end -= len(end_space_mat.group())
                    return span_start, span_end


    # At this stage, still no title found.
    # Will take "This xxx agreement (the "agreement")
    for i, se_line in enumerate(se_lines):
        start, end, line = se_line
        # print('trying out ({}, {}) [{}]'.format(start, end, line))
        # must start from ebgin of a sentence
        mat = re.match(r'((\w+)(\s+\w+)+)\s+\(the [“"”]agreement[“"”]\)', line, re.I)
        if mat and len(mat.group(3).split()) < 5:
            # maybe_title_st = mat.group(1)
            first_word = mat.group(2)
            span_start = mat.start(1)
            if first_word in set(['the', 'this']):
                span_start = mat.start(3)
            span_end = mat.end(1)
            return span_start, span_end
        if i > 50:  # we only do this up till first 50 lines
            break

    return None, None


# pylint: disable=too-few-public-methods
class TitleAnnotator:

    def __init__(self, provision):
        self.provision = 'title'

    # pylint: disable=no-self-use
    def extract_provision_offsets(self,
                                  paras_with_attrs,
                                  paras_text: str) -> Tuple[Optional[int],
                                                            Optional[int]]:

        if IS_DEBUG:
            print("title called extract_provision_offsets()")

        # print("tag date: {}".format(tag('12 January 2017')))
        return extract_offsets(paras_with_attrs, paras_text)

    def extract_provision_offsets_not_line(self,
                                           paras_with_attrs,
                                           paras_text: str) -> Tuple[Optional[int],
                                                                     Optional[int]]:
        """Extract title based on regex.

           Example: 'This consuting agreement (the "agreement")...
        """

        if IS_DEBUG:
            print("title called extract_provision_offsets_not_line()")

        # print("tag date: {}".format(tag('12 January 2017')))
        return extract_offsets_not_line(paras_with_attrs, paras_text)

    def extract_nl_provision_offsets(self,
                                     nl_text: str) -> Tuple[Optional[int],
                                                            Optional[int]]:
        if IS_DEBUG:
            print("title called extract_nl_provision_offsets()")

        # print("tag date: {}".format(tag('12 January 2017')))
        return extract_nl_offsets(nl_text)
