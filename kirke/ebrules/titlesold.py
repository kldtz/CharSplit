
from math import ceil
import re
import string
# pylint: disable=unused-import
from typing import Dict, List

from scipy import stats

from fuzzywuzzy import fuzz, process

from kirke.utils import regexutils

# TODO, jshaw, remove this module
# The code here is superceded by titles.py
# Keeping the old code for now because kirke/ebrules/parties.py is
# calling title_ratio().  Should try toremove this module in the future.

IS_DEBUG = False

"""Config. Weights: ratio, length, title, maybe_title. Last two not reliable."""


DATA_DIR = './dict/titles/'
TITLES_MATCH_PCT = 0.387
MAX_TITLE_LINES = 5
WEIGHTS = [313, 30, 0, 0]
# this number is somewhat random, 0.5 was set before, also random.
MIN_TITLE_RATIO = 0.5


"""Process strings as lines."""


ALNUM = set(string.ascii_letters).union(string.digits)
NON_TITLE_LABELS = [r'exhibit[^A-Za-z0-9]+\d+(?:\.\d+)*',
                    r'execution[^A-Za-z0-9]+copy']
LABEL_REGEXES = [re.compile(label) for label in NON_TITLE_LABELS]


def alnum_strip(line: str):
    non_alnum = set(line) - ALNUM
    return line.strip(str(non_alnum)) if non_alnum else line


def remove_label_regexes(line: str):
    for label_regex in LABEL_REGEXES:
        line = label_regex.sub('', line)
    return line


def process_as_line(astr: str):
    if astr:
        # print("process_as_line({})".format(astr))
        out = alnum_strip(remove_label_regexes(alnum_strip(astr.lower())))
        # print("   return: {}".format(x))
        return out
    return astr


"""Process strings as titles."""


def regex_of(category):
    with open(DATA_DIR + category + '.list') as fin_outer1:
        terms = [t.lower() for t in fin_outer1.read().split('\n') if t.strip()]
    return re.compile(r'\b({})\b'.format('|'.join(terms)))


TAG_REGEXES_1 = [(re.compile(r'\d'), 1), (regex_of('cardinals'), 4),
                 (regex_of('ordinals'), 6), (re.compile(r'1+(?:st|nd|rd|th)'), 6),
                 (regex_of('months'), 7), (regex_of('states'), 9)]

TAG_REGEXES = [(regex, str(tag) * tag) for (regex, tag) in TAG_REGEXES_1]


def tag(line):
    for (regex, atag) in TAG_REGEXES:
        line = regex.sub(atag, line)
    return line


def process_as_title(line):
    """Must lower before tag (1st) & tag before remove whitespace (New York)"""
    return ''.join(tag(alnum_strip(line.lower())).split())


"""Get and process past titles to match against."""


with open(DATA_DIR + 'past_titles_train.list') as fin_outer2:
    TITLES = {process_as_title(t) for t in fin_outer2.read().split('\n') if t.strip()}

with open(DATA_DIR + 'uk_titles_train.list') as fin_outer3:
    UK_TITLES = {process_as_title(t) for t in fin_outer3.read().split('\n') if t.strip()}

# print("titles:")
# print(titles)
TITLES.update(UK_TITLES)


"""Extract and format lines from a file"""


CANT_BEGIN = ['and', 'for', 'to']
CANT_END = ['and', 'co', 'corp', 'for', 'limited', 'the', 'this', 'to']
# any line cannot have 'cant_have' words
CANT_HAVE = ['among', 'amongst', 'between', 'by and', 'date', 'dated',
             'effective', 'entered', 'for', 'this', 'vice']
CANT_BEGIN_REGEX = re.compile(r'(?:{})\b'.format('|'.join(CANT_BEGIN)))
CANT_END_REGEX = re.compile(r'\b(?:{})$'.format('|'.join(CANT_END)))
CANT_HAVE_PATTERN = r'^(?:(.*?)\s+)??(?:{})\b'
CANT_HAVE_REGEX = re.compile(CANT_HAVE_PATTERN.format('|'.join(CANT_HAVE)))

# the goal here is to find lines that might have titles
# pylint: disable=too-many-locals
def extract_lines_v2(paras_attr_list):
    lines = []  # type: List[Dict]
    start_end_list = []

    offset = 0
    is_found_party_line, unused_is_found_first_eng_para = False, False
    is_found_toc = False
    max_maybe_title_lines = 75
    # num_maybe_title_lines = -1
    num_lines_before_first_eng_para = 0
    for i, (line_st, para_attrs) in enumerate(paras_attr_list):
        if IS_DEBUG:
            attrs_st = '|'.join([str(attr) for attr in para_attrs])
            print("titles.extract_line_v2()\t{}".format('\t'.join([attrs_st,
                                                                   '[{}]'.format(line_st)])))

        line_st_len = len(line_st)

        if i >= max_maybe_title_lines or \
           'toc' in para_attrs:
            is_found_toc = True
            break
        if 'party_line' in para_attrs:
            is_found_party_line = True
            # num_maybe_title_line = i  # title cannot be in party line
            break
        if 'first_eng_para' in para_attrs:
            num_lines_before_first_eng_para = len(lines)

        # 'skip_as_template' not in para_attrs:
        line_st = regexutils.remove_non_alpha_num(line_st)
        if line_st:
            title = 1 if 'title' in para_attrs else 0
            maybe_title = 1 if 'maybe_title' in para_attrs else 0
            # end_char is exclusive-end for last line (-1 = entire line)
            line = {'line': line_st, 'start': i, 'end': i, 'end_char': -1,
                    'title': title, 'maybe_title': maybe_title}
            lines.append(line)
        # lines and start_end_list are NOT synchronized
        # start_end_list must have the same size as i
        start_end_list.append((offset, offset + line_st_len))
        offset += line_st_len + 1

    if is_found_party_line or is_found_toc:
        return lines, start_end_list

    return lines[:num_lines_before_first_eng_para], start_end_list


# pylint: disable=too-many-locals
def extract_lines_v2_old(paras_attr_list):
    lines = []
    num_lines_before_first_eng_para = 0
    offset = 0
    start_end_list = []
    for i, (line_st, para_attrs) in enumerate(paras_attr_list):
        attrs_st = '|'.join([str(attr) for attr in para_attrs])
        print("titles.extract_line_v2()\t{}".format('\t'.join([attrs_st, '[{}]'.format(line_st)])))

        line_st_len = len(line_st)

        if 'party_line' in para_attrs:
            break
        if 'first_eng_para' in para_attrs:
            num_lines_before_first_eng_para = len(lines)
        if 'toc' not in para_attrs and 'skip_as_template' not in para_attrs:
            line_st = process_as_line(line_st)
            if line_st:
                title = 1 if 'title' in para_attrs else 0
                maybe_title = 1 if 'maybe_title' in para_attrs else 0
                # end_char is exclusive-end for last line (-1 = entire line)
                line = {'line': line_st, 'start': i, 'end': i, 'end_char': -1,
                        'title': title, 'maybe_title': maybe_title}
                lines.append(line)
        start_end_list.append((offset, offset + line_st_len))
        offset += line_st_len + 1
    else:
        # No party_line found (loop did not break)
        lines = lines[:num_lines_before_first_eng_para]

    # Terminate at first cant_have word
    # pylint: disable=consider-using-enumerate
    for i in range(len(lines)):
        match = CANT_HAVE_REGEX.findall(lines[i]['line'])
        if match:
            if match[0]:
                lines[i]['line'] = match[0]
                lines[i]['end_char'] = len(match[0])
                lines = lines[:i + 1]
            else:
                lines = lines[:i]
            break

    # Bail if no lines or neither 'party_line' or 'first_eng_para' found
    if not lines:
        return lines, start_end_list

    # Some words should not start and/or end a title
    new_line = [lines[0]]
    for i in range(1, len(lines)):
        prev_cant_end = CANT_END_REGEX.search(new_line[-1]['line'])
        cant_begin = CANT_BEGIN_REGEX.match(lines[i]['line'])
        if prev_cant_end or cant_begin:
            new_line[-1]['line'] += ' ' + lines[i]['line']
            new_line[-1]['end'] = lines[i]['end']
            new_line[-1]['end_char'] = lines[i]['end_char']
            new_line[-1]['title'] *= lines[i]['title']
            new_line[-1]['maybe_title'] *= lines[i]['maybe_title']
        else:
            new_line.append(lines[i])

    #for lxline in lines:
    #    print("tt_line: {}".format(lxline))
    #for i, se in enumerate(start_end_list):
    #    print("start_end_list[{}]= {}".format(i, se))

    # Return new_line lines
    return new_line, start_end_list


def extract_lines(filepath):
    """Grab lines before party_line if it exists else before first_eng_para"""
    lines = []
    num_lines_before_first_eng_para = 0
    with open(filepath, encoding='utf-8') as fin:
        for i, line in enumerate(fin):
            tags = line.split('\t')[0].split('|')
            if 'party_line' in tags:
                break
            if 'first_eng_para' in tags:
                num_lines_before_first_eng_para = len(lines)
            if 'toc' not in tags and 'skip_as_template' not in tags:
                after_first_bracket = ''.join(line.split('[')[1:])
                between_brackets = ''.join(after_first_bracket.split(']')[:-1])
                line = process_as_line(between_brackets)
                if line:
                    title = 1 if 'title' in tags else 0
                    maybe_title = 1 if 'maybe_title' in tags else 0
                    # end_char is exclusive-end for last line (-1 = entire line)
                    line = {'line': line, 'start': i, 'end': i, 'end_char': -1,
                            'title': title, 'maybe_title': maybe_title}
                    lines.append(line)
        else:
            # No party_line found (loop did not break)
            lines = lines[:num_lines_before_first_eng_para]

    # Terminate at first cant_have word
    # pylint: disable=consider-using-enumerate
    for i in range(len(lines)):
        match = CANT_HAVE_REGEX.findall(lines[i]['line'])
        if match:
            if match[0]:
                lines[i]['line'] = match[0]
                lines[i]['end_char'] = len(match[0])
                lines = lines[:i + 1]
            else:
                lines = lines[:i]
            break

    # Bail if no lines or neither 'party_line' or 'first_eng_para' found
    if not lines:
        return lines

    # Some words should not start and/or end a title
    new_line = [lines[0]]
    for i in range(1, len(lines)):
        prev_cant_end = CANT_END_REGEX.search(new_line[-1]['line'])
        cant_begin = CANT_BEGIN_REGEX.match(lines[i]['line'])
        if prev_cant_end or cant_begin:
            new_line[-1]['line'] += ' ' + lines[i]['line']
            new_line[-1]['end'] = lines[i]['end']
            new_line[-1]['end_char'] = lines[i]['end_char']
            new_line[-1]['title'] *= lines[i]['title']
            new_line[-1]['maybe_title'] *= lines[i]['maybe_title']
        else:
            new_line.append(lines[i])

    # Return new_lines
    return new_line


"""Extract title from a file"""


NUM_TITLES_TO_MATCH = ceil(TITLES_MATCH_PCT / 100 * len(TITLES))


def title_ratio(line: str):
    line = process_as_title(line)
    if line:
        # To handle some UK documents, the score of those special cases
        # are not high enough (50's).  Set artificial high scores.
        if (line == "ownersconsent" or \
            line == "occupiersconsent"):
            return 88
        if line.startswith("leaseofland"):
            return 88
        # end of special cases

        ratios = process.extract(line,
                                 TITLES,
                                 scorer=fuzz.ratio,
                                 limit=NUM_TITLES_TO_MATCH)
        # print("titlxxx: ratios = {}".format(ratios))
        ratios = [ratio[1] for ratio in ratios]
        if 0 not in ratios:
            return stats.hmean(ratios)

    # If either s is empty after processing or a ratio was 0
    return 0


# pylint: disable=too-many-locals
def extract_title(filepath):
    # Grab lines from the file
    lines = extract_lines(filepath)
    if not lines:
        return None

    # Calculate line span as well as each line's title ratio
    line_span = lines[-1]['end'] - lines[0]['start'] + 1
    title_ratios = [title_ratio(l['line']) for l in lines]

    # Placeholder title. offsets: start, end, end_char (exclusive)
    title = {'offsets': (-1, -1, -1), 'score': -1, 'ratio': -1}

    # Find best title; all individual scores range from 0 to 1
    # pylint: disable=consider-using-enumerate
    for i in range(len(lines)):
        # pylint: disable=consider-using-enumerate
        for j in range(i, min(i + MAX_TITLE_LINES, len(lines))):
            # Ratio score
            ratios = title_ratios[i:j + 1]
            ratio_score = stats.hmean(ratios) / 100 if 0 not in ratios else 0

            # Length score
            start, end = lines[i]['start'], lines[j]['end']
            length_score = (end - start + 1) / line_span

            # Title and maybe_title scores
            all_title = all(l['title'] for l in lines[i:j + 1])
            all_maybe_title = all(l['maybe_title'] for l in lines[i:j + 1])
            title_score = length_score if all_title else 0
            maybe_title_score = length_score if all_maybe_title else 0

            # Calculated weighted score
            scores = (ratio_score, length_score, title_score, maybe_title_score)
            score = sum(w * s for w, s in zip(WEIGHTS, scores))

            # Update title if higher score
            if score > title['score']:
                title = {'offsets': (start, end, lines[j]['end_char']),
                         'score': score, 'ratio': ratio_score}

    # Return the title's start and end lines
    return title['offsets'] if title['ratio'] >= MIN_TITLE_RATIO else None

# paras_text is ignored for now
def extract_offsets(paras_attr_list, paras_text):
    # Grab lines from the file
    lines, start_end_list = extract_lines_v2(paras_attr_list)
    if not lines:
        return None, None

    if IS_DEBUG:
        for line2, start_end2 in zip(lines, start_end_list):
            print("{}\t{}".format(start_end2, line2))

    # Calculate line span as well as each line's title ratio
    line_span = lines[-1]['end'] - lines[0]['start'] + 1
    title_ratios = [title_ratio(l['line']) for l in lines]

    # Placeholder title. offsets: start, end, end_char (exclusive)
    title = {'offsets': (-1, -1, -1), 'score': -1, 'ratio': -1}

    # Find best title; all individual scores range from 0 to 1
    # pylint: disable=consider-using-enumerate
    for i in range(len(lines)):
        for j in range(i, min(i + MAX_TITLE_LINES, len(lines))):
            # Ratio score
            ratios = title_ratios[i:j + 1]
            ratio_score = stats.hmean(ratios) / 100 if 0 not in ratios else 0

            # Length score
            start, end = lines[i]['start'], lines[j]['end']
            length_score = (end - start + 1) / line_span

            # Title and maybe_title scores
            all_title = all(l['title'] for l in lines[i:j + 1])
            all_maybe_title = all(l['maybe_title'] for l in lines[i:j + 1])
            title_score = length_score if all_title else 0
            maybe_title_score = length_score if all_maybe_title else 0

            # Calculated weighted score
            scores = (ratio_score, length_score, title_score, maybe_title_score)
            score = sum(w * s for w, s in zip(WEIGHTS, scores))

            # Update title if higher score
            if score > title['score']:
                # special cases for UK, remove certain ending if there is only 1 line
                if start == end:
                    tmp_chopped_offset = lines[j]['end_char']
                    if tmp_chopped_offset == -1:
                        tmp_end_offset = start_end_list[end][1]
                    else:
                        tmp_end_offset = start_end_list[end][0] + tmp_chopped_offset
                    thx_text = paras_text[start_end_list[start][0]:tmp_end_offset]
                    # print("thx_text = [{}]".format(thx_text))
                    if thx_text.lower().startswith("lease of land at"):
                        tmp_chopped_offset = len("lease of land")
                # special cases for UK

                title = {'offsets': (start, end, tmp_chopped_offset),
                         'score': score, 'ratio': ratio_score}

    # Return the title's start and end lines
    line_start, line_end, chopped_offset = \
        title['offsets'] if title['ratio'] >= MIN_TITLE_RATIO else (None, None, None)
    if IS_DEBUG:
        print("line_start = {}, line_end = {}, chopped_offset = {}".format(line_start,
                                                                           line_end,
                                                                           chopped_offset))
    if line_start is not None:
        start_offset = start_end_list[line_start][0]
        if chopped_offset == -1:
            end_offset = start_end_list[line_end][1]
        else:
            end_offset = start_end_list[line_end][0] + chopped_offset
        return start_offset, end_offset

    return None, None


# pylint: disable=too-few-public-methods
class TitleAnnotator:

    # pylint: disable=unused-argument
    def __init__(self, provision: str) -> None:
        self.provision = 'title'

    # pylint: disable=no-self-use
    def extract_provision_offsets(self, paras_with_attrs, paras_text):
        if IS_DEBUG:
            print("title called extract_provision_offsets()")
        return extract_offsets(paras_with_attrs, paras_text)
