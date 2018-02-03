from fuzzywuzzy import fuzz, process
from math import ceil
import re
import string
from typing import Dict, List, Tuple

from scipy import stats

from kirke.utils import regexutils

is_debug = False

"""Config. Weights: ratio, length, title, maybe_title. Last two not reliable."""


DATA_DIR = './dict/titles/'
TITLES_MATCH_PCT = 0.387
MAX_TITLE_LINES = 5
WEIGHTS = [313, 30, 0, 0]
# this number is somewhat random, 0.5 was set before, also random.
MIN_TITLE_RATIO = 0.5


"""Process strings as lines."""


alnum = set(string.ascii_letters).union(string.digits)
non_title_labels = [r'exhibit[^A-Za-z0-9]+\d+(?:\.\d+)*',
                    r'execution[^A-Za-z0-9]+copy']
label_regexes = [re.compile(label) for label in non_title_labels]


def alnum_strip(s):
    non_alnum = set(s) - alnum
    return s.strip(str(non_alnum)) if non_alnum else s


def remove_label_regexes(s):
    for label_regex in label_regexes:
        s = label_regex.sub('', s)
    return s


def process_as_line(astr: str):
    if astr:
        # print("process_as_line({})".format(astr))
        x = alnum_strip(remove_label_regexes(alnum_strip(astr.lower())))
        # print("   return: {}".format(x))
        return x
    return astr


"""Process strings as titles."""


def regex_of(category):
    with open(DATA_DIR + category + '.list') as f:
        terms = [t.lower() for t in f.read().split('\n') if t.strip()]
    return re.compile(r'\b({})\b'.format('|'.join(terms)))


# These appeared in utils/regexutils.py also
tag_regexes0 = [(re.compile(r'\d'), 1), (regex_of('cardinals'), 4),
                (regex_of('ordinals'), 6), (re.compile(r'1+(?:st|nd|rd|th)'), 6),
                (regex_of('months'), 7), (regex_of('states'), 9)]

tag_regexes = [(regex, str(tag) * tag) for (regex, tag) in tag_regexes0]


def tag(s):
    for (regex, tag) in tag_regexes:
        s = regex.sub(tag, s)
    return s


def process_as_title(s):
    """Must lower before tag (1st) & tag before remove whitespace (New York)"""
    return ''.join(tag(alnum_strip(s.lower())).split())


"""Get and process past titles to match against."""


with open(DATA_DIR + 'past_titles_train.list') as f:
    titles = {process_as_title(t) for t in f.read().split('\n') if t.strip()}

with open(DATA_DIR + 'uk_titles_train.list') as f:
    uk_titles = {process_as_title(t) for t in f.read().split('\n') if t.strip()}

# print("titles:")
# print(titles)
titles.update(uk_titles)


"""Extract and format lines from a file"""


cant_begin = ['and', 'for', 'to']
cant_end = ['and', 'co', 'corp', 'for', 'limited', 'the', 'this', 'to']
# any line cannot have 'cant_have' words
cant_have = ['among', 'amongst', 'between', 'by and', 'date', 'dated',
             'effective', 'entered', 'for', 'this', 'vice']
cant_begin_regex = re.compile(r'(?:{})\b'.format('|'.join(cant_begin)))
cant_end_regex = re.compile(r'\b(?:{})$'.format('|'.join(cant_end)))
cant_have_pattern = r'^(?:(.*?)\s+)??(?:{})\b'
cant_have_regex = re.compile(cant_have_pattern.format('|'.join(cant_have)))

# the goal here is to find lines that might have titles
def extract_lines_v2(paras_attr_list):
    lines = []  # type: List[Dict]
    start_end_list = []

    offset = 0
    is_found_party_line, is_found_first_eng_para = False, False
    is_found_toc = False
    max_maybe_title_lines = 75
    # num_maybe_title_lines = -1
    num_lines_before_first_eng_para = 0
    for i, (line_st, para_attrs) in enumerate(paras_attr_list):
        if is_debug:
            attrs_st = '|'.join([str(attr) for attr in para_attrs])
            print("titles.extract_line_v2()\t{}".format('\t'.join([attrs_st,
                                                                   '[{}]'.format(line_st)])))

        line_st_len = len(line_st)

        if (i >= max_maybe_title_lines or \
            'toc' in para_attrs):
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
    for i in range(len(lines)):
        match = cant_have_regex.findall(lines[i]['line'])
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
        prev_cant_end = cant_end_regex.search(new_line[-1]['line'])
        cant_begin = cant_begin_regex.match(lines[i]['line'])
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
    with open(filepath, encoding='utf-8') as f:
        for i, line in enumerate(f):
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
    for i in range(len(lines)):
        match = cant_have_regex.findall(lines[i]['line'])
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
        prev_cant_end = cant_end_regex.search(new_line[-1]['line'])
        cant_begin = cant_begin_regex.match(lines[i]['line'])
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


num_titles_to_match = ceil(TITLES_MATCH_PCT / 100 * len(titles))


def title_ratio(s):
    s = process_as_title(s)
    if s:
        # To handle some UK documents, the score of those special cases
        # are not high enough (50's).  Set artificial high scores.
        if (s == "ownersconsent" or \
            s == "occupiersconsent"):
            return 88
        if s.startswith("leaseofland"):
            return 88

        # should have a set of very bad titles here
        if s == 'executionversion':
            return 0

        ratios = process.extract(s,
                                 titles,
                                 scorer=fuzz.ratio,
                                 limit=num_titles_to_match)

        # end of special cases
        # print("\ntitle_ratio(%s) = %r" % (s[:40], ratios))

        # print("titlxxx: ratios = {}".format(ratios))
        ratios = [ratio[1] for ratio in ratios]
        if 0 not in ratios:
            return stats.hmean(ratios)

    # If either s is empty after processing or a ratio was 0
    return 0


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

    if is_debug:
        for line2, start_end2 in zip(lines, start_end_list):
            print("{}\t{}".format(start_end2, line2))

    # Calculate line span as well as each line's title ratio
    line_span = lines[-1]['end'] - lines[0]['start'] + 1
    title_ratios = [title_ratio(l['line']) for l in lines]

    # Placeholder title. offsets: start, end, end_char (exclusive)
    title = {'offsets': (-1, -1, -1), 'score': -1, 'ratio': -1}

    # Find best title; all individual scores range from 0 to 1
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
    line_start, line_end, chopped_offset = title['offsets'] if title['ratio'] >= MIN_TITLE_RATIO else (None, None, None)
    if is_debug:
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


class TitleAnnotator:

    def __init__(self, provision):
        self.provision = 'title'

    def extract_provision_offsets(self, paras_with_attrs, paras_text):
        if is_debug:
            print("title called extract_provision_offsets()")
        return extract_offsets(paras_with_attrs, paras_text)

