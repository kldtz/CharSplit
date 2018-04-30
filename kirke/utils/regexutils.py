import re

"""Utilities for regex."""

import re
from typing import List, Match, Pattern, Tuple


DATA_DIR = './dict/titles/'

def regex_of(category):
    with open(DATA_DIR + category + '.list') as fin:
        terms = [term.lower() for term in fin.read().split('\n') if term.strip()]
    return re.compile(r'\b({})\b'.format('|'.join(terms)))

# These appeared in utils/regexutils.py also
TAG_REGEXES0 = [(re.compile(r'\d'), 1), (regex_of('cardinals'), 4),
                (regex_of('ordinals'), 6), (re.compile(r'1+(?:st|nd|rd|th)'), 6),
                (regex_of('months'), 7), (regex_of('states'), 9)]

TAG_REGEXES1 = TAG_REGEXES0

TAG_REGEXES = [(regex, str(tag) * tag) for (regex, tag) in TAG_REGEXES0]


def tag(line: str) -> str:
    for (regex, atag) in TAG_REGEXES:
        line = regex.sub(atag, line)
    return line


NOT_ALPHA_NUM_PAT = re.compile(r'[^a-zA-Z0-9]')
REPEAT_SPACES_PAT = re.compile(r'\s+')

def remove_non_alpha_num(astr: str) -> str:
    if not astr:
        return astr
    out_st = NOT_ALPHA_NUM_PAT.sub(' ', astr.lower())
    out_st = out_st.strip()
    if '  ' in out_st:  # has at least 1 two-spaces
        out_st = REPEAT_SPACES_PAT.sub(' ', out_st)
    return out_st


# For parties

# igs = igmore_space
# cannot use '\b({})\b'.format('|'.join(word)
# because "I.B.M.', \b doesn't match the '.' at the end
# as a result, all \b happens per phrase, not phrases
def igs_escape(line: str) -> str:
    if line.endswith('.'):
        line = igs_escape_aux(line[:-1])
        line = line + r"\.*"
        # pylint: disable=line-too-long
        # https://stackoverflow.com/questions/18004955/regex-using-word-boundary-but-word-ends-with-a-period
        # Not yet used, too scary:
        # http://www.rexegg.com/regex-boundaries.html
        # Basically, we want the regex
        #    1. to match term + space or eoln (not not space)
        #    2. to match term + any non word, such as ,
        line = r'\b{}((?!\S)|(?!\w))'.format(line)
    else:
        line = r'\b{}\b'.format(igs_escape_aux(line))
    # print('escaped_line = [{}]'.format(line))
    return line

def igs_escape_aux(line: str) -> str:
    # print("line = [{}]".format(line))
    period_line = line.replace('.', '. ')
    escaped_line = re.escape(period_line)
    spaced_line = escaped_line.replace(' ', r' *')
    spaced_line = spaced_line.replace('.', r'.*')
    return spaced_line


def phrase_to_igs_pattern_st(phrase: str):
    igs_pat_st = r'({})'.format(igs_escape(phrase))
    # print("escaped_st = [{}]".format(igs_pat_st))
    return igs_pat_st


def phrases_to_igs_pattern_st(phrase_list: List[str]) -> str:
    if not phrase_list:
        return ''
    # Use set to remove redundancies first
    # sort the set in reverse order so that IBM Corp will matched before IBM.
    phrase_list = sorted(set(phrase_list), key=len, reverse=True)
    escaped_phrase_list = [igs_escape(phrase) for phrase in phrase_list]
    igs_pat_st = r'({})'.format('|'.join(escaped_phrase_list))
    # print("escaped_st = [{}]".format(igs_pat_st))
    return igs_pat_st


def phrase_to_igs_pattern(phrase: str,
                          flags: int = 0) \
                          -> Pattern[str]:
    igs_pat_st = phrase_to_igs_pattern_st(phrase)
    return re.compile(igs_pat_st, flags)


def phrases_to_igs_pattern(phrase_list: List[str],
                           flags: int = 0) \
                           -> Pattern[str]:
    if not phrase_list:
        return []
    igs_pat_st = phrases_to_igs_pattern_st(phrase_list)
    return re.compile(igs_pat_st, flags)


def find_phrase(line: str,
                phrase: str,) \
                -> List[str]:
    pat = phrase_to_igs_pattern(phrase, re.I)
    return [mat.group(1) for mat in pat.finditer(line)]


def find_phrases(line: str,
                 phrases: List[str]) \
                -> List[str]:
    pat = phrases_to_igs_pattern(phrases, re.I)
    return [mat.group(1) for mat in pat.finditer(line)]


def search_space_plus(regex_st: str, line: str, flags: int = 0) -> Match[str]:
    """re.search() with a space will match multiple spaces."""
    spcplus_st = regex_st.replace(' ', r'\s+')
    # must begin and end word boundaries
    b_bound_st = r'\b{}\b'.format(spcplus_st)
    return re.search(b_bound_st, line, flags)


def match_space_plus(regex_st: str, line: str, flags: int = 0) -> Match[str]:
    """re.search() with a space will match multiple spaces."""
    spcplus_st = regex_st.replace(' ', r'\s+')
    # must end word boundaries
    b_bound_st = r'{}\b'.format(spcplus_st)
    return re.match(b_bound_st, line, flags)
