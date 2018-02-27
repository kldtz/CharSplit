import re

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
