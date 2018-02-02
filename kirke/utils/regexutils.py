import re
import string

DATA_DIR = './dict/titles/'

def regex_of(category):
    with open(DATA_DIR + category + '.list') as f:
        terms = [t.lower() for t in f.read().split('\n') if t.strip()]
    return re.compile(r'\b({})\b'.format('|'.join(terms)))


tag_regexes = [(re.compile(r'\d'), 1), (regex_of('cardinals'), 4),
               (regex_of('ordinals'), 6), (re.compile(r'1+(?:st|nd|rd|th)'), 6),
               (regex_of('months'), 7), (regex_of('states'), 9)]

tag_regexes1 = tag_regexes

tag_regexes = [(regex, str(tag) * tag) for (regex, tag) in tag_regexes]


def tag(s):
    for (regex, tag) in tag_regexes:
        s = regex.sub(tag, s)
    return s


"""
alnum = set(string.ascii_letters).union(string.digits)
non_title_labels = [r'exhibit[^A-Za-z0-9]+\d+(?:\.\d+)*',
                    r'execution[^A-Za-z0-9]+copy']
label_regexes = [re.compile(label) for label in non_title_labels]


def alnum_strip(s):
    non_alnum = set(s) - alnum
    # print('  non_alumn = [{}]'.format(non_alnum))
    if non_alnum:
        y2 = s.strip(str(non_alnum))
        # print('  y2 = [{}]'.format(y2))
        return y2
    else:
        return s


def remove_label_regexes(s):
    for label_regex in label_regexes:
        s = label_regex.sub('', s)
    return s


def process_as_line(astr: str):
    if astr:
        # print("process_as_line({})".format(astr))
        x1 = alnum_strip(astr.lower())
        # print("  x1 = [{}]".format(x1))
        x2 = remove_label_regexes(x1)
        # print("  x2 = [{}]".format(x2))
        x3 = alnum_strip(x2)
        # print("  x3 = [{}]".format(x3))
        return x3
    return astr
"""

not_alpha_num_pat = re.compile(r'[^a-zA-Z0-9]')
repeat_spaces_pat = re.compile(r'\s+')
def remove_non_alpha_num(astr: str):
    if not astr:
        return astr
    out_st = not_alpha_num_pat.sub(' ', astr.lower())
    out_st = out_st.strip()
    if '  ' in out_st:  # has at least 1 two-spaces
        out_st = repeat_spaces_pat.sub(' ', out_st)
    return out_st
        
    
