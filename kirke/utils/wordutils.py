
import re

ALPHA_NUM_GE2_PAT = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9]+')
def get_all_words_ge_2chars(astr: str):
    return ALPHA_NUM_GE2_PAT.findall(astr)


def is_word_overlap_ge_66p(st1: str, st2: str) -> bool:
    st1 = st1.lower()
    st2 = st2.lower()
    words1 = get_all_words_ge_2chars(st1)
    len1 = len(words1)    
    words2 = get_all_words_ge_2chars(st2)
    len2 = len(words2)
    if len1 == 0 and len2 == 0:  # 2 empty string, even after norm, cannot be empty
        return False
    if len1 > len2:  # make sure words2 is the longer one; swap
        words1, len1, words2, len2 = words2, len2, words1, len1

    if len2 <= 2:  # two strings must be exact
        return st1 == st2

    # len2 >= 3
    if float(len1) / len2 >= 0.65:  # minimal 2 out of 3 words are the same
        # 'International Business Machine' == 'International Business"
        set1 = set(words1)
        set2 = set(words2)
        overlap = set1.intersection(set2)
        # there is a possibility that there might be duplicate words in
        # set1 or set2, but ignore them for now.
        return float(len(overlap)) / len2 >= 0.65
    else:
        return False
