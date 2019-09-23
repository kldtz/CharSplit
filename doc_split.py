#!/usr/bin/python3
# Copyright eBrevia.com 2019
"""Document-level splitting module."""

import re
import sys

import doc_config

import char_split


DE_LETTER = ('ABCDEFGHIJKLMNOPQRSTUVWXYZ' +
             'abcdefghijklmnopqrstuvwxyz' +
             'ÄÖÜäöüß')
DE_WORD_PAT = re.compile(r'\b[%s]+\b' % (DE_LETTER))
GLUES = ['ens', 'en', 'er', 'es', 'e', 'n', 's']
MIDDLE_DOT = '\N{MIDDLE DOT}'

# Caches
KNOWN_WORDS = set()
ALREADY_SPLIT_WORDS = {}

def find_root(word):
    """Looks in KNOWN_WORDS; if that fails, peels off German compound glue.
       If nothing works, return None.
    """
    if word in KNOWN_WORDS:
        return word
    # might not actually be a noun
    not_noun = word.lower()
    if not_noun in KNOWN_WORDS:
        return not_noun
    # do glue processing
    for glue in GLUES:
        if word.endswith(glue):
            root_length = len(word) - len(glue)
            root = word[:root_length]
            if root in KNOWN_WORDS:
                return word
    return None

def get_best_split(word):
    """First look up the word, and if found, return it.
       Then get the best available split of a word into 2 words.
       If no split is available, the two words are the same; return one.
    """
    root = find_root(word)
    if root:
        return [root]
    candidate = char_split.split_compound(word)[0]
    if candidate[1] == candidate[2]:
        return [candidate[1]]
    return [candidate[1], candidate[2]]

def maximal_split(word, de_dict=doc_config.DEFAULT_DICT):
    """Recursively split a single word into a list of words.
       Only the first result is split further, as CharSplit divides
       compounds into non-final and final parts.
    """
    # This is an entry point, so load the dictionary just in case.
    load_known_words(de_dict)
    word_list = get_best_split(word)
    if len(word_list) == 1:
        # Binary splitter was unable to split
        return word_list
    if len(word_list[0]) < 4 or len(word_list[1]) < 4:
        # If split product is too short, ignore the split
        return [word]
        # Recursively split the non-head and prepend it to the head
    return maximal_split(word_list[0]) + [word_list[1]]

def load_known_words(de_dict=doc_config.DEFAULT_DICT):
    """Load the dictionary into KNOWN_WORDS."""
    if KNOWN_WORDS:
        return   # already loaded
    if de_dict is None:
        de_dict = doc_config.DEFAULT_DICT
    with open(de_dict) as file:
        for word in file:
            if not (word == '' or word.startswith('#')):
                KNOWN_WORDS.add(word.strip())
    sys.stderr.write("%d known words loaded\n" % (len(KNOWN_WORDS)))

def maximal_split_str(word, de_dict=None):
    """Maximally split a word and return it with middle dots."""
    # This is an entry point, so load the dictionary just in case.
    load_known_words(de_dict)
    # if memoized, don't split
    try:
        return ALREADY_SPLIT_WORDS[word]
    except KeyError:
        pass
    upper_case = word[0].isupper()
    result_list = [w.lower() for w in maximal_split(word)]
    result_str = MIDDLE_DOT.join(result_list)
    if upper_case:
        result0 = result_str[0].upper()
    else:
        result0 = result_str[0]
    result = result0 + result_str[1:]
    ALREADY_SPLIT_WORDS[word] = result
    return result


def doc_split(doc, de_dict=None):
    """Split a whole document (a string) using the specified dictionary.
       Return the whole document with splitting dots.
    """
    if doc == '':
        return ''
    # This is an entry point, so load the dictionary just in case.
    load_known_words(de_dict)
    result = []
    windexes = [(mobj.start(), mobj.end()) for mobj in DE_WORD_PAT.finditer(doc)]
    # Non-word before the first word (OK if empty)
    result.append(doc[:windexes[0][0]])
    # pylint: disable=consider-using-enumerate
    for i in range(0, len(windexes)):
        # Add a split word and the following non-word (OK if empty)
        (start, end) = windexes[i]
        result.append(maximal_split_str(doc[start:end]))
        if i == len(windexes) - 1:
            next_start = len(doc)
        else:
            next_start = windexes[i + 1][0]
        result.append(doc[end:next_start])
    return ''.join(result)

def main():
    """Read whole document from stdin, output in maximally split format.
       Usage: ./doc_split.py dict
    """
    if len(sys.argv) > 1:
        de_dict = sys.argv[1]
    else:
        de_dict = doc_config.DEFAULT_DICT
    input_str = sys.stdin.read()
    output_str = doc_split(input_str, de_dict)
    sys.stdout.write(output_str)

if __name__ == "__main__":
    main()
