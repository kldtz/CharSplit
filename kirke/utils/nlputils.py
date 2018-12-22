# pylint: disable=too-many-lines

from enum import Enum
import re

# pylint: disable=unused-import
from typing import (Any, Dict, Generator, List, Match, Optional, Pattern,
                    Set, Tuple, Union)

import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.chunk import RegexpParser
from nltk.tree import Tree

from kirke.utils import mathutils, regexutils, strutils

IS_DEBUG_CHUNK = False
IS_DEBUG_ORGS_TERM = False
IS_DEBUG_RERANK_DEFINED_TERM = False

# IS_DEBUG_CHUNK = True
# IS_DEBUG_ORGS_TERM = True
# IS_DEBUG_RERANK_DEFINED_TERM = True

# bank is after 'n.a.' because 'bank, n.a.' is more desirable
# 'Credit Suisse Ag, New York Branch', 39893.txt,  why 'branch' is early
# pylint: disable=fixme
# TODO, handle "The bank of Nova Scotia", this is NOT org suffix case
# pylint: disable=fixme
# TODO, not handling 'real estate holdings fiv'
# pylint: disable=fixme
# TODO, remove 'AS AMENDED' as a party, 'the customer'?
# pylint: disable=fixme
# TODO, 'seller, seller' the context?
ORG_SUFFIX_LIST = strutils.load_non_empty_str_list('dict/parties/organization.suffix.list')
PERS_SUFFIX_LIST = strutils.load_non_empty_str_list('dict/parties/person.suffix.list')

ORG_PERSON_SUFFIX_LIST = list(ORG_SUFFIX_LIST)
ORG_PERSON_SUFFIX_LIST.extend(PERS_SUFFIX_LIST)

# copied from kirke/ebrules/parties.py on 2/4/2016
ORG_PERSON_SUFFIX_PAT = regexutils.phrases_to_igs_pattern(ORG_PERSON_SUFFIX_LIST, re.I)
ORG_PERSON_SUFFIX_END_PAT = \
    re.compile(regexutils.phrases_to_igs_pattern_st(ORG_PERSON_SUFFIX_LIST) + r'\s*$', re.I)

ORG_PREFIX_LIST = strutils.load_non_empty_str_list('dict/parties/organization.prefix.list')
ORG_PREFIX_PAT = regexutils.phrases_to_igs_pattern(ORG_PREFIX_LIST, re.I)

LOCATION_LIST = strutils.load_non_empty_str_list('dict/parties/location.list')
LC_LOCATION_SET = set([loc.lower() for loc in LOCATION_LIST])

# pylint: disable=line-too-long
US_STATE_ABBRV_INCOMPLETE = strutils.load_non_empty_str_list('dict/parties/us_state_abbr.incomplete.list')
US_STATE_SET = set(US_STATE_ABBRV_INCOMPLETE)

DEFINED_TERM_WORDS = strutils.load_non_empty_str_list('dict/parties/defined_term.words')
DEFINED_TERM_WORD_SET = set(DEFINED_TERM_WORDS)

UK_STD = r'[A-Z]{1,2}[0-9R][0-9A-Z]? (?:(?![CIKMOV])[0-9][a-zA-Z]{2})'
UK_ZIP_PAT = re.compile(r'\b' + UK_STD + r'\b')


def find_uk_zip_code(line: str) -> bool:
    return bool(UK_ZIP_PAT.search(line))


def is_us_state_abbrv(line: str) -> bool:
    return line in US_STATE_SET


def is_a_location(line: str) -> bool:
    if not line:
        return False

    if not strutils.is_digits(line[0]) and not line[0].isupper():  # must capitalize
        return False

    if find_uk_zip_code(line):
        return True

    words = line.split()
    if is_us_state_abbrv(words[-1]):
        # xxxx MA
        return True
    if words[-1].lower() in set(['avenue', 'ave', 'ave.',
                                 'drive', 'dr', 'dr.',
                                 'road', 'rd', 'rd.',
                                 'street', 'st', 'st.']):
        return True

    return line.lower() in LC_LOCATION_SET


# print("org_person_suffix_pattern_st:")
# print(regexutils.phrases_to_igs_pattern_st(ORG_PERSON_SUFFIX_LIST))

def org_person_suffix_search(line: str) -> Optional[Match[str]]:
    return ORG_PERSON_SUFFIX_PAT.search(line)


def org_person_suffix_match(line: str) -> Optional[Match[str]]:
    return ORG_PERSON_SUFFIX_PAT.match(line)


def is_invalid_org_suffix_aux(line: str) -> bool:
    """Verify line (not lc'ed) is a real org_suffix."""
    if line == 'corporation':
        # 'a Virgin Island corporation'
        return True
    elif line in set(['NV', 'CO']):
        # 'Las Vegas, NV'
        return True
    return False


def is_org_suffix(line: str) -> bool:
    # print("is_org_suffix({})".format(line))
    if line.lower() == 'partnership':
        # line is sometimes just one word, 'limited partnership'
        # Of course, it can be other type of 'xxx partnership', but
        # worry about this later.
        return True
    elif is_invalid_org_suffix_aux(line):
        return False
    return bool(ORG_PERSON_SUFFIX_END_PAT.match(line))


def is_valid_org_and_prefix(words: List[str]) -> bool:
    if len(words) == 1 and \
       words[0] in set(['Johnson']):
        return True
    if words[-1] in set(['Culture', 'Information', 'Light',
                         # 'Light and Magic'
                         'Media',
                         'Science', 'Services', 'Technology']):
        return True
    return False


def get_org_prefix_mat_list(line: str) -> List[Match[str]]:
    """Get all org prefix matching mat extracted from line.

    When there are multiple orgs that are consecutive, merge them.
    """

    lc_mat_list = list(ORG_PREFIX_PAT.finditer(line))
    result = []  # type: List[Match[str]]
    for lc_mat in lc_mat_list:
        mat_st = line[lc_mat.start():lc_mat.end()]
        if mat_st[0].isupper():
            result.append(lc_mat)

    return result


def find_org_suffix_mat_list_raw(line: str) -> List[Match[str]]:
    """
    Find all the org suffix matches in a string.

    :param line: str
    :return: List[Match[str]]
    """
    lc_mat_list = list(ORG_PERSON_SUFFIX_PAT.finditer(line))
    return lc_mat_list


def get_org_suffix_mat_list(line: str) -> List[Match[str]]:
    """Get all org suffix matching mat extracted from line.

    Because of capitalization concerns, the word before org
    must not be lower cased, such as 'a limited company'.

    When there are multiple orgs that are consecutive, merge them.
    """

    lc_mat_list = list(ORG_PERSON_SUFFIX_PAT.finditer(line))

    #for lc_mat in lc_mat_list:
    #    mat_st = line[lc_mat.start():lc_mat.end()]
    #    print("get_org_suffix cand1: [{}]".format(mat_st))

    result = []  # type: List[Match[str]]
    for lc_mat in lc_mat_list:
        prev_space_idx = lc_mat.start() -1
        # the previous word must be capitalized
        pword_start, unused_pword_end, pword = strutils.find_previous_word(line, prev_space_idx)
        if pword_start != -1:
            if pword[0].isupper():
                result.append(lc_mat)
            elif pword.isdigit():
                # 'Partner 4, LLC', doc102.txt
                result.append(lc_mat)
            elif strutils.is_both_alpha_and_num(pword):
                # 'ante4, Inc', 40331.txt
                result.append(lc_mat)
            elif strutils.is_cap_not_first_char(pword):
                # 'eBrevia'
                result.append(lc_mat)

    result_st_list = []
    for lc_mat in result:
        mat_st = line[lc_mat.start():lc_mat.end()]
        result_st_list.append(mat_st)
        # print("get_org_suffix cand2: [{}]".format(mat_st))

    # when there is adjcent ones, take the last one
    # 'xxx Group, Ltd.', will return 'ltd'
    prev_mat = None
    result2 = [] # type: List[Match[str]]
    # Only if we now that the current mat is not adjacent to
    # the previous mat, we can add previous mat.
    # Remember the last one.
    for amat, mat_st in zip(result, result_st_list):
        # a Delaware corporation
        if mat_st.islower():
            continue

        elif is_invalid_org_suffix_aux(mat_st):
            # 'Cold Spring, CO'
            continue

        # 2 is chosen, just in case, normally the diff is 1
        if prev_mat and \
           (amat.start() - prev_mat.end() > 2 or \
            mat_st.lower() in set(['bank', 'banco', 'banc'])):
            result2.append(prev_mat)
        prev_mat = amat
    if prev_mat:
        result2.append(prev_mat)

    return result2




# ?= lookahead assertion, doesn't consume
# ?: non-capturing
# checking if spaces followed by capitalized letter or uncap letter by cap letter
# cannot add quotes and '(', example: '(a) The Princeton Review, Inc. (the “Issuer”),...'
SENTENCE_DOES_NOT_CONTINUE = r'(?=\s+(?:[A-Z0-9].|[a-z][A-Z0-9]))'
# ?<! negative lookahead assertion, not preceded by \w\. or \w\w\.
NOT_FEW_LETTERS = r'(?<!\b[A-Za-z]\.)(?<!\b[A-Za-z]{2}\.)'
# not preceded by No\. or Nos\.
NOT_NUMBER = r'(?<!\bN(O|o)\.)(?<!\bN(O|o)(S|s)\.)'
NOT_ACRONYM = r'(?<!\bS(T|t)(E|e)\.)'
# period followed by above patterns
REAL_PERIOD = r'\.' + SENTENCE_DOES_NOT_CONTINUE + NOT_FEW_LETTERS + NOT_NUMBER + NOT_ACRONYM
FIRST_SENT_PAT_ST = r'(.*?' + REAL_PERIOD + ')'
FIRST_SENT_PAT = re.compile(FIRST_SENT_PAT_ST)

# print("first_sent regex = {}".format(r'(.*?' + REAL_PERIOD + ')'))

def first_sentence(text: str) -> str:
    """Trying to avoid sentence tokenizing since this occurs before CoreNLP.

    Author: Jason Ma
    """
    text = text.replace('\n', ' ')
    match = FIRST_SENT_PAT.search(text)
    return match.group() if match else text


def jm_sent_tokenize(text: str) -> List[Tuple[int, int]]:
    if not text.strip():  # if empty or only spaces
        return []
    after_text = text.replace('\n', ' ')
    text_len = len(after_text)
    result = []
    offset = 0
    mat = FIRST_SENT_PAT.search(after_text)
    while mat:
        # non-space_index cannnot be -1
        non_space_index = strutils.find_non_space_index(mat.group())
        result.append((offset + non_space_index,
                       offset + mat.end()))
        after_text = after_text[mat.end():]
        offset += mat.end()
        mat = FIRST_SENT_PAT.search(after_text)
    if offset < text_len and after_text.strip():
        # non-space_index cannnot be -1
        non_space_index = strutils.find_non_space_index(after_text)
        result.append((offset + non_space_index,
                       text_len))
    return result

# an alias
# pylint: disable=invalid-name
sent_tokenize = jm_sent_tokenize

TREEBANK_WORD_TOKENIZER = TreebankWordTokenizer()
WORD_PUNCT_TOKENIZER = WordPunctTokenizer()

class PunctType(Enum):
    COMMA = 1
    PERIOD = 2
    LEFT_PAREN = 1
    RIGHT_PAREN = 2
    OTHER = 15

class WordTokenType(Enum):
    LETTER = 1
    NUMBER = 2
    ALPHANUM = 3
    PUNCT = 4
    DATE = 5
    OTHER = 10

def span_tokenize(sent_line: str) -> List[Tuple[int, int]]:
    """Returns all the tokens for a sentence.

    Since TreeBankWordTokenizer assume the input is a sentence, we do the same.
    The sent_line MUST be a sentence, otherwise, might not perform as expected.
    """
    # There is a bug in nltk.  Following sentence cause it to fail.
    # tokenze() work, but not span_tokenize().
    # Temporary solution is to replace normal '"' with '“'
    # pylint: disable=line-too-long
    # line = 'ContentX Technologies, EEC a California limited liability company  with an address at 19700 Fairchild, Ste. 260, Irvine, CA 92612 (“ContentX”! and Cybeitnesh International  Corporation, a Nevada corporation with an address at 2715 Indian Farm Lh NW, Albuquerque NM 87107  (“Cybermesh”) (Cybermesh and together with ContentX, the “Members" each a “Member”), pursuant to  the laws of the State of Nevada.'

    # tried it with '"', didn't work.  But this does.
    sent_line = re.sub('[“"”]', '“', sent_line)
    # for OCR error ("ContectX"! -> ("ContectX")
    sent_line = re.sub(r'(\(“[^\)]+“)!', r'\1)', sent_line)

    return TREEBANK_WORD_TOKENIZER.span_tokenize(sent_line)


def tokenize(sent_line: str) -> List[str]:
    """Returns all the span tokens for a sentence."""
    return TREEBANK_WORD_TOKENIZER.tokenize(sent_line)

# pylint: disable=invalid-name
word_tokenize = tokenize


def text_span_tokenize(text: str) -> List[Tuple[int, int]]:
    """Returns all the span tokens for a text.

    The text can have multiple sentences.
    """
    sent_se_list = sent_tokenize(text)
    result = []
    for sstart, send in sent_se_list:
        token_se_list = TREEBANK_WORD_TOKENIZER.span_tokenize(text[sstart:send])
        for token_start, token_end in token_se_list:
            result.append((sstart + token_start, sstart + token_end))
    return result

def text_tokenize(text: str) -> List[str]:
    """Returns all the tokens for a text.

    The text can have multiple sentences.
    """
    sent_se_list = sent_tokenize(text)
    result = []  # type: List[str]
    for sstart, send in sent_se_list:
        result.extend(tokenize(text[sstart:send]))
    return result


def word_punct_tokenize(line: str) -> List[str]:
    se_tokens = WORD_PUNCT_TOKENIZER.tokenize(line)
    return se_tokens


# pylint: disable=too-many-branches
def fix_nltk_pos(wpos_list: List[Tuple[str, str]],
                 t2_map: Dict[str, Tuple[str, str]] = None) -> List[Tuple[str, str]]:
    result = []
    for widx, wpos in enumerate(wpos_list):
        word, tag = wpos
        lc_word = word.lower()
        if lc_word == 'hereto':
            result.append((word, 'IN'))
        elif lc_word == 'registered':
            result.append((word, 'VPD'))
        elif lc_word == 'the':  # 'THE" is attached as 'NNP' sometimes
            result.append((word, 'DT'))
        elif lc_word == 'its':
            result.append((word, 'PRP$'))
        # "Pass Through Trustee", export-train/29749.txt
        # elif lc_word == 'through':
        #    result.append((word, 'IN'))
        elif word.startswith('ZqZ'):
            if t2_map:
                # A token can be 'Corporation.' from 'Corporation.,'
                unused_line, ttype = t2_map[word[:7]]
                if ttype == 'xABN':
                    # we don't want this to merge with anything
                    result.append((word, 'IN'))
                # cannot do this because span_se_list will be out
                # of sync
                # elif len(word) > 7 and word[7] == '.':
                #    result.append((word[:8], 'NNP'))
                #    # there are strange errors in input
                #    # CO.12345   with org.digits
                #    result.append((word[8:], 'NN'))
                else:
                    result.append((word, 'NNP'))
            else:
                result.append((word, 'NNP'))
        elif strutils.is_both_alpha_and_num(word):
            result.append((word, 'NNP'))
        elif strutils.is_cap_not_first_char(word):
            # capitalize the first word for now
            result.append((word.capitalize(), 'NNP'))
        elif word == 'I' and tag == 'PRP' and \
             widx + 1 < len(wpos_list) and \
             (wpos_list[widx + 1][0] == ',' or
              wpos_list[widx + 1][0].startswith('ZqZ')):
            # 'XXX Fund I, L.P.'
            result.append((word, 'NNP'))
        # elif is_org_suffix(word):
        #    result.append((word, 'xNNP'))
        else:
            result.append((word, tag))
    return result


def pos_tag(tokens: List[str], t2_map: Dict[str, Tuple[str, str]] = None) \
    -> List[Tuple[str, str]]:
    postag_list = nltk.pos_tag(tokens)
    fixed_list = fix_nltk_pos(postag_list, t2_map)
    return fixed_list

# pylint: disable=fixme
# TODO: maybe remove in future
# NLTK_TOKENIZER = WhitespaceTokenizer()
# This doesn't handle number correctly
NLTK_TOKENIZER = RegexpTokenizer(r'\w[\w\.]+|(\$|\#)?[\d\.]+|\S')

def word_comma_tokenize(line: str) -> Generator[Tuple[int, int, str], None, None]:
    """Return a list of word, with comma.

    Note: all periods,include those for abbreviation, "I.B.M.", and end of sentence
    ending, "war." are a part of the word.  We assume the line is already a string
    """
    for start, end in NLTK_TOKENIZER.span_tokenize(line):
        word = line[start:end]
        if word[0].isalnum():
            if word[-1] == '.' and not word[0].isupper():  # end of sentence
                yield start, end-1, word[:-1]
            else:
                yield start, end, word
        elif re.search(r'\d', word):
            yield start, end, word
        else:
            if word == ',' or \
               word == '?' or \
               word == '!':
                yield start, end, word
            # otherwise, just ignore all other punctuations

# the CD in xNNP toward the end is for
# 'xxx 14 LTD'
chunker = RegexpParser(r'''
xPAREN:
  {<\(.*><[^\(].*>*<\).*>}
xNNP:
  {<DT>?<JJ.*|CD>*<NN.*>*<NNP>+<CD>?<NN.*>*}
xNN:
  {<DT|PRP.*>?<JJ.*|CD>*<NN.*>+}
  }<VB.*>{
  }<,.*>{
  }<IN.*>{
  }<CC.*>{
''')


def find_known_terms(line: str) -> List[Tuple[int, int, str]]:
    result = []  # type: List[Tuple[int, int, str]]

    mat_list = get_org_prefix_mat_list(line)
    for mat in mat_list:
        result.append((mat.start(), mat.end(), 'xPV_ORG'))

    mat_list = get_org_suffix_mat_list(line)
    for mat in mat_list:
        result.append((mat.start(), mat.end(), 'xORGP'))

    for mat in re.finditer(r'ABN( \d+)+', line, re.I):
        result.append((mat.start(), mat.end(), 'xABN'))

    # print("find_known_terms before: {}".format(result))
    # for i, se_term in enumerate(result):
    #     start, end, term = se_term
    #    print("before term #{}\t[{}]\t{}".format(i, line[start:end], se_term))

    result = mathutils.remove_subsumed(result)
    if IS_DEBUG_CHUNK:
        print("\nfind_known_terms after: {}".format(result))
        for i, se_term in enumerate(result):
            start, end, unused_term = se_term
            print("after term #{}\t[{}]\t{}".format(i, line[start:end], se_term))
    return sorted(result)

def sub_known_terms(line: str,
                    se_ttype_list: List[Tuple[int, int, str]]) \
                    -> Tuple[str, Dict[str, Tuple[str, str]]]:
    known_term_map = {}  # type: Dict[str, Tuple[str, str]]
    kt_count = 0
    parts = []  # type: List[str]
    offset = 0
    for se_ttype in se_ttype_list:
        start, end, kt_type = se_ttype
        orig_span = line[start:end]
        kt_sub = 'ZqZ{:04d}'.format(kt_count)
        kt_count += 1
        known_term_map[kt_sub] = (orig_span, kt_type)

        parts.append(line[offset:start])
        parts.append(kt_sub)
        offset = end
    parts.append(line[offset:])

    return ''.join(parts), known_term_map

def putback_chunk_to_postag_list(chunk: Union[Tree, Tuple[str, str]],
                                 t2_map: Dict[str, Tuple[str, str]]) \
    -> Tuple[bool, Optional[str], List[Tuple[str, str]]]:
    postags = chunk_to_postag_list(chunk)
    postags_out = []  # type: List[Tuple[str, str]]
    is_triggered = False
    ttype = None
    for word_pos in postags:
        word, unused_pos = word_pos
        if word.startswith('ZqZ'):
            # can have extra stuff due to 'Corporation.' in 'Corporation.,'
            orig_term, ttype = t2_map[word[:7]]
            term_tok_list = word_tokenize(orig_term)

            # probably 'N.A.' -> 'N.A' '.'
            # merge the last 2 tokens
            if len(term_tok_list) > 1 and \
                            term_tok_list[-1] == '.':
                term_tok_list[-2] = term_tok_list[-2] + '.'
                term_tok_list = term_tok_list[:-1]
            elif len(word) > 7:
                # can have extra stuff due to 'Corporation.' in 'Corporation.,'
                term_tok_list[-1] = term_tok_list[-1] + word[7:]
            # for term_tok in term_tok_list:
            #     print("fixed term_tok: [{}]".format(term_tok))
            postags_out.extend([(term_tok, 'NNP') for term_tok in term_tok_list])
            is_triggered = True
        else:
            postags_out.append(word_pos)

    return is_triggered, ttype, postags_out


def split_jj_xnnp_chunk_aux(chunk: Union[Tree, Tuple[str, str]]) \
            -> Optional[Tuple[Union[Tree, Tuple[str, str]],
                              Union[Tree, Tuple[str, str]]]]:
    # we know the chunk is already xnnp
    postags = chunk_to_postag_list(chunk)
    # postags_out = []  # type: List[Tuple[str, str]]
    jj_index = -1
    split_index = -1
    for i, word_pos in enumerate(postags):
        word, pos = word_pos
        # if there are multiple JJ, take the last one
        if word.islower() and pos == 'JJ':
            jj_index = i
        # take the first NNP
        if pos == 'NNP':
            split_index = i
            break
    if jj_index != -1 and split_index != -1:
        first_postags = postags[:split_index]
        second_postags = postags[split_index:]
        return (Tree('xNN', first_postags),
                Tree('xNNP', second_postags))
    return None


def split_jj_xnnp_chunk(chunk_list: List[Union[Tree, Tuple[str, str]]]) \
            -> List[Union[Tree, Tuple[str, str]]]:
    """Split xNNP chunk that has mixed lc words and org, such as
       "the undersigned ABC DEF Limited Partnership".

       They should be 'the undersigned' and 'ABC DEF Limited Parnership' instead
       of becoming an xNNP.
    """
    result = []  # type: List[Union[Tree, Tuple[int, int]]]
    for chunk in chunk_list:
        if is_chunk_label(chunk, 'xNNP'):
            jj_xnnp_t2 = split_jj_xnnp_chunk_aux(chunk)
            if jj_xnnp_t2:
                jj_chunk, xnnp_chunk = jj_xnnp_t2
                result.append(jj_chunk)
                result.append(xnnp_chunk)
            else:
                result.append(chunk)
        else:
            result.append(chunk)
    return result


def putback_kterms(chunk_list: List[Union[Tree, Tuple[str, str]]],
                   t2_map: Dict[str, Tuple[str, str]]) -> List[Union[Tree, Tuple[int, int]]]:
    result = []  # type: List[Union[Tree, Tuple[int, int]]]
    for chunk in chunk_list:
        if is_chunk_label(chunk, 'xNNP'):
            is_triggered, ttype, postags_out = putback_chunk_to_postag_list(chunk, t2_map)
            if is_triggered:
                chunk = Tree(ttype, postags_out)
        elif is_chunk_label(chunk, 'xPAREN'):
            # it is possible that zqz is in xPAREN also
            is_triggered, ttype, postags_out = putback_chunk_to_postag_list(chunk, t2_map)
            if is_triggered:
                # to make mypy happy for type checking
                tmp_chunk = chunk # type: Tree
                chunk = Tree(tmp_chunk.label(), postags_out)
        elif not is_chunk_tree(chunk):
            word = postag_word(chunk)
            # only for ABN XXX
            if word.startswith('ZqZ'):
                orig_term, ttype = t2_map[word]
                if ttype == 'xABN':
                    term_tok_list = word_tokenize(orig_term)

                    postags_out = [(term_tok, 'NNP') for term_tok in term_tok_list]
                    chunk = Tree(ttype, postags_out)

        result.append(chunk)

    return result

def split_chunk_with_org(chunk_list: List[Union[Tree, Tuple[str, str]]]) \
    -> List[Union[Tree, Tuple[str, str]]]:
    result = []  # type: List[Union[Tree, Tuple[str, str]]]
    for chunk in chunk_list:
        if is_chunk_label(chunk, 'xORGP'):
            postag_list = chunk_to_postag_list(chunk)

            # 'the Canada Business Corporation Act'
            last_word, unused_last_tag = postag_list[-1]
            if isinstance(chunk, Tree) and \
               last_word.lower() == 'act' and last_word[0] == 'A':
                chunk.set_label('NNP')
                result.append(chunk)
                continue

            found_org_suffix_idx = -1
            for i in range(len(postag_list)-1, -1, -1):
                word, unused_tag = postag_list[i]
                # 'WASTE2ENERGY GROUP company registed in ...', doc118.txt
                if is_org_suffix(word) and \
                   not word in set(['company', 'corporation']):
                    found_org_suffix_idx = i
                    break

            if found_org_suffix_idx == -1 or \
                found_org_suffix_idx == len(postag_list) -1:
                result.append(chunk)
            else:
                result.append(Tree('xORGP', postag_list[:found_org_suffix_idx+1]))
                result.append(Tree('NNP', postag_list[found_org_suffix_idx+1:]))
        else:
            result.append(chunk)

    return result


def chunkize_by_regex_grammar(sent_line: str) -> List[Union[Tree, Tuple[str, str]]]:
    se_term_list = find_known_terms(sent_line)
    sent_line_t2, t2_map = sub_known_terms(sent_line,
                                           se_term_list)
    tokens = word_tokenize(sent_line_t2)
    tok_pos_list = pos_tag(tokens, t2_map)

    if IS_DEBUG_CHUNK:
        for i, tok_pos in enumerate(tok_pos_list):
            print('pos_tok #{}\t{}'.format(i, tok_pos))

    chunk_list = chunker.parse(tok_pos_list)

    chunk_list = split_jj_xnnp_chunk(chunk_list)

    chunk_list = putback_kterms(chunk_list,
                                t2_map)

    # split chunks that has org in the middle
    # 'Johnson & Johnson Medikal Sanayi Ve Ticaret Limited Sirketi'
    # This also handles "the Canada Business Corporation Act", which
    # shouldn't split on org.
    chunk_list = split_chunk_with_org(chunk_list)

    if IS_DEBUG_CHUNK:
        for i, chunk in enumerate(chunk_list):
            print("chunkize #{}\t{}".format(i, chunk))

    return chunk_list


# def get_chunk_postag_list(chunk: Tree) -> List[Tuple[int, int]]:
#    return [postag for postag in chunk]

def rechunk_known_terms(chunk_list: List[Union[Tree, Tuple[int, int]]]) \
    -> List[Union[Tree, Tuple[int, int]]]:
    result = []  # List[Union[Tree, Tuple[int, int]]]
    idx = 0
    while idx < len(chunk_list):
        chunk = chunk_list[idx]
        next_chunk = get_next_chunk(chunk_list, idx)
        if is_chunk_tree(chunk):
            postag_list = chunk_to_postag_list(chunk)
            if postag_word(postag_list[-1]) == 'ABN' and \
                next_chunk and is_chunk_digit(next_chunk):
                abn_tok_list = [postag_list[-1], next_chunk]
                tmp_i = idx + 1  # this is next chunk idx
                tmp_next_chunk = get_next_chunk(chunk_list, tmp_i)
                while tmp_next_chunk and is_chunk_digit(tmp_next_chunk):
                    abn_tok_list.append(tmp_next_chunk)
                    tmp_i += 1
                    tmp_next_chunk = get_next_chunk(chunk_list, tmp_i)
                # to make mypy happy for type checking
                tmp_chunk = chunk  # type: Tree
                result.append(Tree(tmp_chunk.label(), postag_list[:-1]))
                result.append(Tree('ABN_NUM', abn_tok_list))
                idx = tmp_i
            else:
                result.append(chunk)
        else:
            result.append(chunk)
        idx += 1
    return result


def get_nouns(sent_line: str):
    chunk_list = chunkize_by_regex_grammar(sent_line)
    result = []  # List[Tree]
    for chunk in chunk_list:
        if isinstance(chunk, Tree):
            # print('tree #{}\t{}'.format(i, tree))
            result.append(chunk)
        else:
            # print('pos_tag #{}\t{}'.format(i, tree))
            pass

    return result

def get_proper_nouns(sent_line: str):
    chunk_list = chunkize_by_regex_grammar(sent_line)
    result = []  # List[Tree]
    for chunk in chunk_list:
        if isinstance(chunk, Tree) and \
           chunk.label() == 'xNNP':
            # print('tree #{}\t{}'.format(i, tree))
            result.append(chunk)
        else:
            # print('pos_tag #{}\t{}'.format(i, tree))
            pass

    return result

def is_postag_comma(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    return chunk_is_postag(chunk) and is_postag_tag(chunk, ',')

# create an alias
is_chunk_comma = is_postag_comma

def has_pos_article(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    if isinstance(chunk, Tree):
        for postag in chunk:
            if is_postag_tag(postag, 'DT'):
                return True
        return False
    return is_postag_tag(chunk, 'DT')

def chunk_has_jj(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    if isinstance(chunk, Tree):
        for postag in chunk:
            if is_postag_tag(postag, 'JJ'):
                return True
        return False
    return is_postag_tag(chunk, 'JJ')

def get_next_chunk(chunk_list: List[Union[Tree, Tuple[str, str]]], idx):
    if idx + 1 < len(chunk_list):
        return chunk_list[idx + 1]
    return None

def is_chunk_tree(chunk: Optional[Union[Tree, Tuple[str, str]]]) -> bool:
    if not chunk:
        return False
    return isinstance(chunk, Tree)

def get_prev_chunk(chunk_list, idx):
    if idx - 1 >= 0:
        return chunk_list[idx - 1]
    return None

def is_chunk_digit(chunk: Optional[Union[Tree, Tuple[str, str]]]) -> bool:
    if not chunk:
        return False
    if isinstance(chunk, Tree):
        return False
    return is_postag_tag(chunk, 'CD')

def is_chunk_and(chunk: Optional[Union[Tree, Tuple[str, str]]]) -> bool:
    if not chunk:
        return False
    if isinstance(chunk, Tree):
        return False
    # cannot check for tag == 'CC' because it can be 'or'
    word = postag_word(chunk)
    return word.lower() == 'and' or word == '&'

def is_chunk_of(chunk: Optional[Union[Tree, Tuple[str, str]]]) -> bool:
    if not chunk:
        return False
    if isinstance(chunk, Tree):
        return False
    # cannot check for tag == 'CC' because it can be 'or'
    word = postag_word(chunk)
    return word.lower() == 'of'

def is_chunk_ampersand(chunk: Optional[Union[Tree, Tuple[str, str]]]) -> bool:
    if not chunk:
        return False
    if isinstance(chunk, Tree):
        return False
    # cannot check for tag == 'CC' because it can be 'or'
    word = postag_word(chunk)
    return word == '&'

def is_chunk_paren(chunk: Optional[Union[Tree, Tuple[str, str]]]) -> bool:
    if not chunk:
        return False
    if isinstance(chunk, Tree):
        return chunk.label() == 'xPAREN'
    return False

def is_chunk_xnnp(chunk: Optional[Union[Tree, Tuple[str, str]]]) -> bool:
    if not chunk:
        return False
    if isinstance(chunk, Tree):
        return chunk.label() == 'xNNP' and not has_pos_article(chunk)
    return is_postag_tag(chunk, 'NNP')

def is_chunk_label(chunk: Optional[Union[Tree, Tuple[str, str]]], label: str) -> bool:
    if not chunk:
        return False
    if isinstance(chunk, Tree):
        return chunk.label() == label
    return False

def chunk_num_tokens(chunk: Union[Tree, Tuple[str, str]]) -> int:
    if not chunk:
        return 0
    elif isinstance(chunk, Tree):
        return len(chunk)
    return 1

def is_chunk_org(chunk: Optional[Union[Tree, Tuple[str, str]]]) -> bool:
    if not chunk:
        return False
    return isinstance(chunk, Tree) and \
        chunk.label() == 'xORGP'

def is_chunk_a_corporation(chunk: Optional[Union[Tree, Tuple[str, str]]]) -> bool:
    if not chunk:
        return False
    if isinstance(chunk, Tree):
        if chunk.label() == 'xNNP':
            words = chunk_to_words(chunk)
            if len(words) >= 2 and \
               words[0] == 'a' and \
               words[-1] == 'corporation':
                return True
    return False

def is_abbrev_word(line: str) -> bool:
    """Verify if a word is an abbreviation."""
    return len(line) == 2 and line[0].isupper() and line[1] == '.'


def is_chunk_likely_person_name(chunk: Optional[Union[Tree, Tuple[str, str]]]) -> bool:
    if not chunk:
        return False
    if isinstance(chunk, Tree):
        if chunk.label() == 'xNNP':
            words = chunk_to_words(chunk)
            # must have an abbreviation
            if len(words) >= 3 and \
               strutils.is_all_title_words(words) and \
               (is_abbrev_word(words[0]) or
                is_abbrev_word(words[1]) or
                is_abbrev_word(words[2])):
                return True
            # in future, can check for name-like word, such as
            # 'James' or 'Dennis', or "Mary'
    return False



def is_chunk_xpv_org(chunk: Optional[Union[Tree, Tuple[str, str]]]) -> bool:
    if not chunk:
        return False
    return isinstance(chunk, Tree) and \
        chunk.label() == 'xPV_ORG'

def is_postag_tag(postag: Optional[Tuple[str, str]], tag: str) -> bool:
    if not postag:
        return False
    return postag[1] == tag

def postag_word(postag: Tuple[str, str]) -> str:
    return postag[0]

def postag_tag(postag: Tuple[str, str]) -> str:
    return postag[1]

def is_zipcode(line: str) -> bool:
    return len(line) == 5 and line.isdigit()

# assume all the chunk is either Tree or postag (Tuple[str, str])
def chunk_is_postag(obj):
    return not isinstance(obj, Tree)

def chunk_to_postag_list(chunk: Union[Tree, Tuple[str, str]]) -> List[Tuple[str, str]]:
    if isinstance(chunk, Tree):
        return [postag for postag in chunk]
    return [chunk]

def chunk_to_words(chunk: Union[Tree, Tuple[str, str]]) -> List[str]:
    postag_list = chunk_to_postag_list(chunk)
    return [postag[0] for postag in postag_list]

def is_chunk_org_suffix_only(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    words = chunk_to_words(chunk)
    term = ' '.join(words)
    # assume there is one org suffix
    if is_org_suffix(term):
        return True
    org_term_list = find_org_suffix_mat_list_raw(term)
    if len(words) == len(org_term_list):
        return True
    return False

def is_chunk_all_caps(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    words = chunk_to_words(chunk)
    if not words:
        return False
    return strutils.is_all_upper_words(words)

def chunk_has_words(chunk: Union[Tree, Tuple[str, str]], word_set: Set[str]) -> bool:
    words = chunk_to_words(chunk)
    for word in words:
        if word.lower() in word_set:
            return True
    return False


# def chunks_to_postag_list(chunk_list: List[Union[Tree, Tuple[str, str]]]) -> List[Tuple[str, str]]:

def flatten_chunks(chunk_list: List[Union[Tree, Tuple[str, str]]]) -> List[Tuple[str, str]]:
    result = []  # type: List[Tuple[str, str]]
    for chunk in chunk_list:
        result.extend(chunk_to_postag_list(chunk))
    return result


def update_with_address(chunks_result) -> None:
    """If found address around state_sip_idx, update the result.

    returns if address is found, the new address_idx, otherwise
    state_zip_idx"""

    state_zip_idx = len(chunks_result) - 1
    # loop from state_zip_idx -1, up-to-4-lines (4 + 3 commas = 7)
    address_chunk_list = [chunks_result[state_zip_idx]]
    for idx in range(state_zip_idx -1, max(-1, state_zip_idx - 8), -1):
        chunk = chunks_result[idx]
        if chunk_is_postag(chunk):
            if is_postag_tag(chunk, ','):
                address_chunk_list.append(chunk)
            elif is_postag_tag(chunk, 'CD'):
                address_chunk_list.append(chunk)
            elif is_postag_tag(chunk, 'IN'):
                break
            else:
                break
        elif chunk.label() == 'xNNP':
            address_chunk_list.append(chunk)
        else:
            break
    address_chunk_list = list(reversed(address_chunk_list))
    # print("address_chunk_list = {}".format(address_chunk_list))

    address_chunk = Tree('xADDRESS', flatten_chunks(address_chunk_list))
    # print("address_chunk = {}".format(address_chunk))

    # pylint: disable=unused-variable
    for i in range(len(address_chunk_list)):
        del chunks_result[-1]

    chunks_result.append(address_chunk)
    # for i, chunk in enumerate(address_chunk_list):
    # print("address chunk #{}\t{}".format(i, chunk))
    # for i in range(len(address_chunk_list)):

# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def update_with_org_person(chunks_result,
                           chunk_list,
                           chunk_idx: int,
                           chunk_taken_idx: int) \
    -> int:
    """If found org_person, update the result.

    returns new chunk_idx if digested more chunks than before.
    """

    cur_chunk = chunk_list[chunk_idx]
    # remember is first word in the cur_chunk is all-caps
    org_words = chunk_to_words(cur_chunk)
    is_cur_chunk_all_caps = not is_org_suffix(org_words[0]) and \
                            org_words[0].isupper()
    is_cur_chunk_org_suffix = is_chunk_org_suffix_only(cur_chunk)

    # previous token cannot be "of".  This is NOT a party with "of"
    if chunk_idx - 1 > chunk_taken_idx:
        chunk = chunk_list[chunk_idx - 1]
        if is_chunk_of(chunk):
            # remove it from considered being an ORG
            # cur_chunk.set_label('xNNP')
            # for now, simply take the org, not throw it away
            chunks_result.append(cur_chunk)
            return chunk_idx

    prefix_chunk_list = []
    suffix_chunk_list = []
    # find the end of the org
    i = chunk_idx + 1
    num_comma = 0
    while i < len(chunk_list):
        chunk = chunk_list[i]
        if is_chunk_label(chunk, 'xPAREN') or has_pos_article(chunk):
            break
        elif is_chunk_of(chunk):
            next_chunk = get_next_chunk(chunk_list, i)
            if is_chunk_org(next_chunk):
                # try Bank of America, or Banc of America Securities LLC
                suffix_chunk_list.append(chunk)
                suffix_chunk_list.append(next_chunk)
                i += 2
                break
            else:
                break
        elif is_postag_comma(chunk):
            if num_comma >= 1:  # can only have 1 comma in proper name
                break
            suffix_chunk_list.append(chunk)
            i += 1
            num_comma += 1
        elif is_chunk_label(chunk, 'xPV_ORG'):
            # running into 'Bank of'
            break
        elif num_comma == 0 and \
             is_chunk_label(chunk, 'NNP') and \
             is_cur_chunk_all_caps and \
             is_chunk_all_caps(chunk):
            # running into 'Bank of'
            suffix_chunk_list.append(chunk)
            i += 1
            # 'WELLS FARGO BANK NORTHWEST, NATIONAL ASSOCIATION,', 'National Association' is an org_suffix
            # adding extra words to an org based on capitalization is == has_comma
            # num_comma += 1
        elif num_comma == 0 and \
             is_chunk_label(chunk, 'NNP') and \
             chunk_has_words(chunk, set(['fund'])):
            # (xORGP Tontine/NNP Capital/NNP)
            # (NNP Overseas/NNP Master/NNP Fund/NNP II/NNP)
            suffix_chunk_list.append(chunk)
            i += 1
        elif is_chunk_label(chunk, 'xORGP') and \
             is_chunk_org_suffix_only(chunk):
            # elif chunk_has_org_person_suffix(chunk):
            # postag_list = chunk_to_postag_list(chunk)
            # don't want to merge two full comany names together
            # only merge LLC or Corp.  Just in case an org is made
            # of 2 words, 'b. v.'

            suffix_chunk_list.append(chunk)
            i += 1
        else:
            break

    # if the last chunk is comma, give it back
    if suffix_chunk_list and is_postag_comma(suffix_chunk_list[-1]):
        num_comma -= 1
        del suffix_chunk_list[-1]

    # find the beginning or the org
    i = chunk_idx - 1
    iplus_chunk = None  # the last processed chunk, for merging paren purpose
    while i >= chunk_taken_idx:
        chunk = chunk_list[i]
        # if first word before org suffix
        if not is_cur_chunk_all_caps and \
           is_chunk_all_caps(chunk):
            is_cur_chunk_all_caps = True
        if not is_postag_comma(chunk):
            is_this_chunk_all_caps = is_chunk_all_caps(chunk)
        prev_chunk = get_prev_chunk(chunk_list, i)

        if is_chunk_label(chunk, 'xPAREN'):
            if is_cur_chunk_org_suffix and \
               i == chunk_idx - 1:  # right before org, 'Kodak (UK) Ltd'
                prefix_chunk_list.append(chunk)
                i -= 1
            elif is_cur_chunk_org_suffix and \
                     i == chunk_idx - 2 and \
                 is_postag_comma(iplus_chunk):
                prefix_chunk_list.append(chunk)
                i -= 1
            elif i == chunk_idx - 3 and \
                 is_chunk_all_caps(get_next_chunk(chunk_list, i)):
                # xxx (yyy) MEMBER, LLC, exprot-train/38282.txt
                prefix_chunk_list.append(chunk)
                i -= 1
            else:
                break
        elif is_chunk_label(chunk, 'xPV_ORG'):
            # found the beginning, 'Bank of'
            prefix_chunk_list.append(chunk)
            i -= 1
            break
        elif has_pos_article(chunk):
            if not chunk_has_jj(chunk):
                # don't want to have 'the undersigned xxx_org'
                prefix_chunk_list.append(chunk)
                i -= 1
            break
        elif is_postag_comma(chunk):
            if num_comma >= 1:
                break
            prefix_chunk_list.append(chunk)
            i -= 1
            num_comma += 1
        elif is_chunk_xnnp(chunk) or \
             is_chunk_label(chunk, 'NNP') or \
             is_chunk_label(chunk, 'xNN'):
            if chunk_num_tokens(chunk) == 1 or \
               is_cur_chunk_org_suffix:
                # is_chunk_label(chunk, 'NNP'):
                # 'in Rotterdam, UNILIVER PLC, ...'
                if is_cur_chunk_all_caps and \
                   not is_this_chunk_all_caps:
                    break
                prefix_chunk_list.append(chunk)
                i -= 1
                # can only add those multiple-word xNNP once based on xORGP
                is_cur_chunk_org_suffix = False
            else:
                break
        elif is_chunk_digit(chunk) and \
             prev_chunk and is_chunk_xnnp(prev_chunk):
            # ko
            prefix_chunk_list.append(chunk)
            prefix_chunk_list.append(prev_chunk)
            i -= 2
        elif prev_chunk and \
             ((is_chunk_and(chunk) and
               is_valid_org_and_prefix(chunk_to_words(prev_chunk))) or
              is_chunk_ampersand(chunk)):
            # 'and' is only applicable to a subset of nouns
            # otherwise, we have address Hong Kong and XXX Corp.
            # Must know Hong Kong is a part of an address.
            # Probably easier to limit to "Johnson &"

            # (xNNP Hadasit/NNP Medical/NNP Research/NNP Services/NNPS)
            # ('and', 'CC')
            # (ORG_PERSON Development/NNP Ltd/NNP)

            # (xNNP Box.com/NNP)
            # (xPAREN (/( UK/NNP )/))
            # (ORG_PERSON Ltd/NNP)
            prefix_chunk_list.append(chunk)
            prefix_chunk_list.append(prev_chunk)
            i -= 2
        else:
            break
        # this remember the last processed chunk
        # similar to 'prev_chunk', but that means
        # the chunk before chunk...  very different.
        iplus_chunk = chunk

    # if the first chunk is comma, give it back
    # the list is not reversed yet, so similar to suffix_chunk_list
    if prefix_chunk_list and is_postag_comma(prefix_chunk_list[-1]):
        num_comma -= 1
        del prefix_chunk_list[-1]

    org_chunk_list = list(reversed(prefix_chunk_list))
    org_chunk_list.append(chunk_list[chunk_idx])
    org_chunk_list.extend(suffix_chunk_list)
    flatten_chunk_list = flatten_chunks(org_chunk_list)
    # 'the Company', 'a Company'
    if len(flatten_chunk_list) == 2 and \
       flatten_chunk_list[0][1] == 'DT':
        chunks_result.append(cur_chunk)
        return chunk_idx

    org_chunk = Tree('xORGP', flatten_chunk_list)
    # print("org_chunk = {}".format(org_chunk))

    # current chunk is not added yet
    for i in range(len(prefix_chunk_list)):
        del chunks_result[-1]

    chunks_result.append(org_chunk)
    # for i, chunk in enumerate(org_chunk_list):
    # print("address chunk #{}\t{}".format(i, chunk))
    # for i in range(len(org_chunk_list)):

    return chunk_idx + len(suffix_chunk_list)


def update_with_prefix_org(chunks_result,
                           chunk_list,
                           chunk_idx: int) \
    -> int:
    """If found prefix_org, update the result.

    returns new chunk_idx if digested more chunks than before.
    """
    # cur_chunk = chunk_list[chunk_idx]

    suffix_chunk_list = []
    # find the end of the org
    i = chunk_idx + 1
    num_comma = 0
    while i < len(chunk_list):
        chunk = chunk_list[i]
        if is_chunk_label(chunk, 'xPAREN'):
            break
        elif is_postag_comma(chunk):
            if num_comma >= 1:  # can only have 1 comma in proper name
                break
            suffix_chunk_list.append(chunk)
            i += 1
            num_comma += 1
        elif is_chunk_label(chunk, 'xORGP') or \
             is_chunk_label(chunk, 'xPV_ORG') or \
             is_chunk_label(chunk, 'NNP'):
            # the board of trustees of the university of illinois, which is
            # made of 'the board of trustees of', and "the university of illinois'
            suffix_chunk_list.append(chunk)
            i += 1
            break
        elif has_pos_article(chunk):
            break
        else:
            break

    # if the last chunk is comma, give it back
    if suffix_chunk_list and is_postag_comma(suffix_chunk_list[-1]):
        num_comma -= 1
        del suffix_chunk_list[-1]

    org_chunk_list = [chunk_list[chunk_idx]]
    org_chunk_list.extend(suffix_chunk_list)
    flatten_chunk_list = flatten_chunks(org_chunk_list)

    org_chunk = Tree('xORGP', flatten_chunk_list)
    # print("org_chunk = {}".format(org_chunk))

    chunks_result.append(org_chunk)
    # for i, chunk in enumerate(org_chunk_list):
    # print("address chunk #{}\t{}".format(i, chunk))
    # for i in range(len(org_chunk_list)):

    return chunk_idx + len(suffix_chunk_list)


def mark_org_appositions(chunk_list: List[Union[Tree, Tuple[str, str]]]) \
    -> None:
    i = 0
    while i < len(chunk_list):
        chunk = chunk_list[i]
        if is_chunk_org(chunk):
            next_chunk = get_next_chunk(chunk_list, i)
            next2_chunk = get_next_chunk(chunk_list, i+1)
            # print("next_chunk {}".format(next_chunk))
            if next_chunk and \
               is_chunk_org(next_chunk) and \
               has_pos_article(next_chunk):
                next_chunk.set_label('xAPPOSITION_' + next_chunk.label())
                i += 1
            elif next_chunk and \
                 is_postag_comma(next_chunk) and \
                 next2_chunk and \
                 is_chunk_org(next2_chunk) and \
                 has_pos_article(next2_chunk):
                next2_chunk.set_label('xAPPOSITION_' + next2_chunk.label())
                i += 2
        i += 1

def mark_org_prev_xnnp_as_xorgp(chunk_list: List[Union[Tree, Tuple[str, str]]]) \
    -> None:
    i = 0
    prev_chunk = None
    prev2_chunk = None
    while i < len(chunk_list):
        chunk = chunk_list[i]
        # pylint: disable=too-many-boolean-expressions
        if (is_chunk_org(chunk) or is_chunk_xpv_org(chunk)) and \
           (is_chunk_comma(prev_chunk) or is_chunk_and(prev_chunk)) and \
           is_chunk_xnnp(prev2_chunk) and \
           chunk_num_tokens(prev2_chunk) > 1:
            if is_chunk_all_caps(chunk) and \
               not is_chunk_all_caps(prev2_chunk):
                # skip 'Hong Kong and ABBY CORP'
                pass
            elif prev2_chunk:
                prev2_chunk.set_label('xORGP')
        # 'XXX Southern, a Delaware corporation,
        # cannot use is_chunk_xnnp(chunk) because it checks for article
        elif is_chunk_label(chunk, 'xNNP') and \
             is_chunk_a_corporation(chunk) and \
             is_chunk_comma(prev_chunk) and \
             is_chunk_xnnp(prev2_chunk) and \
             isinstance(prev2_chunk, Tree):
            prev2_chunk.set_label('xORGP')

        # Dennis J. XXX ("Executive")
        elif is_chunk_label(chunk, 'xPAREN') and \
             is_chunk_likely_person_name(prev_chunk) and \
             isinstance(prev_chunk, Tree):
            prev_chunk.set_label('xORGP')

        prev2_chunk = prev_chunk
        prev_chunk = chunk
        i += 1


def mark_org_next_xnnp_as_xorgp(chunk_list: List[Union[Tree, Tuple[str, str]]]) \
    -> None:
    if not chunk_list:
        return
    i = 0
    chunk = chunk_list[i]
    next_chunk = get_next_chunk(chunk_list, i)
    next2_chunk = get_next_chunk(chunk_list, i+1)
    while i < len(chunk_list) - 2:
        # pylint: disable=too-many-boolean-expressions
        if (is_chunk_org(chunk) or is_chunk_xpv_org(chunk)) and \
           (is_chunk_comma(next_chunk) or is_chunk_and(next_chunk)) and \
           is_chunk_xnnp(next2_chunk) and \
           chunk_num_tokens(next2_chunk) > 1:
            if is_chunk_all_caps(chunk) and \
               not is_chunk_all_caps(next2_chunk):
                # skip 'Hong Kong and ABBY CORP'
                pass
            elif next2_chunk:
                next2_chunk.set_label('xORGP')
        i += 1
        chunk = next_chunk
        next_chunk = next2_chunk
        next2_chunk = get_next_chunk(chunk_list, i+1)


def mark_an_org_not_org(chunk_list: List[Union[Tree, Tuple[str, str]]]) \
    -> None:
    for chunk in chunk_list:
        if is_chunk_org(chunk):
            postag_list = chunk_to_postag_list(chunk)
            word, tag = postag_list[0]
            last_word, unused_last_tag = postag_list[-1]
            # 'the Pass Through Trust'; this is a valid company
            # 'the Gap' is also valid
            # 'the Board of Directors
            if tag == 'DT' and \
               (word in set(['a', 'an']) or
                last_word[0].islower()):
                # to make mypy happy for type checking
                tmp_chunk = chunk  # type: Tree
                tmp_chunk.set_label('xNN')
                continue

            # "Delaware corporation"
            if len(postag_list) < 3 and \
                postag_list[-1][0] == 'corporation':
                # to make mypy happy for type checking
                tmp_chunk = chunk
                tmp_chunk.set_label('xNN')
                continue

            # company number
            is_all_lower = True
            for postag in postag_list:
                word, tag = postag
                if word[0].isupper():
                    is_all_lower = False
                    break
            if is_all_lower:
                # to make mypy happy for type checking
                tmp_chunk = chunk
                tmp_chunk.set_label('xNN')


def get_org_aware_chunks(sent_line: str):
    chunk_list = chunkize_by_regex_grammar(sent_line)

    # for i, chunk in enumerate(chunk_list):
    #     print('raw chunk #{}\t{}'.format(i, chunk))

    mark_an_org_not_org(chunk_list)

    # print("chunk_list333: {}".format(chunk_list))
    result = []  # List[Tree]
    chunk_list_len = len(chunk_list)
    chunk_idx = 0
    # cannot go back before this index when doing
    # semantic chunking.  Several party names might
    # next to each other.  Cannot group them all together.
    chunk_taken_idx = 0
    # prev_chunk = None
    # prev2_chunk = None
    while chunk_idx < chunk_list_len:
        chunk = chunk_list[chunk_idx]
        if isinstance(chunk, Tree) and \
           chunk.label() in set(['xNNP', 'xORGP', 'xPV_ORG']):
            next_chunk = get_next_chunk(chunk_list, chunk_idx)
            # try to identify address
            if next_chunk and \
               chunk_is_postag(next_chunk) and \
               is_postag_tag(next_chunk, 'CD') and \
               is_zipcode(postag_word(next_chunk)):
                pos_list = chunk_to_postag_list(chunk)
                # pos_list = list(chunk.subtrees(lambda t: t.height() == 2))
                pos_list.append(next_chunk)
                # print("pos_list: {}".format(pos_list))
                result.append(Tree('xSTATE_ZIP', pos_list))
                update_with_address(result)
                chunk_idx += 1
                chunk_taken_idx = chunk_idx
            # elif chunk_has_org_person_suffix(chunk):
            elif is_chunk_label(chunk, 'xORGP'):
                chunk_idx = update_with_org_person(result,
                                                   chunk_list,
                                                   chunk_idx,
                                                   chunk_taken_idx)
                chunk_taken_idx = chunk_idx
            elif is_chunk_label(chunk, 'xPV_ORG'):
                chunk_idx = update_with_prefix_org(result,
                                                   chunk_list,
                                                   chunk_idx)
                chunk_taken_idx = chunk_idx
                # pylint: disable=pointless-string-statement
                """
            elif is_chunk_label(chunk, 'xNNP') and \
                 is_chunk_comma(prev_chunk) and \
                 is_chunk_label(prev2_chunk, 'xORGP'):
                # this takes away the ability to handle
                # J.P. Morgan  Securities Inc., Merrill Lynch, Pierce, Fenner & Smith Incorporated,
                # Morgan Stanley & Co.  Incorporated
                chunk.set_label('xORGP')
                result.append(chunk)
                """
            else:
                result.append(chunk)
        else:
            # print('pos_tag #{}\t{}'.format(i, tree))
            # pass
            result.append(chunk)
        # prev2_chunk = prev_chunk
        # prev_chunk = chunk
        chunk_idx += 1

    # mark_an_org_not_org(chunk_list)
    mark_org_appositions(result)
    # for org that has no suffix, try to add it in if it is somewhat obviosu (all-caps)
    # before or after an xorg
    mark_org_prev_xnnp_as_xorgp(result)
    mark_org_next_xnnp_as_xorgp(result)
    return result


# not used by anyone
def extract_proper_names(sent_line: str):
    chunk_list = get_org_aware_chunks(sent_line)

    se_tok_list = span_tokenize(sent_line)
    result = []
    tok_idx = 0
    for chunk in chunk_list:
        postag_list = chunk_to_postag_list(chunk)
        #print("tok_idx = {}, end = {}".format(tok_idx,
        #                                      tok_idx + len(postag_list)))
        start_se = se_tok_list[tok_idx]
        end_se = se_tok_list[tok_idx + len(postag_list) - 1]
        result.append((start_se[0], end_se[1], chunk))

        print("chunk: [{}]\t({},{}\t{})".format(sent_line[start_se[0]:
                                                          end_se[1]],
                                                start_se[0],
                                                end_se[1],
                                                chunk.label() if isinstance(chunk, Tree) else chunk))

        tok_idx += len(postag_list)


class SpanChunk:

    # pylint: disable=too-many-arguments
    def __init__(self,
                 start: int,
                 end: int,
                 tok_idx: int,
                 nempty_tok_idx: int,
                 chunk: Union[Tree, Tuple[str, str]],
                 text: str,
                 se_tok_list: List[Tuple[int, int]]) -> None:
        self.start = start
        self.end = end
        self.tok_idx = tok_idx
        self.nempty_tok_idx = nempty_tok_idx
        self.chunk = chunk
        self.text = text
        self.se_tok_list = se_tok_list

    def label(self) -> str:
        if not is_chunk_tree(self.chunk):
            raise ValueError
        # to make mypy happy for type checking
        tmp_chunk = self.chunk  # type: Tree
        return tmp_chunk.label()

    def has_label(self, label: str) -> bool:
        if is_chunk_tree(self.chunk):
            # to make mypy happy for type checking
            tmp_chunk = self.chunk  # type: Tree
            return tmp_chunk.label() == label
        return False

    def is_phrase(self) -> bool:
        return is_chunk_tree(self.chunk)

    def get_lc_word(self) -> str:
        if is_chunk_tree(self.chunk):
            raise ValueError
        return postag_word(self.chunk).lower()

    def is_lc_word(self, word: str) -> bool:
        if is_chunk_tree(self.chunk):
            return False
        return postag_word(self.chunk).lower() == word

    def is_comma(self) -> bool:
        # no need to lc(), but to keep code small
        return self.is_lc_word(',')

    def get_words(self) -> List[str]:
        postag_list = self.to_postag_list()
        return [postag[0] for postag in postag_list]

    def startswith(self, line: str) -> bool:
        return self.text.lower().startswith(line)

    def is_word_and(self) -> bool:
        return self.is_lc_word('and') or self.is_lc_word('&')

    def is_word_between(self) -> bool:
        return not self.is_phrase() and \
            self.get_lc_word() in set(['and', 'among'])

    def is_org(self) -> bool:
        return self.has_label('xORGP')

    def is_org_suffix(self) -> bool:
        if not self.has_label('xORGP'):
            return False
        return is_org_suffix(self.text)

    def is_paren(self) -> bool:
        return self.has_label('xPAREN')

    def to_postag_list(self) -> List[Tuple[str, str]]:
        return chunk_to_postag_list(self.chunk)

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if is_chunk_tree(self.chunk):
            return "{} ({:3d}, {:3d})\tidx={}, nidx={}\t{}".format('   Phrase',
                                                                   self.start, self.end,
                                                                   self.tok_idx, self.nempty_tok_idx,
                                                                   phrase_to_str(self.chunk))
        # for non-phrase
        return "{} ({:3d}, {:3d})\tidx={}, nidx={}\t{}".format('NotPhrase',
                                                               self.start, self.end,
                                                               self.tok_idx, self.nempty_tok_idx,
                                                               self.chunk)


class PhrasedSent:

    # pylint: disable=too-many-instance-attributes
    def __init__(self, sent_line: str, is_chopped: bool) -> None:
        if IS_DEBUG_CHUNK:
            print('PhrasedSent("{}")'.format(sent_line))
        self.is_chopped = is_chopped
        self.sent_line = sent_line
        self.span_chunk_list = tokenize_to_span_chunks(sent_line)
        self.good_cand_list = []  # type: List[SpanChunk]
        self.maybe_cand_list = []  # type: List[SpanChunk]
        self.bad_cand_list = []  # type: List[SpanChunk]
        self.unknown_cand_list = []  # type: List[SpanChunk]
        self.not_tree_list = []  # type: List[SpanChunk]

        self.parse_nouns()  # populate the above lists

    def parse_nouns(self) -> None:
        for span_chunk in self.span_chunk_list:
            if span_chunk.is_phrase():
                # print("achunk #{}\t{}\t{}".format(i, chunk.label(), chunk))
                if span_chunk.label() == 'NP':
                    self.bad_cand_list.append(span_chunk)
                elif span_chunk.label() in set(['xORGP',
                                                'xAPPOSITION_ORG_PERSON',
                                                'xPAREN',
                                                'xADDRESS']):
                    self.good_cand_list.append(span_chunk)
                elif span_chunk.label() in set(['xNNP']):
                    self.maybe_cand_list.append(span_chunk)
                # elif chunk.label() in set('NP'):
                #     bad_cand_list.append(span_chunk)
                else:
                    self.unknown_cand_list.append(span_chunk)
            else:
                self.not_tree_list.append(span_chunk)

    def print_parsed(self):
        for alist in [('good cand', self.good_cand_list),
                      ('maybe cand', self.maybe_cand_list),
                      ('unknown cand', self.unknown_cand_list),
                      ('bad cand', self.bad_cand_list),
                      ('not_tree cand', self.not_tree_list)]:
            cat, alist = alist
            print("{}:".format(cat))
            for i, span_chunk in enumerate(alist):
                if span_chunk.is_phrase():
                    print("    #{}\t{}".format(i, span_chunk))
                else:
                    print("    #{}\t{}".format(i, span_chunk))

    def extract_orgs_term_offset_list(self) \
        -> List[Tuple[List[Tuple[int, int]],
                      Optional[Tuple[int, int]]]]:
        orgs_term_list = self.extract_orgs_term_list()

        result = []  # type: List[Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]]
        for orgs_term in orgs_term_list:
            orgs, term = orgs_term

            term_offset = None
            if term:
                term_offset = (term[0].start, term[-1].end)

            # terms sometimes can overlap with org
            # we prefer terms over org
            # 'WACHOVIA BANK, NATIONAL ASSOCIATION, a national banking association, in its capacity as Issuing Bank'
            # both 'Issuing Bank' (party) and 'as Issuing Bank' (defined term) are possible

            orgs_offset = []  # type: List[Tuple[int, int]]
            for org in orgs:
                if not term_offset:
                    orgs_offset.append((org.start, org.end))
                elif term_offset and \
                     not mathutils.start_end_subsume(term_offset,
                                                     (org.start, org.end)):
                    orgs_offset.append((org.start, org.end))

            result.append((orgs_offset, term_offset))
        return result

    def extract_orgs_term_list(self) \
        -> List[Tuple[List[SpanChunk], List[SpanChunk]]]:
        """Find the list of org_person_term groups.

        There can be multiple org per term.
        """
        DEBUG = False
        if re.match(r'(is\s+delivered\s+by\s+)', self.sent_line, re.I):
            # equity agreement is delivered by XXX Corporation (the “Company”),
            # to Alice Bob (the “Grantee”).
            delivered_by_tok_idx = find_delivered_by_tok_index(self.span_chunk_list)
            to_org_tok_idx = find_to_tok_index(self.span_chunk_list)

            sep_idx_list = [delivered_by_tok_idx]

            if DEBUG:
                print("delivered_by_tok_idx:", delivered_by_tok_idx)
                print("to_org_tok_idx:", to_org_tok_idx)
            if to_org_tok_idx != -1:
                sep_idx_list.append(to_org_tok_idx)
            sep_idx_list.append(len(self.span_chunk_list))
            sep_idx_list.sort()
        elif re.search(r'(.+) for\s+value\s+received,?\s+(.*)$', self.sent_line, re.I):
            # xxx corp, for value received, agreed to pay to the order of yyy trust
            sep_idx_list = [0]
            for_value_tok_idx = find_for_value_tok_index(self.span_chunk_list)
            order_of_tok_idx = find_order_of_tok_index(self.span_chunk_list)

            sep_idx_list.append(for_value_tok_idx)
            if order_of_tok_idx != -1:
                sep_idx_list.append(order_of_tok_idx)
            sep_idx_list.append(len(self.span_chunk_list))
            sep_idx_list.sort()
        else:
            # normal agreement using between
            and_org_tok_idx_list = find_and_org_tok_indices(self.span_chunk_list)
            # add other breaking possibilities
            and_org_tok_idx_list.extend(find_payto_org_tok_indices(self.span_chunk_list))

            between_tok_idx = 0
            if not self.is_chopped:
                between_tok_idx = find_between_tok_index(self.span_chunk_list)
                if between_tok_idx == -1:
                    return []
            if DEBUG:
                print("betwen_tok_idx: ", between_tok_idx)
                print("and_org_tok_idx_list: {}".format(and_org_tok_idx_list))

            paren_org_tok_idx_list = find_paren_org_tok_indices(self.span_chunk_list)

            sep_idx_list = [between_tok_idx]
            sep_idx_list.extend(and_org_tok_idx_list)
            sep_idx_list.extend(paren_org_tok_idx_list)
            sep_idx_list.append(len(self.span_chunk_list))
            sep_idx_list.sort()

        if DEBUG:
            print("sep_idx_list: {}".format(sep_idx_list))

        org_term_spchunk_list_list = []  # List[List[SpunkChunk]]
        tok_start = sep_idx_list[0]
        for tok_end in sep_idx_list[1:]:
            org_term_spchunk_list_list.append(self.span_chunk_list[tok_start:tok_end])
            tok_start = tok_end

        result = []  # type: List[Tuple[List[SpanChunk], List[SpanChunk]]]
        for i, orgs_term_spchunk_list in enumerate(org_term_spchunk_list_list):
            if IS_DEBUG_ORGS_TERM:
                print("\norg_term_spchunk_list #{}".format(i))
                for j, spchunk in enumerate(orgs_term_spchunk_list):
                    print("    chunk #{}\t{}".format(j, spchunk))


            orgs_term = extract_orgs_term_in_span_chunk_list(orgs_term_spchunk_list)
            if orgs_term:
                result.append(orgs_term)
        return result

    def extract_orgs_term(self) \
        -> Optional[Tuple[List[SpanChunk], List[SpanChunk]]]:
        """Find the org_person_term groups.

        There can be multiple org per term.
        """
        return extract_orgs_term_in_span_chunk_list(self.span_chunk_list)

    def extract_orgs_term_offset(self) \
        -> Optional[Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]]:
        """Find the org_person_term groups.

        There can be multiple org per term.
        """
        orgs_term = extract_orgs_term_in_span_chunk_list(self.span_chunk_list)

        if not orgs_term:
            return None

        orgs, term = orgs_term
        orgs_offset = []  # type: List[Tuple[int, int]]
        for org in orgs:
            orgs_offset.append((org.start, org.end))
        term_offset = None
        if term:
            # term is a list of SpanList
            term_offset = (term[0].start, term[-1].end)

        return orgs_offset, term_offset


def span_chunk_list_to_words(span_chunk_list: List[SpanChunk]) \
    -> List[str]:
    all_words = []
    for span_chunk in span_chunk_list:
        span_words = span_chunk.get_words()
        all_words.extend(span_words)
    return all_words


def find_as_span_chunks(span_chunk_list: List[SpanChunk]) \
        -> List[List[SpanChunk]]:
    as_idx_list = []  # type: List[int]
    for i, span_chunk in enumerate(span_chunk_list):
        if span_chunk.is_lc_word('as'):
            as_idx_list.append(i)

    if as_idx_list:
        result = []  # type: List[List[SpanChunk]]
        prev_as_idx = as_idx_list[0]
        for as_idx in as_idx_list[1:]:
            # skip the word 'as'
            result.append(span_chunk_list[prev_as_idx+1:as_idx])
            prev_as_idx = as_idx
        result.append(span_chunk_list[prev_as_idx+1:])
        # return only non-empty list in this list
        return [x for x in result if x != []]
    return []


def find_as_in_paren(span_chunk_list: List[SpanChunk]) \
        -> List[SpanChunk]:
    # it is assume that a paren is only 1 SpanChunk
    if len(span_chunk_list) != 1:
        return []

    as_idx_list = []  # type: List[int]
    postag_list = span_chunk_list[0].to_postag_list()
    for i, postag in enumerate(postag_list):
        word, unused_tag = postag
        if word.lower() == 'as':
            as_idx_list.append(i+1)  # take the token after 'as'

    if as_idx_list:
        # no need to check if as_idx_list exceed end of list, because must have right paren

        cur_tok_idx = as_idx_list[0]
        if len(as_idx_list) == 1:
            last_tok_idx = len(postag_list)
        else:  # for any larger than 1
            # since this is not the end, no removal
            last_tok_idx = as_idx_list[1]
        postags_out = postag_list[cur_tok_idx:last_tok_idx]
        cur_span_chunk = span_chunk_list[0]
        se_tok_list_out = cur_span_chunk.se_tok_list[cur_tok_idx:last_tok_idx]
        # if there is nothing after the word "as"
        if not se_tok_list_out:
            return []
        sc_start = cur_span_chunk.se_tok_list[0][0]
        nstart, nend = se_tok_list_out[0][0], se_tok_list_out[-1][1]
        shorten_text = cur_span_chunk.text[nstart - sc_start:nend - sc_start]
        shorten_span_chunk = SpanChunk(nstart,
                                       nend,
                                       cur_span_chunk.tok_idx + cur_tok_idx,
                                       cur_span_chunk.nempty_tok_idx + cur_tok_idx,
                                       Tree('xNNP', postags_out),
                                       shorten_text,
                                       se_tok_list_out)
        # 'in such capacity, “Agent” as hereinafter further defined'
        # 37169.txt
        if shorten_text == 'hereinafter further defined':
            return []
        return [shorten_span_chunk]

    return []

def make_span_chunk_from_span_chunk_list(span_chunk_list: List[SpanChunk]) -> SpanChunk:
    postags_out = []  # type: List[Tuple[str, str]]
    se_tok_list_out = []  # type: List[Tuple[int, int]]
    st_list = []  # type: List[str]
    offset = 0  # space between span_chunk
    for span_chunk in span_chunk_list:
        postags_out.extend(span_chunk.to_postag_list())
        se_tok_list_out.extend(span_chunk.se_tok_list)
        st_list.append(' ' * (span_chunk.se_tok_list[0][0] - offset))
        st_list.append(span_chunk.text)
        offset = span_chunk.se_tok_list[-1][1]

    first_span_chunk = span_chunk_list[0]

    merged_text = ''.join(st_list)
    print("merged_text = [{}]".format(merged_text))

    nstart, nend = se_tok_list_out[0][0], se_tok_list_out[-1][1]
    merged_span_chunk = SpanChunk(nstart,
                                  nend,
                                  first_span_chunk.tok_idx,
                                  first_span_chunk.nempty_tok_idx,
                                  Tree('xNNP', postags_out),
                                  merged_text,
                                  se_tok_list_out)
    return merged_span_chunk


def chop_spanchunk_paren(span_chunk: SpanChunk) -> SpanChunk:

    cur_tok_idx = 1
    last_tok_idx = -1
    postag_list = span_chunk.to_postag_list()
    postags_out = postag_list[cur_tok_idx:last_tok_idx]
    se_tok_list_out = span_chunk.se_tok_list[cur_tok_idx:last_tok_idx]
    nstart = se_tok_list_out[0][0]
    nend = se_tok_list_out[-1][1]
    sc_start = span_chunk.se_tok_list[0][0]
    shorten_text = span_chunk.text[nstart - sc_start:nend - sc_start]

    shorten_span_chunk = SpanChunk(nstart,
                                   nend,
                                   span_chunk.tok_idx + cur_tok_idx,
                                   span_chunk.nempty_tok_idx + cur_tok_idx,
                                   Tree('xPAREN', postags_out),
                                   shorten_text,
                                   se_tok_list_out)
    return shorten_span_chunk


def remove_invalid_defined_terms_parens(span_chunk_list: List[SpanChunk]) \
    -> List[SpanChunk]:
    result = []
    for span_chunk in span_chunk_list:
        # this is another potential location to throw away empty parenthesis
        # if span_chunk.text.startswith('(') and \
        #    span_chunk.text.endswith(')') and \
        #    not span_chunk.text[1:-1].strip():
        #     pass
        if len(span_chunk.text) < 30 and \
           re.search(r'.*\btogether.*the.*parties', span_chunk.text, re.I):
            # '(together, the "Parties")', not precise enough
            pass
        elif re.search(r'\b(agreement|note)s?\b', span_chunk.text, re.I):
            pass
        elif re.match(r'\b(and)\b', span_chunk.text, re.I):
            pass
        elif re.search(r'\b(subject\s+to|any\s+such|resulting|closing|article)\b', span_chunk.text, re.I):
            # export-train/35633.txt
            pass
        elif re.search(r'\b(date|day|amend(ed)?)\b', span_chunk.text, re.I):
            # (as effective date)
            pass
        elif re.search(r'\b(number|loan|rate|amount|principal|warrant|act|registration)\b', span_chunk.text, re.I):
            # (registered number SC183333)
            pass
        elif re.search(r'\bparty\b.*and.*collectively.*parties.*', span_chunk.text, re.I):
            # pylint: disable=fixme
            # TODO, not sure why adding following caused failure in
            # export-train/52082.txt failed??
            # or \
            #  re.search(r'\beach.*party\b.*and.*together.*parties.*', span_chunk.text, re.I):
            # 'a “Party” and collectively the “Parties”'
            # 'each a "Party", and together, the "Parties"
            pass
        elif re.search(r'(\$\d|\d%)', span_chunk.text, re.I):
            pass
        elif re.match(r'\d+\)', span_chunk.text, re.I):  # just a number
            # caused by an itemized list in party_line
            # 35670.txt
            # TODO, this is NOT resolved yet
            pass
        elif re.match(r'\d+\b', span_chunk.text, re.I):  # just a number
            pass
        elif re.search(r'\b(section|article|act|defined|below)\b', span_chunk.text, re.I):
            words = span_chunk.get_words()
            # print("words2555: [{}]".format(' '.join(words)))
            if strutils.has_quote(' '.join(words)):
                result.append(span_chunk)
            else:
                pass
        elif re.match(r'(this)\b', span_chunk.text, re.I):
            # cannot be 'this xxx"
            pass
        else:
            result.append(span_chunk)
    return result

# a 'term' might have multiple span_chunk because 'as' defined term might have
# multiple spanchunk instead of parens
def remove_invalid_defined_terms_as(span_chunk_list: List[SpanChunk]) \
    -> List[SpanChunk]:
    for span_chunk in span_chunk_list:
        if re.search(r'\b(date|amend(ed)?|follows?)\b', span_chunk.text, re.I):
            return []
        elif re.search(r'\b(part)\b', span_chunk.text, re.I):
            return []
        elif re.match(r'(this)\b', span_chunk.text, re.I):
            # cannot be 'this xxx"
            return []
        elif re.match(r'(of)\b', span_chunk.text, re.I):
            # don't like 'as of xxx date'
            return []
    return span_chunk_list



def rerank_defined_term_parens(paren_list: List[SpanChunk],
                               org_list: List[SpanChunk]) \
                               -> List[SpanChunk]:
    org_lc_words = []
    for spchunk in org_list:
        spc_words = spchunk.get_words()
        org_lc_words.extend([spc_word.lower() for spc_word in spc_words])
    org_word_set = set(org_lc_words)

    score_list = [0.5] * len(paren_list)
    len_list = []  # type: List[int]
    # mainly for breaking a tie in above
    seq_list = []  # type: List[int]
    if IS_DEBUG_RERANK_DEFINED_TERM:
        print("\nrerank_defined_term_parens()")
    for idx, span_chunk in enumerate(paren_list):
        words = span_chunk.get_words()
        matched = 0
        num_alpha_word = 0
        for word in words:
            if strutils.is_alpha_word(word):
                lc_word = word.lower()
                if lc_word in DEFINED_TERM_WORD_SET or \
                   lc_word in org_word_set:
                    matched += 1
                num_alpha_word += 1
        if num_alpha_word != 0:
            score = matched / num_alpha_word
        else:
            score = 0.0
        score_list[idx] = score
        len_list.append(100 - num_alpha_word)
        seq_list.append(idx)  # prefer the last one
        if IS_DEBUG_RERANK_DEFINED_TERM:
            print("rerank spank_chunk: score={}, [{}]".format(span_chunk.get_words(), score))

    ordered_list = sorted(zip(score_list, len_list, seq_list, paren_list), reverse=True)
    return [paren for score, xlen, xseq, paren in ordered_list]


def remove_invalid_parties(span_chunk_list: List[SpanChunk]) \
    -> List[SpanChunk]:
    result = []
    for span_chunk in span_chunk_list:
        if re.search(r'\b(acting|through)\b', span_chunk.text, re.I):
            if IS_DEBUG_ORGS_TERM:
                print('removed invalid party, "acting through"')
            # pass
        elif re.search(r'\bbranch$', span_chunk.text, re.I):
            if IS_DEBUG_ORGS_TERM:
                print('removed invalid party, "branch"')
            # pass
        elif re.search(r'\bc/o\b', span_chunk.text, re.I) or \
             re.match(r'a ', span_chunk.text, re.I):
            # 'a Corporation'
            # c/o XXX Agency
            if IS_DEBUG_ORGS_TERM:
                print('removed invalid party, "a, c/o"')
            # pass
        elif re.search(r'\d+$', span_chunk.text, re.I):
            # 'Suite 123', 'CO 23534'
            if IS_DEBUG_ORGS_TERM:
                print('removed invalid party, ends in a number')
            # pass
        elif re.search(r'\b(agreement|extension|amendment)\b', span_chunk.text, re.I):
            if IS_DEBUG_ORGS_TERM:
                print('removed invalid party, invalid word')
            # pass
        elif re.match(r'_+$', span_chunk.text):
            if IS_DEBUG_ORGS_TERM:
                print('removed invalid party, only ___')
            # pass
        elif is_a_location(span_chunk.text):
            # such as 'Wales' or 'London'
            if IS_DEBUG_ORGS_TERM:
                print('removed invalid party, is_location({})'.format(span_chunk.text))
            # pass
        elif len(span_chunk.text) < 3:  # never happened before
            # too short
            if IS_DEBUG_ORGS_TERM:
                print('removed invalid party, len({}) < 3'.format(span_chunk.text))
            # pass
        elif re.match(r'(this|agreement|lease|now|therefore)\b', span_chunk.text, re.I):
            # Sometimes got have the first part of a sentence,
            # got the wrong heading instead.
            if IS_DEBUG_ORGS_TERM:
                print('removed invalid party, invalid prefix({})'.format(span_chunk.text))
            # pass
        else:
            result.append(span_chunk)
    return result

# because of 'term' can be a list of words in "as xxx xxx", so it is a list
def extract_orgs_term_in_span_chunk_list(span_chunk_list: List[SpanChunk]) \
    -> Optional[Tuple[List[SpanChunk], List[SpanChunk]]]:

    if not span_chunk_list:
        print("extract_orgs_term_in_span_chunk_list([]) is called!!!!")
        return None

    paren_list = [span_chunk for span_chunk in span_chunk_list if span_chunk.has_label('xPAREN')]
    org_list = [span_chunk for span_chunk in span_chunk_list if span_chunk.has_label('xORGP')]
    org_list = remove_invalid_parties(org_list)

    # if no org_list is found, take the first xNNP and make it a party
    # This handles Person name much better
    if not org_list:  # didn't find any org based on org_suffix
        # for i, span_chunk in enumerate(span_chunk_list):
            # print("5234 #{} span_chunk: {}".format(i, span_chunk))
        span_chunk = span_chunk_list[0]
        next1_span_chunk = next_span_chunk(span_chunk_list, 0)
        next2_span_chunk = next_span_chunk(span_chunk_list, 1)
        # pylint: disable=too-many-boolean-expressions
        if span_chunk.is_phrase() and \
           span_chunk.has_label('xNNP') and \
            next1_span_chunk and next1_span_chunk.is_lc_word('and') and\
            next2_span_chunk and next2_span_chunk.has_label('xNNP'):
            tmp_span_chunk_list = span_chunk_list[0:3]
            org_list.append(make_span_chunk_from_span_chunk_list(tmp_span_chunk_list))
        elif span_chunk.is_phrase() and \
           span_chunk.has_label('xNNP') and \
           not span_chunk.startswith('this'):
            org_list.append(span_chunk)

    # there can be still bad org in 'and org'
    org_list = remove_invalid_parties(org_list)

    as_list = find_as_span_chunks(span_chunk_list)

    term = []  # type: List[SpanChunk]
    if IS_DEBUG_ORGS_TERM:
        for pprn in paren_list:
            print("potential paren: {}".format(pprn))
    # Apply more specific rules to parens than just
    # generally to all terms, such as "as ..."
    # maybe in the future, only have 1
    paren_list = remove_invalid_defined_terms_parens(paren_list)
    if IS_DEBUG_ORGS_TERM:
        for pprn in paren_list:
            print("filtered paren: {}".format(pprn))
    if paren_list:
        if len(paren_list) == 1:
            if len(paren_list[0].se_tok_list) > 2:  # must have more than just '(' and ')'
                term = [chop_spanchunk_paren(paren_list[0])]
            else:
                term = []
        else:  # there are multiple


            # # if the 2nd paren has 'each a  "Party", and collectively the "Parties"'
            # last_paren = paren_list[-1]
            # if re.search(r'\b(parties)\b', last_paren.text, re.I):
            #     # if 'parties' is the last one, now take the first one instead
            #     term = [chop_spanchunk_paren(paren_list[0])]
            # else:
            #     term = [chop_spanchunk_paren(last_paren)]

            ordered_paren_list = rerank_defined_term_parens(paren_list, org_list)
            term = [chop_spanchunk_paren(ordered_paren_list[0])]

            # in future, might check if term/paren doesn't overlap with org
            # pylint: disable=line-too-long
            # if last_paren.nempty_tok_idx >= span_chunk_list[-1] - 3:  # parent is really at the end of phrase
            #     term = last_paren
            # else:
            #     term = paren_list[0]

        # if there is 'as' in this list
        as_in_paren = find_as_in_paren(term)
        if as_in_paren:
            # remove the last paren
            term = as_in_paren

    elif as_list:
        if len(as_list) == 1:
            term = as_list[0]  # List[SpanChunk]
        elif len(as_list) == 2:
            # as a “Party” and together as the “Parties”.
            term = as_list[0]  # List[SpanChunk]
            term.extend(as_list[1])
        else:  # there are multiple, take the first one
            term = as_list[0]
        # now filtering out any invalid term
        term = remove_invalid_defined_terms_as(term)

    if not org_list and not term:
        return None
    return org_list, term



def phrase_to_str(chunk: Tree) -> str:
    postag_st_list = []
    for postag in chunk:
        postag_st_list.append('{}/{}'.format(postag[0], postag[1]))
    return '({} {})'.format(chunk.label(), ' '.join(postag_st_list))

def next_span_chunk(span_chunk_list: List[SpanChunk], idx: int) -> Optional[SpanChunk]:
    if idx + 1 < len(span_chunk_list):
        return span_chunk_list[idx + 1]
    return None

def prev_span_chunk(span_chunk_list: List[SpanChunk], idx: int) -> Optional[SpanChunk]:
    if idx - 1 >= 0:
        return span_chunk_list[idx - 1]
    return None

def find_and_org_tok_indices(span_chunk_list: List[SpanChunk]) -> List[int]:
    """Find the list of tok_idx of 'and org'.
    """
    result = []  # type: List[int]
    # This is for 'and' followed by xNNP, so might or might not be and-org
    maybe_result = []  # type: List[int]
    idx = 0
    prev_spchunk = None
    while idx < len(span_chunk_list):
        spchunk = span_chunk_list[idx]
        next_spchunk = next_span_chunk(span_chunk_list, idx)
        if spchunk.is_word_and():
            # export-train/40170.txt
            if prev_spchunk and \
               prev_spchunk.has_label('xPAREN') and \
               next_spchunk:
                result.append(next_spchunk.tok_idx)
                idx += 1
            elif next_spchunk:
                if next_spchunk.is_org():
                    result.append(next_spchunk.tok_idx)
                    idx += 1
                elif next_spchunk.has_label('xNNP'):
                    # 'and Arrayit Diagnostics', doc111.txt
                    maybe_result.append(next_spchunk.tok_idx)
                    idx += 1
                elif next_spchunk.is_lc_word('each'):
                    # 'and each of the investors' 35753.txt
                    maybe_result.append(next_spchunk.tok_idx)
                    idx += 1

        prev_spchunk = spchunk
        idx += 1

    # replace result only if valid and-org is not found
    if not result and maybe_result:
        result = maybe_result
    return result


# pay to
# successor interest to, 36039.txt
def find_payto_org_tok_indices(span_chunk_list: List[SpanChunk]) -> List[int]:
    """Find the list of tok_idx of 'pay to'.
    """
    result = []  # type: List[int]
    # This is for 'and' followed by xNNP, so might or might not be and-org
    maybe_result = []  # type: List[int]
    idx = 0
    while idx < len(span_chunk_list):
        spchunk = span_chunk_list[idx]
        next_spchunk = next_span_chunk(span_chunk_list, idx)
        # next2_spchunk = next_span_chunk(span_chunk_list, idx+1)
        if spchunk.is_lc_word('pay') and \
            next_spchunk and \
            next_spchunk.is_lc_word('to'):
            # take the org after 'pay to'
            # 'to pay to the order of Integrated xxx, Inc.', 35642.txt
            for j in range(4):
                next_j_spchunk = next_span_chunk(span_chunk_list, idx+1+j)
                if not next_j_spchunk:
                    break
                if next_j_spchunk.is_org():
                    result.append(next_j_spchunk.tok_idx)
                    idx += (2 + j)
                    break
                elif next_j_spchunk.has_label('xNNP'):
                    # 'and Arrayit Diagnostics', doc111.txt
                    maybe_result.append(next_j_spchunk.tok_idx)
                    idx += (2 + j)
                    break
        elif spchunk.is_lc_word('to') and \
            next_spchunk:
            if next_spchunk.is_org():
                result.append(next_spchunk.tok_idx)
                idx += 1
                # don't want to be too agreesive on all "to"
                # pylint: disable=pointless-string-statement
                """
            elif next_spchunk.has_label('xNNP'):
                # 'and Arrayit Diagnostics', doc111.txt
                maybe_result.append(next_spchunk.tok_idx)
                idx += 1
                """

        idx += 1

    # replace result only if valid and-org is not found
    if not result and maybe_result:
        result = maybe_result
    return result


def find_paren_org_tok_indices(span_chunk_list: List[SpanChunk]) -> List[int]:
    """Find the list of tok_idx of '), org'.
    """
    result = []  # type: List[int]
    idx = 0
    while idx < len(span_chunk_list):
        spchunk = span_chunk_list[idx]
        next_spchunk = next_span_chunk(span_chunk_list, idx)
        next2_spchunk = next_span_chunk(span_chunk_list, idx + 1)
        # pylint: disable=too-many-boolean-expressions
        if spchunk.is_paren() and \
            next_spchunk and next_spchunk.is_comma() and \
           next2_spchunk and \
           (next2_spchunk.is_org() or
            next2_spchunk.has_label('xNNP')):
            # we need to make sure that this is not '(UK), Inc'
            # or 'XXX Shops Canada (Calgary), Inc'
            if not next2_spchunk.is_org_suffix():
                result.append(next_spchunk.tok_idx)
                idx += 2
        idx += 1
    return result


def find_between_tok_index(span_chunk_list: List[SpanChunk]) -> int:
    """Find the list of tok_idx of 'between' or 'among'.
    """
    for idx, spchunk in enumerate(span_chunk_list):
        if not spchunk.is_phrase() and \
           spchunk.is_word_between():
            next_spchunk = next_span_chunk(span_chunk_list, idx)
            if next_spchunk:
                return next_spchunk.tok_idx
    return -1


def find_delivered_by_tok_index(span_chunk_list: List[SpanChunk]) -> int:
    """Find the list of tok_idx of 'delivered by'
    """
    for idx, spchunk in enumerate(span_chunk_list):
        if spchunk.is_lc_word('by'):
            next_spchunk = next_span_chunk(span_chunk_list, idx)
            if next_spchunk:
                return next_spchunk.tok_idx
    return -1

def find_for_value_tok_index(span_chunk_list: List[SpanChunk]) -> int:
    """Find the list of tok_idx of 'delivered by'
    """
    for idx, spchunk in enumerate(span_chunk_list):
        if spchunk.is_lc_word('for'):
            next_spchunk = next_span_chunk(span_chunk_list, idx)
            next2_spchunk = next_span_chunk(span_chunk_list, idx+1)
            if next_spchunk and \
               next2_spchunk and \
               next_spchunk.get_words() == ['value'] and \
               next2_spchunk.is_lc_word('received'):
                return spchunk.tok_idx
    return -1


def find_order_of_tok_index(span_chunk_list: List[SpanChunk]) -> int:
    """Find the list of tok_idx of 'delivered by'
    """
    for idx, spchunk in enumerate(span_chunk_list):
        if spchunk.get_words() == ['the', 'order']:
            next_spchunk = next_span_chunk(span_chunk_list, idx)
            if next_spchunk and \
                next_spchunk.is_lc_word('of'):
                return next_spchunk.tok_idx
    return -1


def find_to_tok_index(span_chunk_list: List[SpanChunk]) -> int:
    """Find the list of tok_idx of 'to'
    """
    for idx, spchunk in enumerate(span_chunk_list):
        if spchunk.is_lc_word('to'):
            next_spchunk = next_span_chunk(span_chunk_list, idx)
            if next_spchunk:
                return next_spchunk.tok_idx
    return -1


def tokenize_to_span_chunks(sent_line: str) -> List[SpanChunk]:

    if IS_DEBUG_ORGS_TERM:
        print("\ntokenize_to_span_chunks({})".format(sent_line))

    # please see note in span_tokenize()
    sent_line = re.sub('[“"”]', '“', sent_line)
    # for OCR error ("ContectX"! -> ("ContectX")
    sent_line = re.sub(r'(\(“[^\)]+“)!', r'\1)', sent_line)

    chunk_list = get_org_aware_chunks(sent_line)
    se_tok_list = span_tokenize(sent_line)

    # for i, chunk in enumerate(chunk_list):
    #     print("2534 chunk #{} {}".format(i, chunk))

    span_chunk_list = []  # List[SpanChunk]
    tok_idx = 0
    span_chunk_idx = 0
    nempty_span_chunk_idx = 0
    for chunk in chunk_list:
        postag_list = chunk_to_postag_list(chunk)
        start_se = se_tok_list[tok_idx]
        end_se = se_tok_list[tok_idx + len(postag_list) - 1]
        tmp_is_phrase = is_chunk_tree(chunk)
        nempty_val = nempty_span_chunk_idx

        is_punct = False
        if not tmp_is_phrase:
            tag = postag_tag(chunk)
            if len(tag) == 1:  # is punct
                is_punct = True
                nempty_val = -1

        span_chunk_list.append(SpanChunk(start_se[0],
                                         end_se[1],
                                         span_chunk_idx,
                                         nempty_val,
                                         chunk,
                                         sent_line[start_se[0]:
                                                   end_se[1]],
                                         se_tok_list[tok_idx:tok_idx + len(postag_list)]))
        span_chunk_idx += 1
        if tmp_is_phrase:
            nempty_span_chunk_idx += 1
        else:  # not phrase
            if not is_punct:
                nempty_span_chunk_idx += 1

        # print("chunk: [{}]\t({},{}\t{})".format(sent_line[start_se[0]:
        #                                                  end_se[1]],
        #                                        start_se[0],
        #                                        end_se[1],
        #                                        chunk.label() if isinstance(chunk, Tree) else chunk))
        tok_idx += len(postag_list)
    return span_chunk_list



def find_span_chunks(sent_line: str):
    chunk_list = get_org_aware_chunks(sent_line)

    se_tok_list = span_tokenize(sent_line)

    span_chunk_list = []
    tok_idx = 0
    for chunk in chunk_list:
        postag_list = chunk_to_postag_list(chunk)
        start_se = se_tok_list[tok_idx]
        end_se = se_tok_list[tok_idx + len(postag_list) - 1]
        span_chunk_list.append((start_se[0], end_se[1], chunk))

        # print("chunk: [{}]\t({},{}\t{})".format(sent_line[start_se[0]:
        #                                                  end_se[1]],
        #                                        start_se[0],
        #                                        end_se[1],
        #                                        chunk.label() if isinstance(chunk, Tree) else chunk))
        tok_idx += len(postag_list)
    return span_chunk_list
