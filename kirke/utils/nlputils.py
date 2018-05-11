
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

from kirke.utils import regexutils, strutils


# bank is after 'n.a.' because 'bank, n.a.' is more desirable
# 'Credit Suisse Ag, New York Branch', 39893.txt,  why 'branch' is early
# TODO, handle "The bank of Nova Scotia", this is NOT org suffix case
# TODO, not handling 'real estate holdings fiv'
# TODO, remove 'AS AMENDED' as a party, 'the customer'?
# TODO, 'seller, seller' the context?
ORG_SUFFIX_LIST = strutils.load_non_empty_str_list('dict/parties/organization.suffix.list')
PERS_SUFFIX_LIST = strutils.load_non_empty_str_list('dict/parties/person.suffix.list')

ORG_PERSON_SUFFIX_LIST = list(ORG_SUFFIX_LIST)
ORG_PERSON_SUFFIX_LIST.extend(PERS_SUFFIX_LIST)

# copied from kirke/ebrules/parties.py on 2/4/2016
ORG_PERSON_SUFFIX_PAT = regexutils.phrases_to_igs_pattern(ORG_PERSON_SUFFIX_LIST, re.I)
ORG_PERSON_SUFFIX_END_PAT = \
    re.compile(regexutils.phrases_to_igs_pattern_st(ORG_PERSON_SUFFIX_LIST) + r'\s*$', re.I)

# print("org_person_suffix_pattern_st:")
# print(regexutils.phrases_to_igs_pattern_st(ORG_PERSON_SUFFIX_LIST))

def org_person_suffix_search(line: str) -> Match[str]:
    return ORG_PERSON_SUFFIX_PAT.search(line)

def org_person_suffix_match(line: str) -> Match[str]:
    return ORG_PERSON_SUFFIX_PAT.match(line)

def is_org_suffix(line: str) -> bool:
    # print("is_org_suffix({})".format(line))
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

def get_org_suffix_mat_list(line: str) -> List[Match[str]]:
    """Get all org suffix matching mat extracted from line.

    Because of capitalization concerns, the word before org
    must not be lower cased, such as 'a limited company'.

    When there are multiple orgs that are consecutive, merge them.
    """

    lc_mat_list = list(ORG_PERSON_SUFFIX_PAT.finditer(line))
    result = []  # type: List[Match[str]]
    for lc_mat in lc_mat_list:
        prev_space_idx = lc_mat.start() -1
        # the previous word must be capitalized
        pword_start, pword_end, pword = strutils.find_previous_word(line, prev_space_idx)
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

    # when there is adjcent ones, take the last one
    # 'xxx Group, Ltd.', will return 'ltd'
    prev_mat = None
    result2 = [] # type: List[Match[str]]
    # Only if we now that the current mat is not adjacent to
    # the previous mat, we can add previous mat.
    # Remember the last one.
    for amat in result:
        # 2 is chosen, just in case, normally the diff is 1
        if prev_mat and amat.start() - prev_mat.end() > 2:
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
    match = FIRST_SENT_PAT.search(text)
    return match.group() if match else text


def jm_sent_tokenize(text: str) -> List[Tuple[int, int]]:
    if not text.strip():  # if empty or only spaces
        return []
    after_text = text.replace('\n', ' ')
    sent_start = 0
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
    # line = 'ContentX Technologies, EEC a California limited liability company  with an address at 19700 Fairchild, Ste. 260, Irvine, CA 92612 (“ContentX”! and Cybeitnesh International  Corporation, a Nevada corporation with an address at 2715 Indian Farm Lh NW, Albuquerque NM 87107  (“Cybermesh”) (Cybermesh and together with ContentX, the “Members" each a “Member”), pursuant to  the laws of the State of Nevada.'

    # tried it with '"', didn't work.  But this does.
    sent_line = re.sub('[“"”]', '“', sent_line)
    # for OCR error ("ContectX"! -> ("ContectX")
    sent_line = re.sub(r'(\(“[^\)]+“)!', r'\1)', sent_line)

    return TREEBANK_WORD_TOKENIZER.span_tokenize(sent_line)


def tokenize(sent_line: str) -> List[str]:
    """Returns all the span tokens for a sentence."""
    return TREEBANK_WORD_TOKENIZER.tokenize(sent_line)

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


def fix_nltk_pos(wpos_list: List[Tuple[str, str]],
                 t2_map: Dict[str, Tuple[str, str]] = None) -> List[Tuple[str, str]]:
    result = []
    for wpos in wpos_list:
        word, tag = wpos
        lc_word = word.lower()
        if lc_word == 'hereto':
            result.append((word, 'IN'))
        elif word.startswith('ZqZ'):
            if t2_map:
                # A token can be 'Corporation.' from 'Corporation.,'
                line, ttype = t2_map[word[:7]]
                if ttype == 'xABN':
                    # we don't want this to merge with anything
                    result.append((word, 'IN'))
                else:
                    result.append((word, 'NNP'))
            else:
                result.append((word, 'NNP'))
        elif strutils.is_both_alpha_and_num(word):
            result.append((word, 'NNP'))
        elif strutils.is_cap_not_first_char(word):
            # capitalize the first word for now
            result.append((word.capitalize(), 'NNP'))
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

"""
def get_consecutive_cap_word_phrases(line: str) \
    -> List[List[Tuple[int, int]]]:
"""

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

"""
chunker = RegexpParser(r'''
NP:
  {<\(.*><[^\(].*>*<\).*>}
  {<DT><NN.*><.*>*<NN.*>}
  {<CD><NN.*><.*>*<NN.*>}
  {<PRP.*><NN.*><.*>*<NN.*>}
  }<VB.*>{
  }<,.*>{
  }<IN.*>{
  }<CC.*>{
  }<\(.*>{
''')
"""

chunker = RegexpParser(r'''
xPAREN:
  {<\(.*><[^\(].*>*<\).*>}
xNNP:
  {<DT>?<JJ.*|CD>*<NN.*>*<NNP>+<NN.*>*}
xNP:
  {<DT|PRP.*>?<JJ.*|CD>*<NN.*>+}
  }<VB.*>{
  }<,.*>{
  }<IN.*>{
  }<CC.*>{
''')

def chunkize(sent_line: str) -> List[Union[Tree, Tuple[int, int]]]:
    tokens = word_tokenize(sent_line)
    tok_pos_list = pos_tag(tokens)
    # for i, tok_pos in enumerate(tok_pos_list):
    #     print('pos_tok #{}\t{}'.format(i, tok_pos))
    chunk_list = chunker.parse(tok_pos_list)
    # extract 'ABN ID+' as an apposition
    chunk_list = rechunk_known_terms(chunk_list)
    return chunk_list

def find_known_terms(line: str) -> List[Tuple[int, int, str]]:
    result = []  # type: List[Tuple[int, int, str]]:
    mat_list = get_org_suffix_mat_list(line)
    for mat in mat_list:
        result.append((mat.start(), mat.end(), 'xORGP'))

    mat_list = re.finditer(r'ABN( \d+)+', line, re.I)
    for mat in mat_list:
        result.append((mat.start(), mat.end(), 'xABN'))
    return result

def sub_known_terms(line: str,
                    se_ttype_list: List[Tuple[int, int, str]]) \
                    -> Tuple[str, Dict[str, Tuple[str, str]]]:
    known_term_map = {}  # type: Dict[str, str]:
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
    -> Tuple[bool, str, List[Tuple[str, str]]]:
    postags = chunk_to_postag_list(chunk)
    postags_out = []  # type: List[Tuple[str, str]]
    is_triggered = False
    ttype = None
    for word_pos in postags:
        word, pos = word_pos
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


def putback_kterms(chunk_list: List[Union[Tree, Tuple[str, str]]],
                   t2_map: Dict[str, Tuple[str, str]]) -> List[Union[Tree, Tuple[int, int]]]:
    result = []  # type: List[Union[Tree, Tuple[int, int]]]:
    for chunk in chunk_list:
        if is_chunk_label(chunk, 'xNNP'):
            is_triggered, ttype, postags_out = putback_chunk_to_postag_list(chunk, t2_map)
            if is_triggered:
                chunk = Tree(ttype, postags_out)
        elif is_chunk_label(chunk, 'xPAREN'):
            # it is possible that zqz is in xPAREN also
            is_triggered, ttype, postags_out = putback_chunk_to_postag_list(chunk, t2_map)
            if is_triggered:
                chunk = Tree(chunk.label(), postags_out)
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

            found_org_suffix_idx = -1
            for i in range(len(postag_list)-1, -1, -1):
                word, tag = postag_list[i]
                # 'WASTE2ENERGY GROUP company registed in ...', doc118.txt
                if is_org_suffix(word) and \
                        not word in set(['company', 'corporation']):
                    found_org_suffix_idx = i
                    break

            if found_org_suffix_idx == -1 or \
                found_org_suffix_idx == len(postag_list) -1:
                result.append(chunk)
            else:
                result.append(Tree('xORGP', postag_list[:i+1]))
                result.append(Tree('NNP', postag_list[i+1:]))
        else:
            result.append(chunk)

    return result


def chunkize2(sent_line: str) -> List[Union[Tree, Tuple[str, str]]]:
    se_term_list = find_known_terms(sent_line)
    sent_line_t2, t2_map = sub_known_terms(sent_line,
                                           se_term_list)
    tokens = word_tokenize(sent_line_t2)
    tok_pos_list = pos_tag(tokens, t2_map)

    # for i, tok_pos in enumerate(tok_pos_list):
    #     print('pos_tok #{}\t{}'.format(i, tok_pos))
    chunk_list = chunker.parse(tok_pos_list)

    chunk_list = putback_kterms(chunk_list,
                                t2_map)

    # split chunks that has org in the middle
    # 'Johnson & Johnson Medikal Sanayi Ve Ticaret Limited Sirketi'
    chunk_list = split_chunk_with_org(chunk_list)

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
                result.append(Tree(chunk.label(), postag_list[:-1]))
                result.append(Tree('ABN_NUM', abn_tok_list))
                idx = tmp_i
            else:
                result.append(chunk)
        else:
            result.append(chunk)
        idx += 1
    return result


def get_nouns(sent_line: str):
    chunk_list = chunkize(sent_line)
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
    chunk_list = chunkize(sent_line)
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

def has_pos_article(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    if isinstance(chunk, Tree):
        for postag in chunk:
            if is_postag_tag(postag, 'DT'):
                return True
        return False
    return is_postag_tag(chunk, 'DT')

def get_next_chunk(chunk_list, idx):
    if idx + 1 < len(chunk_list):
        return chunk_list[idx + 1]
    return None

def is_chunk_tree(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    return isinstance(chunk, Tree)

def get_prev_chunk(chunk_list, idx):
    if idx - 1 >= 0:
        return chunk_list[idx - 1]
    return None

def is_chunk_digit(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    if isinstance(chunk, Tree):
        return False
    return is_postag_tag(chunk, 'CD')

def is_chunk_and(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    if isinstance(chunk, Tree):
        return False
    # cannot check for tag == 'CC' because it can be 'or'
    word = postag_word(chunk)
    return word.lower() == 'and' or word == '&'

def is_chunk_ampersand(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    if isinstance(chunk, Tree):
        return False
    # cannot check for tag == 'CC' because it can be 'or'
    word = postag_word(chunk)
    return word == '&'

def is_chunk_paren(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    if isinstance(chunk, Tree):
        return chunk.label() == 'xPAREN'
    return False

def is_chunk_xnnp(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    if isinstance(chunk, Tree):
        return chunk.label() == 'xNNP' and not has_pos_article(chunk)
    return is_postag_tag(chunk, 'NNP')

def is_chunk_label(chunk: Union[Tree, Tuple[str, str]], label: str) -> bool:
    if isinstance(chunk, Tree):
        return chunk.label() == label
    return False

def is_chunk_org(chunk: Union[Tree, Tuple[str, str]]) -> bool:
    return isinstance(chunk, Tree) and \
        chunk.label() == 'xORGP'

def is_postag_tag(postag: Tuple[str, str], tag: str) -> bool:
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

    for i in range(len(address_chunk_list)):
        del chunks_result[-1]

    chunks_result.append(address_chunk)
    # for i, chunk in enumerate(address_chunk_list):
    # print("address chunk #{}\t{}".format(i, chunk))
    # for i in range(len(address_chunk_list)):

"""
def chunk_has_org_person_suffix(chunk: Tree) -> bool:
    if isinstance(chunk, Tree):
        for postag in chunk:
            word = postag_word(postag)
            if is_org_suffix(word):
                return True
        return False
    word = postag_word(chunk)
    return is_org_suffix(word)
"""

def update_with_org_person(chunks_result,
                           chunk_list,
                           chunk_idx: int,
                           chunk_taken_idx: int) \
    -> int:
    """If found org_person, update the result.

    returns new chunk_idx if digested more chunks than before.
    """

    # this is for debug purpose only, can del
    cur_chunk = chunk_list[chunk_idx]
    prefix_chunk_list = []
    suffix_chunk_list = []
    # find the end of the org
    i = chunk_idx + 1
    num_comma = 0
    while i < len(chunk_list):
        chunk = chunk_list[i]
        if is_chunk_label(chunk, 'xPAREN') or has_pos_article(chunk):
            break
        elif is_postag_comma(chunk):
            if num_comma >= 1:  # can only have 1 comma in proper name
                break
            suffix_chunk_list.append(chunk)
            i += 1
            num_comma += 1
        # elif chunk_has_org_person_suffix(chunk):
        elif is_chunk_label(chunk, 'xORGP'):
            postag_list = chunk_to_postag_list(chunk)
            # don't want to merge two full comany names together
            # only merge LLC or Corp.  Just in case an org is made
            # of 2 words, 'b. v.'
            if len(postag_list) > 2:
                break
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
    while i > chunk_taken_idx:
        chunk = chunk_list[i]
        prev_chunk = get_prev_chunk(chunk_list, i)
        if is_chunk_label(chunk, 'xPAREN'):
            if i == chunk_idx -1:  # right before org, 'Kodak (UK) Ltd'
                prefix_chunk_list.append(chunk)
                i -= 1
            else:
                break
        elif has_pos_article(chunk):
            prefix_chunk_list.append(chunk)
            i -= 1
            break
        elif is_postag_comma(chunk):
            if num_comma >= 1:
                break
            prefix_chunk_list.append(chunk)
            i -= 1
            num_comma += 1
        elif is_chunk_xnnp(chunk):
            prefix_chunk_list.append(chunk)
            i -= 1
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

def mark_an_org_not_org(chunk_list: List[Union[Tree, Tuple[str, str]]]) \
    -> None:
    for chunk in chunk_list:
        if is_chunk_org(chunk):
            postag_list = chunk_to_postag_list(chunk)
            word, tag = postag_list[0]
            # 'the Pass Through Trust'
            if tag == 'DT' and \
               (word == 'a' or word == 'an' or word == 'the'):
                chunk.set_label('xNN')
                continue

            # "Delaware corporation"
            if len(postag_list) < 3 and \
                postag_list[-1][0] == 'corporation':
                chunk.set_label('xNN')
                continue

            # company number
            is_all_lower = True
            for postag in postag_list:
                word, tag = postag
                if word[0].isupper():
                    is_all_lower = False
            if is_all_lower:
                chunk.set_label('xNN')


def get_better_nouns(sent_line: str):
    # chunk_list = chunkize(sent_line)
    chunk_list = chunkize2(sent_line)

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
    chunk_taken_idx = -1
    while chunk_idx < chunk_list_len:
        chunk = chunk_list[chunk_idx]
        if isinstance(chunk, Tree) and \
           chunk.label() in set(['xNNP', 'xORGP']):
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
            else:
                result.append(chunk)
        else:
            # print('pos_tag #{}\t{}'.format(i, tree))
            # pass
            result.append(chunk)
        chunk_idx += 1

    # mark_an_org_not_org(chunk_list)
    mark_org_appositions(result)

    return result


def extract_proper_names(sent_line: str):
    chunk_list = get_better_nouns(sent_line)

    se_tok_list = span_tokenize(sent_line)

    """
    tok_idx = 0
    result = []
    for chunk in chunk_list:
        postag_list = chunk_to_postag_list(chunk)
        result.extend(postag_list)

    print("len(se_tok_list) = {}".format(len(se_tok_list)))
    print("len(chunk_toks) = {}".format(len(result)))

    for i in range(min(len(se_tok_list),
                       len(result))):
        start, end = se_tok_list[i]
        chunk_tok = result[i]
        print("se_tok_list[{}]: {} {} [{}], chunk_tok = {}".format(i, start, end,
                                                                   sent_line[start:end],
                                                                   chunk_tok))
    """
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

    def __init__(self,
                 start: int,
                 end: int,
                 tok_idx: int,
                 nempty_tok_idx: int,
                 chunk: Union[Tree, Tuple[str, str]],
                 text: str,
                 se_tok_list: List[Tuple[int, int]]):
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
        return self.chunk.label()

    def has_label(self, label: str) -> bool:
        return is_chunk_tree(self.chunk) and self.chunk.label() == label

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

    def is_word_and(self) -> bool:
        return self.is_lc_word('and') or self.is_lc_word('&')

    def is_word_between(self) -> bool:
        return not self.is_phrase() and \
            self.get_lc_word() in set(['and', 'among'])

    def is_org(self) -> bool:
        return self.has_label('xORGP')

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

    def __init__(self, sent_line: str, is_chopped: bool) -> None:
        self.is_chopped = is_chopped
        self.span_chunk_list = tokenize_to_span_chunks(sent_line)
        self.good_cand_list = []  # type: List[SpanChunk]
        self.maybe_cand_list = []  # type: List[SpanChunk]
        self.bad_cand_list = []  # type: List[SpanChunk]
        self.unknown_cand_list = []  # type: List[SpanChunk]
        self.not_tree_list = []  # type: List[SpanChunk]

        self.parse_nouns()  # populate the above lists

    def parse_nouns(self) -> None:
        for i, span_chunk in enumerate(self.span_chunk_list):
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

        result = []  # type: List[Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]]:
        for i, orgs_term in enumerate(orgs_term_list):
            orgs, term = orgs_term

            orgs_offset = []  # type: List[Tuple[int, int]]
            for org in orgs:
                orgs_offset.append((org.start, org.end))
            term_offset = None
            if term:
                term_offset = (term[0].start, term[-1].end)
            result.append((orgs_offset, term_offset))
        return result

    def extract_orgs_term_list(self) \
        -> List[Tuple[List[SpanChunk], List[SpanChunk]]]:
        """Find the list of org_person_term groups.

        There can be multiple org per term.
        """
        DEBUG = False
        and_org_tok_idx_list = find_and_org_tok_indices(self.span_chunk_list)
        if not and_org_tok_idx_list:
            and_org_tok_idx_list = find_payto_org_tok_indices(self.span_chunk_list)
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

        result = []  # type: List[Tuple[List[SpanChunk], Optional[SpanChunk]]]
        for i, org_term_spchunk_list in enumerate(org_term_spchunk_list_list):
            if DEBUG:
                print("\norg_term_spchunk_list #{}".format(i))
                for j, spchunk in enumerate(org_term_spchunk_list):
                    print("    chunk #{}\t{}".format(j, spchunk))

            orgs_term = extract_orgs_term_in_span_chunk_list(org_term_spchunk_list)
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

        result = []  # type: List[Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]]]]:
        orgs, term = orgs_term
        orgs_offset = []  # type: List[Tuple[int, int]]
        for org in orgs:
            orgs_offset.append((org.start, org.end))
        term_offset = None
        if term:
            # term is a list of SpanList
            term_offset = (term[0].start, term[-1].end)

        return orgs_offset, term_offset


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
        word, tag = postag
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

        sc_start = cur_span_chunk.se_tok_list[0][0]
        nstart, nend = se_tok_list_out[0][0], se_tok_list_out[-1][1]
        shorten_text = cur_span_chunk.text[nstart - sc_start:nend - sc_start],
        shorten_span_chunk = SpanChunk(nstart,
                                       nend,
                                       cur_span_chunk.tok_idx + cur_tok_idx,
                                       cur_span_chunk.nempty_tok_idx + cur_tok_idx,
                                       Tree('xNNP', postags_out),
                                       shorten_text,
                                       se_tok_list_out)
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
        if re.search(r'\b(agreement|note)s?\b', span_chunk.text, re.I):
            pass
        elif re.search(r'\bdate\b', span_chunk.text, re.I):
            # (as effective date)
            pass
        elif re.search(r'\b(section|article)\b', span_chunk.text, re.I):
            # (as effective date)
            pass
        else:
            result.append(span_chunk)
    return result

# because of 'term' can be a list of words in "as xxx xxx", so it is a list
def extract_orgs_term_in_span_chunk_list(span_chunk_list: List[SpanChunk]) \
    -> Optional[Tuple[List[SpanChunk], List[SpanChunk]]]:
    paren_list = [span_chunk for span_chunk in span_chunk_list if span_chunk.has_label('xPAREN')]
    org_list = [span_chunk for span_chunk in span_chunk_list if span_chunk.has_label('xORGP')]

    if not org_list:  # didn't find any org based on org_suffix
        # for i, span_chunk in enumerate(span_chunk_list):
            # print("5234 #{} span_chunk: {}".format(i, span_chunk))
        span_chunk = span_chunk_list[0]
        next1_span_chunk = next_span_chunk(span_chunk_list, 0)
        next2_span_chunk = next_span_chunk(span_chunk_list, 1)
        if span_chunk.is_phrase() and \
           span_chunk.has_label('xNNP') and \
            next1_span_chunk and next1_span_chunk.is_lc_word('and') and\
            next2_span_chunk and next2_span_chunk.has_label('xNNP'):
            tmp_span_chunk_list = span_chunk_list[0:3]
            org_list.append(make_span_chunk_from_span_chunk_list(tmp_span_chunk_list))
        elif span_chunk.is_phrase() and \
           span_chunk.has_label('xNNP'):
            org_list.append(span_chunk)

    as_list = find_as_span_chunks(span_chunk_list)

    term = []
    paren_list = remove_invalid_defined_terms_parens(paren_list)
    if paren_list:
        if len(paren_list) == 1:
            term = [chop_spanchunk_paren(paren_list[0])]
        else:  # there are multiple
            # if the 2nd paren has 'each a  "Party", and collectively the "Parties"'
            last_paren = paren_list[-1]
            if re.search(r'\b(parties)\b', last_paren.text, re.I):
                # if 'parties' is the last one, now take the first one instead
                term = [chop_spanchunk_paren(paren_list[0])]
            else:
                term = [chop_spanchunk_paren(last_paren)]
            # in future, might check if term/paren doesn't overlap with org
            """
            if last_paren.nempty_tok_idx >= span_chunk_list[-1] - 3:  # parent is really at the end of phrase
                term = last_paren
            else:
                term = paren_list[0]
            """
        # if there is 'as' in this list
        as_list = find_as_in_paren(term)
        if as_list:
            # remove the last paren
            term = as_list

    elif as_list:
        if len(as_list) == 1:
            term = as_list[0]  # List[SpanChunk]
        elif len(as_list) == 2:
            # as a “Party” and together as the “Parties”.
            term = as_list[0]  # List[SpanChunk]
            term.extend(as_list[1])
        else:  # there are multiple, take the first one
            term = as_list[0]
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
    for idx in range(len(span_chunk_list)):
        spchunk = span_chunk_list[idx]
        next_spchunk = next_span_chunk(span_chunk_list, idx)
        if spchunk.is_word_and() and \
           next_spchunk:
           if next_spchunk.is_org():
               result.append(next_spchunk.tok_idx)
               idx += 1
           elif next_spchunk.has_label('xNNP'):
               # 'and Arrayit Diagnostics', doc111.txt
               maybe_result.append(next_spchunk.tok_idx)
               idx += 1

        idx += 1

    # replace result only if valid and-org is not found
    if not result and maybe_result:
        result = maybe_result
    return result


def find_payto_org_tok_indices(span_chunk_list: List[SpanChunk]) -> List[int]:
    """Find the list of tok_idx of 'pay to'.
    """
    result = []  # type: List[int]
    # This is for 'and' followed by xNNP, so might or might not be and-org
    maybe_result = []  # type: List[int]
    for idx in range(len(span_chunk_list)):
        spchunk = span_chunk_list[idx]
        next_spchunk = next_span_chunk(span_chunk_list, idx)
        next2_spchunk = next_span_chunk(span_chunk_list, idx+1)
        if spchunk.is_lc_word('pay') and \
           next_spchunk and \
           next_spchunk.is_lc_word('to') and\
           next2_spchunk:
           if next2_spchunk.is_org():
               result.append(next2_spchunk.tok_idx)
               idx += 2
           elif next2_spchunk.has_label('xNNP'):
               # 'and Arrayit Diagnostics', doc111.txt
               maybe_result.append(next2_spchunk.tok_idx)
               idx += 2

        idx += 1

    # replace result only if valid and-org is not found
    if not result and maybe_result:
        result = maybe_result
    return result



def find_paren_org_tok_indices(span_chunk_list: List[SpanChunk]) -> List[int]:
    """Find the list of tok_idx of '), org'.
    """
    result = []  # type: List[int]
    for idx in range(len(span_chunk_list)):
        spchunk = span_chunk_list[idx]
        next_spchunk = next_span_chunk(span_chunk_list, idx)
        next2_spchunk = next_span_chunk(span_chunk_list, idx + 1)
        if spchunk.is_paren() and \
            next_spchunk and next_spchunk.is_comma() and \
           next2_spchunk and next2_spchunk.is_org():
            result.append(next_spchunk.tok_idx)
            idx += 2
        idx += 1
    return result

def find_between_tok_index(span_chunk_list: List[SpanChunk]) -> int:
    """Find the list of tok_idx of 'between' or 'among'.
    """
    for idx in range(len(span_chunk_list)):
        spchunk = span_chunk_list[idx]
        if not spchunk.is_phrase() and \
           spchunk.is_word_between():
            next_spchunk = next_span_chunk(span_chunk_list, idx)
            if next_spchunk:
                return next_spchunk.tok_idx
    return -1


def tokenize_to_span_chunks(sent_line: str) -> List[SpanChunk]:
    # please see note in span_tokenize()
    sent_line = re.sub('[“"”]', '“', sent_line)
    # for OCR error ("ContectX"! -> ("ContectX")
    sent_line = re.sub(r'(\(“[^\)]+“)!', r'\1)', sent_line)

    chunk_list = get_better_nouns(sent_line)
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
    chunk_list = get_better_nouns(sent_line)

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


def find_nouns(sent_line: str) -> List[Tuple[int, int, Tree]]:
    span_chunk_list = find_span_chunks(sent_line)

    good_cand_list = []
    maybe_cand_list = []
    bad_cand_list = []
    unknown_cand_list = []
    not_tree_list = []
    for i, span_chunk in enumerate(span_chunk_list):
        cstart, cend, chunk = span_chunk
        if is_chunk_tree(chunk):
            print("achunk #{}\t{}\t{}".format(i, chunk.label(), chunk))
            if chunk.label() == 'NP':
                bad_cand_list.append(span_chunk)
            elif chunk.label() in set(['xORGP',
                                       'xAPPOSITION_ORG_PERSON',
                                       'xPAREN',
                                       'xADDRESS']):
                good_cand_list.append(span_chunk)
            elif chunk.label() in set(['xNNP']):
                maybe_cand_list.append(span_chunk)
            # elif chunk.label() in set('NP'):
            #     bad_cand_list.append(span_chunk)
            else:
                unknown_cand_list.append(span_chunk)
        else:
            not_tree_list.append(span_chunk)

    for alist in [('good cand', good_cand_list),
                  ('maybe cand', maybe_cand_list),
                  ('unknown cand', unknown_cand_list),
                  ('bad cand', bad_cand_list),
                  ('not_tree cand', not_tree_list)]:
        cat, alist = alist
        print("{}:".format(cat))
        for i, span_chunk in enumerate(alist):
            cstart, cend, chunk = span_chunk
            if is_chunk_tree(chunk):
                print("    #{}\t({}, {})\t{}\t{}".format(i, cstart, cend,
                                                         chunk.label(), chunk))
            else:
                print("    #{}\t({}, {})\t{}\t{}".format(i, cstart, cend,
                                                         'NOT_TREE', chunk))




