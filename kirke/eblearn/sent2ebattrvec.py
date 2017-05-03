import re
import time
import logging

from kirke.utils import unicodeutils, entityutils
from kirke.eblearn import ebattrvec

# truncate the following features to avoid outlier issues
ENT_START_MAX = 50000
ENT_END_MAX = 50010
NTH_CANDIDATE_MAX = 500
NUM_CHARS_MAX = 300
NUM_WORDS_MAX = 50

HR_PAT = re.compile(r'(\*\*\*+)|(\.\.\.\.+)|(\-\-\-+)')

# AGREEMENT_PAT = re.compile(r'(["“”][^"“”]*agreement["“”])', re.IGNORECASE)

def count_define_party(line: str) -> int:
    return len(entityutils.extract_define_party(line))

AGREEMENT_PAT = re.compile(r'\bagreements?\b', re.IGNORECASE)

def has_word_agreement(line: str) -> bool:
    return AGREEMENT_PAT.search(line)

#    patlist = AGREEMENT_PAT.finditer(line)
#    result = []
#    if patlist:
#        for pat1 in patlist:
#            result.append((pat1.group(1), pat1.start(1), pat1.end(1)))
#    return len(result)

BTW_PAT = re.compile(r'\b(between|among)\b', re.IGNORECASE)

def has_word_between(line: str) -> bool:
    return BTW_PAT.search(line)


# pylint: disable=R0912,R0913,R0914,R0915
def sent2ebattrvec(file_id, ebsent, sent_seq, prev_ebsent, next_ebsent, atext):
    tokens = ebsent.get_tokens()
    text_len = len(atext)
    raw_sent_text = atext[ebsent.start:ebsent.end]

    # TODO, pass in the token list, with lemma
    # will do chunking in the future also
    fvec = ebattrvec.EbAttrVec(file_id,
                               ebsent.start, ebsent.end,
                               ebsent.get_tokens_text(), ebsent.labels, ebsent.entities)

    tmp_start = min(ENT_START_MAX, ebsent.start)
    tmp_end = min(ENT_END_MAX, ebsent.end)
    fvec.set_val('ent_start', tmp_start)
    fvec.set_val('ent_end', tmp_end)
    fvec.set_val('ent_percent_start', 1.0 * ebsent.start / text_len)
    # fvec.set_val('nth_candidate', sent_seq)
    fvec.set_val('nth_candidate', min(NTH_CANDIDATE_MAX, sent_seq - 1))  # prod version starts with 0
    if prev_ebsent is None:   # the first sentence
        fvec.set_val('prevLength', 0)        # number of words in a sentence
        fvec.set_val('prevLengthChar', 0)
    else:
        fvec.set_val('prevLength', min(NUM_WORDS_MAX, prev_ebsent.get_number_tokens()))
        fvec.set_val('prevLengthChar', min(NUM_CHARS_MAX, prev_ebsent.get_number_chars()))

    if next_ebsent is None:
        fvec.set_val('nextLength', 0)
        fvec.set_val('nextLengthChar', 0)
    else:
        fvec.set_val('nextLength', min(NUM_WORDS_MAX, next_ebsent.get_number_tokens()))
        fvec.set_val('nextLengthChar', min(NUM_CHARS_MAX, next_ebsent.get_number_chars()))

    if ebsent.start > 0:
        prev_char = atext[ebsent.start - 1]
        fvec.set_val('prevChar', ord(prev_char))
        fvec.set_val('prevCharClass',
                     unicodeutils.unicode_char_to_category_id(prev_char))
    else:
        # cannot set to -1 because OneHotEncode don't like negative numbers
        fvec.set_val('prevChar', 0)
        fvec.set_val('prevCharClass', 0)  # Cn: Other, not assigned

    if ebsent.end < text_len - 1:
        next_char = atext[ebsent.end]
        fvec.set_val('nextChar', ord(next_char))
        fvec.set_val('nextCharClass',
                     unicodeutils.unicode_char_to_category_id(next_char))
    else:
        # cannot set to -1 because OneHotEncode don't like negative numbers
        fvec.set_val('nextChar', 0)
        fvec.set_val('nextCharClass', 0)  # Cn: Other, not assigned

    num_sent_tokens = ebsent.get_number_tokens()
    fvec.set_val('length', min(NUM_WORDS_MAX, num_sent_tokens))
    fvec.set_val('lengthChar', min(NUM_CHARS_MAX, ebsent.get_number_chars()))

    fvec.set_val_yesno('le-3-word', num_sent_tokens <= 3)
    fvec.set_val_yesno('le-5-word', num_sent_tokens <= 5)
    fvec.set_val_yesno('le-10-word', num_sent_tokens <= 10)
    fvec.set_val_yesno('ge-05-lt-10-word',
                       num_sent_tokens >= 5 and num_sent_tokens < 10)
    fvec.set_val_yesno('ge-10-lt-20-word',
                       num_sent_tokens >= 10 and num_sent_tokens < 20)
    fvec.set_val_yesno('ge-20-lt-30-word',
                       num_sent_tokens >= 20 and num_sent_tokens < 30)
    fvec.set_val_yesno('ge-30-lt-40-word',
                       num_sent_tokens >= 30 and num_sent_tokens < 40)
    fvec.set_val_yesno('ge-40-word', num_sent_tokens >= 40)

    # for 'party'
    num_define_party = count_define_party(raw_sent_text)
    fvec.set_val_yesno('is-1-num-define-party', num_define_party == 1)
    fvec.set_val_yesno('is-2-num-define-party', num_define_party == 2)
    fvec.set_val_yesno('is-ge2-num-define-party', num_define_party >= 2)
    fvec.set_val_yesno('has-define-party', num_define_party > 0)
    fvec.set_val_yesno('has-define-agreement', has_word_agreement(raw_sent_text))
    fvec.set_val_yesno('has-word-between', has_word_between(raw_sent_text))

    start_char = raw_sent_text[0]
    end_char = raw_sent_text[-1]
    fvec.set_val('startCharClass',
                 unicodeutils.unicode_char_to_category_id(start_char))
    fvec.set_val('endCharClass', unicodeutils.unicode_char_to_category_id(end_char))
    fvec.set_val('startChar', ord(start_char))
    fvec.set_val('endChar', ord(end_char))

    fvec.set_val_yesno('hr', HR_PAT.search(raw_sent_text))

    # feature not directly set, so use default values
    fvec.set_val_yesno('contains_prep_phrase', False)

    # print(sent_ajson)
    has_person, has_location, has_org, has_date = (False, False, False, False)
    for token in tokens:
        ner = token.ner
        # if ner != 'O':
        #     print('ner = ' + str(ner))
        if ner == 'PERSON':
            has_person = True
        elif ner == 'LOCATION':
            has_location = True
        elif ner == 'ORGANIZATION':
            has_org = True
        elif ner == 'DATE':
            has_date = True
    fvec.set_val_yesno('has_person', has_person)
    fvec.set_val_yesno('has_location', has_location)
    fvec.set_val_yesno('has_organization', has_org)
    fvec.set_val_yesno('has_date', has_date)

    return fvec
