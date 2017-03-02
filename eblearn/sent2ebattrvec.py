
from utils import unicodeutils, mathutils
from eblearn import ebattr

import re

hr_pat = re.compile(r'(\*\*\*+)|(\.\.\.\.+)|(\-\-\-+)')


def sent2ebattrvec(file_id, ebsent, sent_seq, prev_ebsent, next_ebsent, atext):
    tokens = ebsent.get_tokens()
    text_len = len(atext)

    # TODO, pass in the token list, with lemma
    # will do chunking in the future also
    fv = ebattr.EbAttrVec(file_id, ebsent.get_start(), ebsent.get_end())

    fv.set_val('ent_start', ebsent.get_start())
    fv.set_val('ent_end', ebsent.get_end())        
    fv.set_val('ent_percent_start', 1.0 * ebsent.get_start() / text_len)
    # fv.set_val('nth_candidate', sent_seq)
    fv.set_val('nth_candidate', sent_seq - 1)  # prod version starts with 0
    if prev_ebsent == None:   # the first sentence
        fv.set_val('prevLength', 0)        # number of words in a sentence
        fv.set_val('prevLengthChar', 0)
    else:
        fv.set_val('prevLength', prev_ebsent.get_number_tokens())
        fv.set_val('prevLengthChar', prev_ebsent.get_number_chars())

    if next_ebsent == None:
        fv.set_val('nextLength', 0)
        fv.set_val('nextLengthChar', 0)
    else:
        fv.set_val('nextLength', next_ebsent.get_number_tokens())
        fv.set_val('nextLengthChar', next_ebsent.get_number_chars())

    if ebsent.get_start() > 0:
        fv.set_val('prevChar', ord(atext[ebsent.get_start()-1]))
        fv.set_val('prevCharClass', unicodeutils.unicode_char_to_category_id(atext[ebsent.get_start()-1]))            
    else:
        fv.set_val('prevChar', 0)       # cannot set to -1 because OneHotEncode don't like negative numbers
        fv.set_val('prevCharClass', 0)  # Cn: Other, not assigned

    if ebsent.get_end() < text_len - 1:
        fv.set_val('nextChar', ord(atext[ebsent.get_end()]))
        fv.set_val('nextCharClass', unicodeutils.unicode_char_to_category_id(atext[ebsent.get_end()]))                        
    else:
        fv.set_val('nextChar', 0)       # cannot set to -1 because OneHotEncode don't like negative numbers
        fv.set_val('nextCharClass', 0)  # Cn: Other, not assigned

    num_sent_tokens = ebsent.get_number_tokens()
    fv.set_val('length', num_sent_tokens)
    fv.set_val('lengthChar', ebsent.get_number_chars())

    fv.set_val_yesno('le-3-word', num_sent_tokens <= 3)
    fv.set_val_yesno('le-5-word', num_sent_tokens <= 5)
    fv.set_val_yesno('le-10-word', num_sent_tokens <= 10)
    fv.set_val_yesno('ge-05-lt-10-word', num_sent_tokens >= 5 and num_sent_tokens < 10)
    fv.set_val_yesno('ge-10-lt-20-word', num_sent_tokens >= 10 and num_sent_tokens < 20)
    fv.set_val_yesno('ge-20-lt-30-word', num_sent_tokens >= 20 and num_sent_tokens < 30)
    fv.set_val_yesno('ge-30-lt-40-word', num_sent_tokens >= 30 and num_sent_tokens < 40)
    fv.set_val_yesno('ge-40-word', num_sent_tokens >= 40)

    sent_text = ebsent.get_text()
    fv.set_val('startCharClass', unicodeutils.unicode_char_to_category_id(sent_text[0]))
    fv.set_val('endCharClass', unicodeutils.unicode_char_to_category_id(sent_text[-1]))
    fv.set_val('startChar', ord(sent_text[0]))
    fv.set_val('endChar', ord(sent_text[-1]))        

    fv.set_val_yesno('hr', hr_pat.search(atext))

    # feature not directly set, so use default values
    fv.set_val_yesno('contains_prep_phrase', False)

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
    fv.set_val_yesno('has_person', has_person)
    fv.set_val_yesno('has_location', has_location)
    fv.set_val_yesno('has_organization', has_org)
    fv.set_val_yesno('has_date', has_date)

    return fv
