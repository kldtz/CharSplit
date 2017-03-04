import re

from kirke.utils import unicodeutils

"""
        self.binary_indices = [0, 1, 2, 3, 9]  # has_org, has_loc, has_person, has_date, hr ('yes', 'no')
        self.numeric_indices = [5, 6, 7, 8]  # ent_start, ent_end, ent_percent_start, nth_candidate
        self.numeric_indices.extend(list(range(18, 24)))  # length, prevLeng... nextLengthChar
        self.categorical_indices = list(range(10, 18))  # start*Class, startChar, endChar, prevChar, nextChar
        self.binary_indices.extend(list(range(24, 32)))
"""

def init_eb_attrs():
    attr_type_st_list = ['has_organization:bool', 'has_location:bool',
                         'has_person:bool', 'has_date:bool',
                         'contains_prep_phrase:bool',
                         'ent_start:numeric', 'ent_end:numeric',
                         'ent_percent_start:numeric',
                         'nth_candidate:numeric', 'hr:bool',
                         'startCharClass:categorical', 'endCharClass:categorical',
                         'prevCharClass:categorical', 'nextCharClass:categorical',
                         'startChar:categorical', 'endChar:categorical',
                         'prevChar:categorical', 'nextChar:categorical',
                         'length:numeric', 'prevLength:numeric',
                         'nextLength:numeric', 'lengthChar:numeric',
                         'prevLengthChar:numeric', 'nextLengthChar:numeric',
                         'le-3-word:bool', 'le-5-word:bool',
                         'le-10-word:bool', 'ge-05-lt-10-word:bool',
                         'ge-10-lt-20-word:bool', 'ge-20-lt-30-word:bool',
                         'ge-30-lt-40-word:bool', 'ge-40-word:bool']
    attr_type_map = {}
    attr_idx_map = {}
    attr_list = []
    binary_indices = []
    numeric_indices = []    
    categorical_indices = []
    for i, attr_type in enumerate(attr_type_st_list):
        attr, atype = attr_type.split(':')
        attr_list.append(attr)
        attr_type_map[attr] = atype
        attr_idx_map[attr] = i
        if atype == 'bool':
            binary_indices.append(i)
        elif atype == 'numeric':
            numeric_indices.append(i)
        elif atype == 'categorical':
            categorical_indices.append(i)
        else:
            raise ValueError('unknown type in init_eb_attrs(): {}'.format(atype))            
    return (attr_list, attr_type_map, attr_idx_map,
            binary_indices, numeric_indices, categorical_indices)

(eb_attr_list, eb_attr_type_map, eb_attr_idx_map,
 binary_indices, numeric_indices, categorical_indices) = init_eb_attrs()

DEFAULT_IS_TO_VALIDATE = True

# replacing sent2attrs
class EbAttrVec:

    def __init__(self, file_id, start, end):
        self.file_id = file_id
        self.start = start
        self.end = end
        self.attr_val_map = {}

        # fv.set_val('bag_of_words', ebsent.get_tokens_text())  # intentionally not using ebsent.get_text()
    
    def to_list(self):
        if DEFAULT_IS_TO_VALIDATE:
            self.validate_types()
        # result = [self.start, self.end]
        result= [self.attr_val_map.get(attr) for attr in eb_attr_list]
        # result.extend([','.join(self.labels), entities_to_st(self.entities), self.text])
        return result

    def get_val(self, attr_name):
        return self.vec[eb_attr_idx_map.get(attr_name)]

    def set_val(self, attr, val):
        self.attr_val_map[attr] = val

    def set_val_yesno(self, attr, val):
        if val:
            self.attr_val_map[attr] = True
        else:
            self.attr_val_map[attr] = False

    def validate_types(self):
        for attr, atype in eb_attr_type_map.items():
            val = self.attr_val_map.get(attr)
            if val is None:
                raise ValueError('failed in validate_type(), attr({}) is None'.format(attr))
            if atype == 'bool':
                if not isinstance(val, bool):
                    raise ValueError('failed in validate_type(), bool != {} {}'.format(type(val), val))
            elif atype == 'numeric':
                if not isinstance(val, (int, float)):
                    raise ValueError('failed in validate_type(), numeric != {} {}'.format(type(val), val))
            elif atype == 'categorical':
                pass
            else:
                raise ValueError('failed in validate_type(), unknown type: {}'.format(atype))


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
        fv.set_val('prevCharClass', unicodeutils.unicode_char_to_category_id(atext[ebsent.get_start() - 1]))
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
