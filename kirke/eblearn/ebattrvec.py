#!/usr/bin/env python

"""
# has_org, has_loc, has_person, has_date, hr ('yes', 'no')
binary_indices = [0, 1, 2, 3, 9]

# ent_start, ent_end, ent_percent_start, nth_candidate
numeric_indices = [5, 6, 7, 8]

# length, prevLeng... nextLengthChar
numeric_indices.extend(list(range(18, 24)))

# start*Class, startChar, endChar, prevChar, nextChar
categorical_indices = list(range(10, 18))

binary_indices.extend(list(range(24, 32)))
"""

attr_type_st_list = [
    'has_organization:bool', 'has_location:bool',
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

# lemma as text/bag-of-words give 3% worse result
# now use corenlp token instead
misc_attr_type_st_list = [
    'bag-of-words:string', 'labels:string-list',
    'entities:other']

extra_party_attr_type_st_list = [
    'is-1-num-define-party:bool',
    'is-2-num-define-party:bool',
    'is-ge2-num-define-party:bool',
    'has-define-party:bool',
    'has-define-agreement:bool',
    'has-word-between:bool']

default_attr_type_st_list = attr_type_st_list + misc_attr_type_st_list
default_attr_type_list = [tuple(attr_type.split(':')) for attr_type
                          in default_attr_type_st_list]
default_attr_st_list = [attr_type[0] for attr_type
                        in default_attr_type_list]

party_attr_type_st_list = (attr_type_st_list + extra_party_attr_type_st_list +
                           misc_attr_type_st_list)
party_attr_type_list = [tuple(attr_type.split(':')) for attr_type
                        in party_attr_type_st_list]
party_attr_st_list = [attr_type[0] for attr_type
                      in party_attr_type_list]

binary_attr_list = [attr_type[0] for attr_type
                    in party_attr_type_list if attr_type[1] == 'bool']
numeric_attr_list = [attr_type[0] for attr_type
                    in party_attr_type_list if attr_type[1] == 'numeric']
categorical_attr_list = [attr_type[0] for attr_type
                    in party_attr_type_list if attr_type[1] == 'categorical']
string_attr_list = [attr_type[0] for attr_type
                    in party_attr_type_list if attr_type[1] == 'string']
string_list_attr_list = [attr_type[0] for attr_type
                    in party_attr_type_list if attr_type[1] == 'string-list']
other_attr_list = [attr_type[0] for attr_type
                    in party_attr_type_list if attr_type[1] not in ['bool',
                                                                    'numeric',
                                                                    'categorical',
                                                                    'string',
                                                                    'string-list']]


print("default_attr_st_list: {}".format(default_attr_st_list))
print("party_attr_st_list: {}".format(party_attr_st_list))

print("binary_attr_list: {}".format(binary_attr_list))
print("numeric_attr_list: {}".format(numeric_attr_list))
print("categorical_attr_list: {}".format(categorical_attr_list))
print("string_attr_list: {}".format(string_attr_list))
print("string_list_attr_list: {}".format(string_list_attr_list))
print("other_attr_list: {}".format(other_attr_list))
    
# replacing sent2attrs
class EbAttrVec:

    def __init__(self, file_id, start, end, sent_text, labels, entities):
        self.file_id = file_id
        self.start = start  # this differs from ent_start, which can be chopped
        self.end = end      # similar to above, ent_end
        self.bag_of_words = sent_text
        self.labels = labels
        self.entities = entities

    def get_val(self, attr_name):
        return getattr(self, attr_name)

    def set_val(self, attr, val):
        setattr(self, attr, val)

    def set_val_yesno(self, attr, val):
        setattr(self, attr, bool(val))

