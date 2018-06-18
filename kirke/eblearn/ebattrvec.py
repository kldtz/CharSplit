#!/usr/bin/env python3

from typing import List, Optional

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

DEFAULT_ATTR_TYPE_ST_LIST = [
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
    'le_3_word:bool', 'le_5_word:bool',
    'le_10_word:bool', 'ge_05_lt_10_word:bool',
    'ge_10_lt_20_word:bool', 'ge_20_lt_30_word:bool',
    'ge_30_lt_40_word:bool', 'ge_40_word:bool']

# lemma as text/bag_of_words give 3% worse result
# now use corenlp token instead
#MISC_ATTR_TYPE_ST_LIST = [
#    'bag_of_words:string', 'labels:string_list',
#    'entities:other']

EXTRA_PARTY_ATTR_TYPE_ST_LIST = [
    'is_1_num_define_party:bool',
    'is_2_num_define_party:bool',
    'is_ge2_num_define_party:bool',
    'has_define_party:bool',
    'has_define_agreement:bool',
    'has_word_between:bool']

DEFAULT_ATTR_TYPE_LIST = [tuple(attr_type.split(':')) for attr_type
                          in DEFAULT_ATTR_TYPE_ST_LIST]
DEFAULT_ATTR_ST_LIST = [attr_type[0] for attr_type
                        in DEFAULT_ATTR_TYPE_LIST]

PARTY_ATTR_TYPE_ST_LIST = DEFAULT_ATTR_TYPE_ST_LIST + EXTRA_PARTY_ATTR_TYPE_ST_LIST
PARTY_ATTR_TYPE_LIST = [tuple(attr_type.split(':')) for attr_type
                        in PARTY_ATTR_TYPE_ST_LIST]
PARTY_ATTR_ST_LIST = [attr_type[0] for attr_type
                      in PARTY_ATTR_TYPE_LIST]

DEFAULT_BINARY_ATTR_LIST = [attr_type[0] for attr_type
                            in DEFAULT_ATTR_TYPE_LIST
                            if attr_type[1] == 'bool']
DEFAULT_NUMERIC_ATTR_LIST = [attr_type[0] for attr_type
                             in DEFAULT_ATTR_TYPE_LIST
                             if attr_type[1] == 'numeric']
DEFAULT_CATEGORICAL_ATTR_LIST = [attr_type[0] for attr_type
                                 in DEFAULT_ATTR_TYPE_LIST
                                 if attr_type[1] == 'categorical']

PARTY_BINARY_ATTR_LIST = [attr_type[0] for attr_type
                          in PARTY_ATTR_TYPE_LIST
                          if attr_type[1] == 'bool']
PARTY_NUMERIC_ATTR_LIST = [attr_type[0] for attr_type
                           in PARTY_ATTR_TYPE_LIST
                           if attr_type[1] == 'numeric']
PARTY_CATEGORICAL_ATTR_LIST = [attr_type[0] for attr_type
                               in PARTY_ATTR_TYPE_LIST if
                               attr_type[1] == 'categorical']


# print("default_attr_st_list: {}".format(DEFAULT_ATTR_ST_LIST))
# print("party_attr_st_list: {}".format(PARTY_ATTR_ST_LIST))

# print("default_binary_attr_list: {}".format(DEFAULT_BINARY_ATTR_LIST))
# print("default_numeric_attr_list: {}".format(DEFAULT_NUMERIC_ATTR_LIST))
# print("default_categorical_attr_list: {}".format(DEFAULT_CATEGORICAL_ATTR_LIST))

# print("party_binary_attr_list: {}".format(PARTY_BINARY_ATTR_LIST))
# print("party_numeric_attr_list: {}".format(PARTY_NUMERIC_ATTR_LIST))
# print("party_categorical_attr_list: {}".format(PARTY_CATEGORICAL_ATTR_LIST))

v1name_to_v1_2name_map = {
    'le-3-word': 'le_3_word',
    'le-5-word': 'le_5_word',
    'le-10-word': 'le_10_word',
    'ge-05-lt-10-word': 'ge_05_lt_10_word',
    'ge-10-lt-20-word': 'ge_10_lt_20_word',
    'ge-20-lt-30-word': 'ge_20_lt_30_word',
    'ge-30-lt-40-word': 'ge_30_lt_40_word',
    'ge-40-word': 'ge_40_word',
    'is-1-num-define-party': 'is_1_num_define_party',
    'is-2-num-define-party': 'is_2_num_define_party',
    'is-ge2-num-define-party': 'is_ge2_num_define_party',
    'has-define-party': 'has_define_party',
    'has-define-agreement': 'has_define_agreement',
    'has-word-between': 'has_word_between'
}

def v1name_to_v1_2name(feat_name: str):
    return v1name_to_v1_2name_map.get(feat_name, feat_name)

# replacing sent2attrs
class EbAttrVec:
    """Store values representing a sentence.  It is the major part
       of ebantdoc"""

    # Intentionally not making this a slot because we want flexibility in defining
    # new features.  There are many other sentence features, such as 'prevLength', etc.
    # Please see kirke.eblearn.sent2ebattrvec
    # Now, we care about memory footprint, so we use __slots__
    __slots__ = ['file_id', 'start', 'end', 'bag_of_words', 'labels',
                 'entities', 'sechead',
                 'ent_start',
                 'ent_end',
                 'ent_percent_start',
                 'nth_candidate',
                 'prevLength', 'prevLengthChar',
                 'nextLength', 'nextLengthChar',
                 'prevChar', 'prevCharClass',
                 'nextChar', 'nextCharClass',
                 'length', 'lengthChar',
                 'le_3_word', 'le_5_word', 'le_10_word',
                 'ge_05_lt_10_word',
                 'ge_10_lt_20_word',
                 'ge_20_lt_30_word',
                 'ge_30_lt_40_word',
                 'ge_40_word',
                 'is_1_num_define_party',
                 'is_2_num_define_party',
                 'is_ge2_num_define_party',
                 'has_define_party',
                 'has_define_agreement',
                 'has_word_between',
                 'startCharClass',
                 'endCharClass',
                 'startChar',
                 'endChar',
                 'hr',
                 'contains_prep_phrase',
                 'has_person', 'has_location', 'has_organization', 'has_date'
    ]


    # pylint: disable=too-many-arguments
    def __init__(self,
                 file_id: Optional[str],
                 start: int,
                 end: int,
                 sent_text: str,
                 labels: List[str],
                 entities,
                 sechead: Optional[str] = None) \
                 -> None:
        self.file_id = None  # we never use this, so not use.
        self.start = start  # this differs from ent_start, which can be chopped
        self.end = end      # similar to above, ent_end
        self.bag_of_words = sent_text
        self.labels = labels
        self.entities = entities
        if sechead is None:
            self.sechead = ''
        else:
            self.sechead = sechead

    def get_val(self, attr_name):
        """Return value of the attribute"""
        attr_name = v1name_to_v1_2name(attr_name)
        return getattr(self, attr_name)

    def set_val(self, attr, val):
        """Set value of the attribute"""
        attr = v1name_to_v1_2name(attr)
        setattr(self, attr, val)

    def set_val_yesno(self, attr, val):
        """Convert val to boolean before set it"""
        attr = v1name_to_v1_2name(attr)
        setattr(self, attr, bool(val))

    def __str__(self):
        attr_val_list = []
        attr_val_list.append(('start', self.start))
        attr_val_list.append(('end', self.end))
        attr_val_list.append(('labels', self.labels))
        attr_val_list.append(('sechead', self.sechead))
        attr_val_list.append(('words', self.bag_of_words))
        return '(attrvec ' + ', '.join(['{}={}'.format(attrval[0], attrval[1]) for attrval in attr_val_list]) + ')'
