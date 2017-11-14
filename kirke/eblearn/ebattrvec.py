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
    'le-3-word:bool', 'le-5-word:bool',
    'le-10-word:bool', 'ge-05-lt-10-word:bool',
    'ge-10-lt-20-word:bool', 'ge-20-lt-30-word:bool',
    'ge-30-lt-40-word:bool', 'ge-40-word:bool']

# lemma as text/bag-of-words give 3% worse result
# now use corenlp token instead
#MISC_ATTR_TYPE_ST_LIST = [
#    'bag-of-words:string', 'labels:string-list',
#    'entities:other']

EXTRA_PARTY_ATTR_TYPE_ST_LIST = [
    'is-1-num-define-party:bool',
    'is-2-num-define-party:bool',
    'is-ge2-num-define-party:bool',
    'has-define-party:bool',
    'has-define-agreement:bool',
    'has-word-between:bool']

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


# replacing sent2attrs
class EbAttrVec:
    """Store values representing a sentence.  It is the major part
       of ebantdoc"""

    # Intentionally not making this a slot because we want flexibility in defining
    # new features.
    # __slots__ = ['file_id', 'start', 'end', 'bag_of_words', 'labels', 'entities']

    # pylint: disable=too-many-arguments
    def __init__(self, file_id, start, end, sent_text, labels, entities, sechead):
        self.file_id = file_id
        self.start = start  # this differs from ent_start, which can be chopped
        self.end = end      # similar to above, ent_end
        self.bag_of_words = sent_text
        self.labels = labels
        self.entities = entities
        self.sechead = sechead

    def get_val(self, attr_name):
        """Return value of the attribute"""
        return getattr(self, attr_name)

    def set_val(self, attr, val):
        """Set value of the attribute"""
        setattr(self, attr, val)

    def set_val_yesno(self, attr, val):
        """Convert val to boolean before set it"""
        setattr(self, attr, bool(val))

    def __str__(self):
        attr_val_list = []
        attr_val_list.append(('file_id', self.file_id))
        attr_val_list.append(('start', self.start))
        attr_val_list.append(('end', self.end))
        attr_val_list.append(('labels', self.labels))
        attr_val_list.append(('sechead', self.sechead))
        attr_val_list.append(('words', self.bag_of_words))
        return '(attrvec ' + ', '.join(['{}={}'.format(attrval[0], attrval[1]) for attrval in attr_val_list]) + ')'
