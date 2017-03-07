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

def init_eb_attrs():
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

(EB_ATTR_LIST, EB_ATTR_TYPE_MAP, EB_ATTR_IDX_MAP,
 BINARY_INDICES, NUMERIC_INDICES, CATEGORICAL_INDICES) = init_eb_attrs()

DEFAULT_IS_TO_VALIDATE = True

# replacing sent2attrs
class EbAttrVec:

    def __init__(self, file_id, start, end):
        self.file_id = file_id
        self.start = start
        self.end = end
        self.attr_val_map = {}
        # fv.set_val('bag_of_words', ebsent.get_tokens_text())
        # intentionally not using ebsent.get_text()

    def to_list(self):
        if DEFAULT_IS_TO_VALIDATE:
            self.validate_types()
        # result = [self.start, self.end]
        result = [self.attr_val_map.get(attr) for attr in EB_ATTR_LIST]
        # result.extend([','.join(self.labels), entities_to_st(self.entities), self.text])
        return result

    # def get_val(self, attr_name):
    #    return self.vec[EB_ATTR_IDX_MAP.get(attr_name)]

    def set_val(self, attr, val):
        self.attr_val_map[attr] = val

    def set_val_yesno(self, attr, val):
        self.attr_val_map[attr] = bool(val)


    def validate_types(self):
        for attr, atype in EB_ATTR_TYPE_MAP.items():
            val = self.attr_val_map.get(attr)
            if val is None:
                raise ValueError('failed in validate_type(), attr(%s) is None', attr)
            if atype == 'bool':
                if not isinstance(val, bool):
                    raise ValueError('failed in validate_type(), bool != %s %s',
                                     type(val), str(val))
            elif atype == 'numeric':
                if not isinstance(val, (int, float)):
                    raise ValueError('failed in validate_type(), numeric != %s %s',
                                     type(val), str(val))
            elif atype == 'categorical':
                pass
            else:
                raise ValueError('failed in validate_type(), unknown type: %s', atype)
