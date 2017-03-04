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
