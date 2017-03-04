
import unicodedata

# Java has Character.getType(), Python has a different mechanism for similar information.
# They don't exactly match, but took the most frequent mapping as the correct mapping.
# Both are 30 and unique.  # id missing 17, but has 30.
UNICODE_CATID_MAP = {'Cc': 15,
                     'Cf': 16,
                     'Cn': 0,    # Other, unknown
                     'Co': 18,
                     'Cs': 19,
                     'Ll': 2,
                     'Lm': 4,
                     'Lo': 5,
                     'Lt': 3,
                     'Lu': 1,
                     'Mc': 8,
                     'Me': 7,
                     'Mn': 6,
                     'Nd': 9,
                     'Nl': 10,
                     'No': 11,
                     'Pc': 23,
                     'Pd': 20,
                     'Pe': 22,
                     'Pf': 30,
                     'Pi': 29,
                     'Po': 24,
                     'Ps': 21,
                     'Sc': 26,
                     'Sk': 27,
                     'Sm': 25,
                     'So': 28,
                     'Zl': 13,
                     'Zp': 14,
                     'Zs': 12}

# not used
UNICODE_CAT_LIST = ['Cc', 'Cf', 'Cn', 'Co', 'Cs', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu',
                    'Mc', 'Me', 'Mn', 'Nd', 'Nl', 'No', 'Pc', 'Pd', 'Pe', 'Pf',
                    'Pi', 'Po', 'Ps', 'Sc', 'Sk', 'Sm', 'So', 'Zl', 'Zp', 'Zs']

def unicode_char_to_category_id(int_ch):
    return UNICODE_CATID_MAP.get(unicodedata.category(int_ch))

def unicode_char_to_category(int_ch):
    return unicodedata.category(int_ch)
