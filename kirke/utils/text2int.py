import re
# pylint: disable=unused-import
from typing import Dict, List, Tuple, Union

# pylint: disable=invalid-name
numwords = {}
numeric_regex_st = ''
numeric_regex_st_with_b = ''
numeric_words_regex_st = ''

ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5,
                 'eighth':8, 'ninth':9, 'twelfth':12}
ordinal_endings = [('ieth', 'y'), ('th', '')]

# (\d{1,3}[,\.]?)+([,\.]\d{,2})?( *[tTbBmM]illion| *[tT]housand| *[TMB])?

# pylint: disable=global-statement
def setup_numwords():
    global numwords
    global numeric_regex_st
    global numeric_regex_st_with_b
    global numeric_words_regex_st
    units = ["zero", "one", "two", "three", "four", "five", "six",
             "seven", "eight", "nine", "ten", "eleven", "twelve",
             "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
             "eighteen", "nineteen"]

    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty",
            "seventy", "eighty", "ninety"]

    scales = ["hundred", "thousand", "million", "billion", "trillion",
              'quadrillion', 'quintillion', 'sexillion', 'septillion',
              'octillion', 'nonillion', 'decillion']

    numwords["and"] = (1, 0)
    numwords["point"] = (0.1, 0)
    for idx, word in enumerate(units):
        numwords[word] = (1, idx)
    for idx, word in enumerate(tens):
        numwords[word] = (1, idx * 10)
    for idx, word in enumerate(scales):
        numwords[word] = (10 ** (idx * 3 or 2), 0)

    # for numword in numwords:
    #    print('numword: {}'.format(numword))

    numwords['m'] = (1000000, 0)
    numwords['b'] = (1000000000, 0)
    numwords['t'] = (1000000000000, 0)

    numeric_words = list(units)
    numeric_words.extend([term for term in tens if term])
    numeric_words.extend(scales)
    numeric_words.extend([r'm\b', r'b\b', r't\b'])
    numeric_words.sort(key=len, reverse=True)

    numeric_and_words = list(numeric_words)
    numeric_and_words.extend(['and', 'point', r'm\b', r'b\b', r't\b'])
    numeric_and_words.sort(key=len, reverse=True)

    numeric_words_no_acronym = list(units)
    numeric_words_no_acronym.extend([term for term in tens if term])
    numeric_words_no_acronym.extend(scales)
    numeric_words_no_acronym.sort(key=len, reverse=True)

    # print('numeric_words: {}'.format(numeric_words))

    num_digit_regex_st = r'[-+]?[0-9,\.]*[0-9]+'

    # this is the best soution
    # pat = re.compile(r'(?=(bond index swaps|bond index))', flags=re.I)
    # (?<!\w)  negative look around, but failed on m-2
    # pylint: disable=unused-variable
    num_digit_regex_st_best = r'(((?<=\s)|(?<=^)|(?<=\()|(?<=\[))({}))\b'.format(num_digit_regex_st)


    num_digit_mbt_regex_st = r'[-+]?[0-9,\.]*[0-9]+([MBT]\b)?'
    # num_digit_mbt_regex_st = r'[-+]?[0-9,\.]*[0-9]+[\.,]?'

    # '\d[\d\,\.]*' is very permissive
    # works on 1,000,000.03
    numeric_regex_st = r'(({}|{})[\-\s]+)*({}|{})'.format(num_digit_mbt_regex_st,
                                                          '|'.join(numeric_and_words),
                                                          num_digit_mbt_regex_st,
                                                          '|'.join(numeric_words))

    # (?:^| |\()
    # pylint: disable=line-too-long
    #numeric_regex_st_with_b_old = r'((({}|{})[\-\s]+)*({}|{}))\b'.format(num_digit_mbt_regex_st,
    #                                                                     '|'.join(numeric_and_words),
    #                                                                     num_digit_mbt_regex_st,
    #                                                                     '|'.join(numeric_words))

    # pylint: disable=line-too-long
    numeric_regex_st_with_b = r'(((?<=\s)|(?<=^)|(?<=\()|(?<=\[))(({}|{})[\-\s]+)*({}|{}))\b'.format(num_digit_mbt_regex_st,
                                                                                                     '|'.join(numeric_and_words),
                                                                                                     num_digit_mbt_regex_st,
                                                                                                     '|'.join(numeric_words))

    # pylint: disable=line-too-long
    numeric_words_regex_st = r'\b(({})[\-\s]+)*({})\b'.format('|'.join(numeric_words_no_acronym + ['and']),
                                                              '|'.join(numeric_words_no_acronym))


setup_numwords()
# print('numeric_regex_st: {}'.format(numeric_regex_st))
# print('numeric_regex_st_with_b: {}'.format(numeric_regex_st_with_b))
# print('numeric_words_regex_st: {}'.format(numeric_words_regex_st))


NUM_REGEX = re.compile(numeric_regex_st_with_b, re.I)

NUM_IN_WORDS_REGEX = re.compile(numeric_words_regex_st, re.I)


NUM_WORD_HYPHEN_NUM_WORD = re.compile(r'({})\-({})'.format(numeric_words_regex_st,
                                                           numeric_words_regex_st))

def remove_num_words_join_hyphen(line: str) -> str:
    line = NUM_WORD_HYPHEN_NUM_WORD.sub(r'\1 \5', line)
    return line


def extract_numbers(line: str) -> List[Dict]:
    # don't want '-' to confuse words
    line = remove_num_words_join_hyphen(line)
    mat_list = list(NUM_REGEX.finditer(line))
    #for cx_mat in mat_list:
    #    print('\nnumber cx_mat group: {} {} [{}]'.format(cx_mat.start(),
    #                                                     cx_mat.end(),
    #                                                     cx_mat.group()))
    #    for gi, group in enumerate(cx_mat.groups(), 1):
    #         print("  cx_mat.group #{}: [{}]".format(gi, cx_mat.group(gi)))

    num_span_list = []  # type: List[Tuple[int, int, str]]
    result = []  # type: List[Dict]
    for mat in mat_list:
        numeric_span = (mat.start(), mat.end(), mat.group())
        # print('numeric_span: {}'.format(numeric_span))
        num_span_list.append(numeric_span)
        val = text2number(mat.group())
        adict = {'start': mat.start(),
                 'end': mat.end(),
                 'text': mat.group(),
                 'value': val}
        result.append(adict)
    return result


def extract_numbers_in_words(line: str) -> List[Dict]:
    # don't want '-' to confuse words
    line = remove_num_words_join_hyphen(line)
    mat_list = list(NUM_IN_WORDS_REGEX.finditer(line))
    # print('mat_listxxxx: {}'.format(mat_list))
    # for cx_mat in mat_list:
        # print('\nnumber cx_mat group: {} {} [{}]'.format(cx_mat.start(),
        #                                                  cx_mat.end(),
        #                                                  cx_mat.group()))
        # for gi, group in enumerate(cx_mat.groups(), 1):
        #    print("  cx_mat.group #{}: [{}]".format(gi, cx_mat.group(gi)))

    num_span_list = []  # type: List[Tuple[int, int, str]]
    result = []  # type: List[Dict]
    for mat in mat_list:
        numeric_span = (mat.start(), mat.end(), mat.group())
        # print('numeric_span: {}'.format(numeric_span))
        num_span_list.append(numeric_span)
        val = text2number(mat.group())
        adict = {'start': mat.start(),
                 'end': mat.end(),
                 'text': mat.group(),
                 'value': val}
        result.append(adict)
    return result


def extract_number(line: str) -> Dict:
    line = remove_num_words_join_hyphen(line)
    mat = NUM_REGEX.search(line)
    if mat:
        # numeric_span = (mat.start(), mat.end(), mat.group())
        # print('numeric_span: {}'.format(numeric_span))
        val = text2number(mat.group())
        adict = {'start': mat.start(),
                 'end': mat.end(),
                 'text': mat.group(),
                 'value': val}
        return adict
    return {}


def is_float(line: str) -> bool:
    try:
        float(line)
        return True
    except ValueError:
        return False

COMMA_1OR2_DIGIT_REGEX = re.compile(r'(,\d\d|,\d)$')

# pylint: disable=too-many-return-statements
def normalize_comma_period(line: str) -> str:
    comma_offset = line.find(',')
    period_offset = line.find('.')

    # no comma found, assume the period is correct
    # 1.401 = US 1.401
    if comma_offset == -1:
        return line

    # so there is comma in the word
    # either 1.401,01 (period-comma)
    # 1,401.01 (comma-period)
    # 1,401  (comma)
    # 14,01  (comma)

    # this is 1,401 (comma only)
    if period_offset == -1:
        # only for 14,02 or 14,2
        if COMMA_1OR2_DIGIT_REGEX.search(line):
            # ran into 10,000,00 but really mean one million
            # since in this domain, decimals are rare, ignore incorrect '.'
            # 10.000.000 causes exception to be thrown
            if len(list(re.finditer(r',', line))) >= 2:
                return line.replace(',', '')
            return line.replace(',', '.')
        # for 143,534 => 143534
        return line.replace(',', '')

    # now, both period_offset != -1 and comma_offset != -1
    if period_offset < comma_offset:
        # european style, 1.234,456
        line = line.replace('.', '')
        line = line.replace(',', '.')
        return line

    if comma_offset < period_offset:
        line = line.replace(',', '')
        return line

    return line

NUM_ALPHA_REGEX = re.compile(r'^(\d[\d\,\.]*)([a-zA-Z]+)$')

def split_num_alpha(tokens: List[str]) -> List[str]:
    result = []  # type: List[str]
    for tok in tokens:
        mat = NUM_ALPHA_REGEX.search(tok)
        if mat:
            result.append(mat.group(1))
            result.append(mat.group(2))
        else:
            result.append(tok)
    return result


# https://stackoverflow.com/questions/493174/is-there-a-way-to-convert-number-words-to-integers
def text2number(textnum: str) -> Union[int, float]:
    textnum = textnum.strip()
    if not textnum:
        return -1
    is_negative = False
    if textnum[0] == '+':
        textnum = textnum[1:]
    elif textnum[0] == '-':
        textnum = textnum[1:]
        is_negative = True

    textnum = textnum.lower()
    current = 0  # type: float
    result = 0  # type: float
    tokens = [tok for tok in re.split(r"[\-\s]+", textnum) if tok]

    tokens = split_num_alpha(tokens)
    # point_scale = 1  # type: Union[float, int]
    has_seen_point = False
    point_scale = 0.1
    for word in tokens:
        word = normalize_comma_period(word)

        if word.isdigit():
            scale, increment = 1, int(word)  # type: Tuple[int, Union[int, float]]
        elif is_float(word):
            scale, increment = 1, float(word)
        elif word == 'point':
            has_seen_point = True
            result += current
            current = 0
            continue
        elif word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                raise Exception("Illegal word: " + word)
                # return -1

            scale, increment = numwords[word]

        # print('  scale = {}, increment = {}, has_seen_point = {}'.format(scale,
        #                                                                  increment,
        #                                                                  has_seen_point))
        # print('  current = {}, result = {}'.format(current, result))

        if current < 1:  # current is never < 1 for non-float points
            # we don't want to reset 'current' in this case
            pass
        elif scale > 1:
            current = max(1, current)

        if not has_seen_point:
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
        else:
            result += point_scale * increment
            point_scale /= 10.0


        # print('  after:')
        # print('    scale = {}, increment = {}'.format(scale, increment))
        # print('    current = {}, result = {}'.format(current, result))
        # print()

    if is_negative:
        return -(result + current)
    return result + current
