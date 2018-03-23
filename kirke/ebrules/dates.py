import calendar
import datetime
import re
from typing import Dict, Optional

from dateutil import parser

from kirke.utils import mathutils


def extract_party_line(paras_attr_list):
    offset = 0
    for i, (line_st, para_attrs) in enumerate(paras_attr_list):
        # attrs_st = '|'.join([str(attr) for attr in para_attrs])
        # print('\t'.join([attrs_st, '[{}]'.format(line_st)]), file=fout1)
        line_st_len = len(line_st)

        if 'party_line' in para_attrs:
            return offset, offset + line_st_len, line_st
        offset += line_st_len + 1

        # don't bother if party_line is too far from start of the doc
        if i > 1000:
            return None

    return None

def extract_before_and_party_line(paras_attr_list):
    offset = 0
    before_lines = []
    for i, (line_st, para_attrs) in enumerate(paras_attr_list):
        # attrs_st = '|'.join([str(attr) for attr in para_attrs])
        # print('\t'.join([attrs_st, '[{}]'.format(line_st)]), file=fout1)
        line_st_len = len(line_st)

        if 'party_line' in para_attrs:
            return before_lines, (offset, offset + line_st_len, line_st)
        else:
            before_lines.append((offset, offset + line_st_len, line_st))
        offset += line_st_len + 1

        # don't bother if party_line is too far from start of the doc
        if i > 1000:
            return before_lines[:100], None

    return before_lines[:100], None

# DATE_AS_OF_PAT = re.compile(r"as of (.*\d.*) by\b", re.IGNORECASE)
# bad, DATE_AS_OF_PAT = re.compile(r"as of ((?!by).)* by\b", re.IGNORECASE)
# DATE_AS_OF_PAT = re.compile(r"as of (\S+\s+){1,2,3,4}(by\b|[\(\"•]+effective)", re.IGNORECASE)
DATE_AS_OF_PAT = re.compile(r"as of (.*)", re.IGNORECASE)
DIGIT_PAT = re.compile(r'[oOl\d]')
BY_PAT = re.compile(r'\s+(by\b|\(|[, ]*between)', re.IGNORECASE)
EFFECTIVE_FOR_AS_IF_PAT = re.compile(r'\s*[\(\“\"]+effective', re.IGNORECASE)
# 'the effective|distribution|lease date'
SET_FORTH_PAT = re.compile(r'\b(the date set forth in section \S+ of the summary|the (\S+) date)\b',
                           re.IGNORECASE)

# handd-written 2012, it can be up to 3 word in hand-writing
DATE_MADE_ON_PAT = re.compile(r'\bmade on ((\S+|\S+\s+\S+|\S+\s+\S+\s+\S+) \d{4})\b', re.IGNORECASE)

# pylint: disable=too-many-branches, too-many-statements
def extract_dates_from_party_line(line):
    result = []
    # very special case, for handling handwriting
    for mat in DATE_AS_OF_PAT.finditer(line):
        maybe_date = mat.group(1)
        by_mat = BY_PAT.search(maybe_date)
        date_start, date_end = -1, -1
        set_forth_mat = SET_FORTH_PAT.search(maybe_date)

        # print("maybe_date: [{}]".format(maybe_date))
        #if len(maybe_date.split()) == 1:  # "as of May'
        #    continue
        if by_mat:  # hand written date
            maybe_date_st = line[mat.start(1):mat.start(1)+by_mat.start()]
            # print("maybe_date_st1: [{}], len= {}".format(maybe_date_st, len(maybe_date_st)))
            if len(maybe_date_st) < 20 or (len(maybe_date_st) < 35 and
                                           'day' in maybe_date_st.lower()):  # signature
                date_start = mat.start(1)
                date_end = mat.start(1) + by_mat.start()
        if not by_mat and set_forth_mat:
            maybe_date_st = line[mat.start(1):mat.start(1)+set_forth_mat.end()]
            date_start = mat.start(1)
            date_end = mat.start(1) + set_forth_mat.end()
        # effective_mat = EFFECTIVE_FOR_AS_IF_PAT.search(maybe_date)
        #if effective_mat:  # hand written date
        #    # maybe_date_st = line[mat.start(1):mat.start(1)+effective_mat.start()]
        #    # print("maybe_date_st2: [{}], len= {}".format(maybe_date_st, len(maybe_date_st)))
        #    if len(maybe_date_st) < 20 or (len(maybe_date_st) < 35 and
        #                                   'day' in maybe_date_st.lower()):  # signature
        #        date_start = mat.start(1)
        #        date_end = mat.start(1)+by_mat.start()
        if date_start != -1:
            char40_before = line[max(mat.start()-40, 0):mat.start()]
            char40_after = line[mat.end():mat.end()+40]
            if EFFECTIVE_PAT.search(char40_before) or \
               EFFECTIVE_PAT.search(char40_after):
                result.append((date_start, date_end, maybe_date_st, 'effectivedate'))
            else:
                result.append((date_start, date_end, maybe_date_st, 'date'))

    for mat in DATE_MADE_ON_PAT.finditer(line):
        maybe_date_st = mat.group(1)
        if maybe_date_st:
            char40_before = line[max(mat.start()-40, 0):mat.start()]
            char40_after = line[mat.end():mat.end()+40]
            if EFFECTIVE_PAT.search(char40_before) or \
               EFFECTIVE_PAT.search(char40_after):
                result.append((mat.start(1), mat.end(1), maybe_date_st, 'effectivedate'))
            else:
                result.append((mat.start(1), mat.end(1), maybe_date_st, 'date'))

    # print("as_if result date: {}".format(result))

    for mat in DATE_PAT1.finditer(line):
        # print("date_pat1: {}".format(mat.group()))
        char40_before = line[max(mat.start()-40, 0):mat.start()]
        char40_after = line[mat.end():mat.end()+40]
        if EFFECTIVE_PAT.search(char40_before) or \
           EFFECTIVE_PAT.search(char40_after):
            result.append((mat.start(), mat.end(), mat.group(), 'effectivedate'))
        else:
            result.append((mat.start(), mat.end(), mat.group(), 'date'))

    for mat in DATE_PAT3.finditer(line):
        # print("date_pat3: {}".format(mat.group()))
        char40_before = line[max(mat.start()-40, 0):mat.start()]
        char40_after = line[mat.end():mat.end()+40]
        if EFFECTIVE_PAT.search(char40_before) or \
           EFFECTIVE_PAT.search(char40_after):
            result.append((mat.start(), mat.end(), mat.group(), 'effectivedate'))
        else:
            result.append((mat.start(), mat.end(), mat.group(), 'date'))

    for mat in DATE_PAT2.finditer(line):
        # print("date_pat2: {}".format(mat.group()))
        char40_before = line[max(mat.start()-40, 0):mat.start()]
        char40_after = line[mat.end():mat.end()+40]
        if EFFECTIVE_PAT.search(char40_before) or \
           EFFECTIVE_PAT.search(char40_after):
            result.append((mat.start(), mat.end(), mat.group(), 'effectivedate'))
        else:
            result.append((mat.start(), mat.end(), mat.group(), 'date'))

    for mat in DATE_PAT4.finditer(line):
        # print("date_pat4: {}".format(mat.group()))
        char40_before = line[max(mat.start()-40, 0):mat.start()]
        char40_after = line[mat.end():mat.end()+40]
        if EFFECTIVE_PAT.search(char40_before) or \
           EFFECTIVE_PAT.search(char40_after):
            result.append((mat.start(), mat.end(), mat.group(), 'effectivedate'))
        else:
            result.append((mat.start(), mat.end(), mat.group(), 'date'))

    result = prefer_effectivedate_over_date(result)
    return mathutils.remove_subsumed(result)

def prefer_effectivedate_over_date(alist):
    start_end_tuple_map = {}
    for elt in alist:
        old_elt = start_end_tuple_map.get((elt[0], elt[1]), [])
        if elt[3] == 'effectivedate_auto':
            # effectivedate_auto overrides others
            start_end_tuple_map[(elt[0], elt[1])] = elt
        elif not old_elt:
            start_end_tuple_map[(elt[0], elt[1])] = elt
    return start_end_tuple_map.values()

def extract_std_dates(line):
    """Extract standard-format dates from a given line."""
    dates = [(mat.start(), mat.end())
             for pat in (DATE_PAT1, DATE_PAT2, DATE_PAT3, DATE_PAT4)
             for mat in pat.finditer(line)]
    pairs = mathutils.remove_subsumed(dates)
    return sorted(pairs)


MONTH_LIST = ['January', 'February', 'March', 'April', 'May',
              'June', 'July', 'August', 'September', 'October',
              'November', 'December',
              # for OCR misspelling?
              'M ay']
MONTH_ABBR_LIST = [r'Jan\.?', r'Feb\.?', r'Mar\.?', r'Apr\.?',
                   r'Jun\.?', r'Jul\.?', r'Sep\.?', r'Sept\.?', r'Oct\.?',
                   r'Nov\.?', r'Dec\.?']
ALL_MONTH_LIST = MONTH_LIST + MONTH_ABBR_LIST

ALL_MONTH_PAT = '|'.join(ALL_MONTH_LIST)

DATE_PAT1_ST = '(' + ALL_MONTH_PAT + r')\s*[oOl\d]{1,2}(\S\S)?[,\s]*[oOl\d]{4}'
DATE_PAT1_1_ST = '(' + ALL_MONTH_PAT + r'|_+' + r')\s*(_+|\[[_•\s]*\])[,\s]*[oOl\d]{4}'
# only month year, 'june 2010'
DATE_PAT1_2_ST = '(' + ALL_MONTH_PAT + r'|(_+|\[[_•\s]*\])' + r')[,\s]*[oOl\d]{4}'

# DATE_PAT_ST = '(' + ALL_MONTH_PAT + r')'
# print('DATE_PAT_ST = "{}"'.format(DATE_PAT1_ST))

DATE_PAT1 = re.compile(r'(' + DATE_PAT1_ST + r'|' + DATE_PAT1_1_ST  +
                       r'|' + DATE_PAT1_2_ST + r')\b', re.IGNORECASE)


DATE_PAT2_ST = r'[oOl\d]{1,2}\s*(' + ALL_MONTH_PAT + r')[,\s]+[oOl\d]{4}'
# '__ Nov 2010'
DATE_PAT2_1_ST = r'_+\s*(' + ALL_MONTH_PAT + r')[,\s]+[oOl\d]{4}'
DATE_PAT2 = re.compile(r'(' + DATE_PAT2_ST + '|' + DATE_PAT2_1_ST + r')\b', re.IGNORECASE)


# 'st|nd|rd' can have ocr errors, so up to 3 chars
DATE_PAT3_ST = r'((the|this)\s*)?[oOl\d]{1,2}(\s*\S\S)?\s*((day )?(of|o f))?\s*(' + \
               ALL_MONTH_PAT + r')[,\s]+[oOl\d]{4}'
DATE_PAT3_1_ST = r'((the|this)\s*)?\S+\s+(day (of|o f))\s+\S+[,\s]+[oOl\d]{4}'
# date without year, "this x 21st day of x december, 2009"
DATE_PAT3_2_ST = r'((the|this)\s*)+\S+\s+((day )?(of|o f))\s+\S*(' + ALL_MONTH_PAT + \
                 r')([,\s]+[oOl\d]{4})?'
# 'this day of 2010
DATE_PAT3_3_ST = r'((the|this)\s+((day )?(of|o f))\s+[oOl\d]{4})'
DATE_PAT3 = re.compile(r'(' + DATE_PAT3_ST + r'|' + DATE_PAT3_1_ST + r'|' + DATE_PAT3_2_ST + r'|' +
                       DATE_PAT3_3_ST + r')\b', re.IGNORECASE)

# pylint: disable=line-too-long
DATE_PAT4_ST = r'\b([oOl\d]{1,2}[\-\/][oOl\d]{1,2}[\-\/][oOl\d]{2,4}|[oOl\d]{4}[\-\/][oOl\d]{1,2}[\-\/][oOl\d]{1,2})\b'
DATE_PAT4 = re.compile(DATE_PAT4_ST, re.IGNORECASE)

EFFECTIVE_PAT = re.compile(r'effective', re.IGNORECASE)

def extract_dates_v2(line, line_start, doc_text=''):
    result = []
    for mat in DATE_PAT1.finditer(line):
        if doc_text:
            char40_before = doc_text[max(line_start + mat.start() - 40, 0):line_start + mat.start()]
            char40_after = doc_text[line_start + mat.end():line_start + mat.end() + 40]
        else:
            char40_before = line[max(mat.start()-40, 0):mat.start()]
            char40_after = line[mat.end():mat.end()+40]
        if EFFECTIVE_PAT.search(char40_before) or \
           EFFECTIVE_PAT.search(char40_after):
            result.append((mat.start(), mat.end(), mat.group(), 'effectivedate'))
        else:
            result.append((mat.start(), mat.end(), mat.group(), 'date'))

    for mat in DATE_PAT3.finditer(line):
        if doc_text:
            char40_before = doc_text[max(line_start + mat.start() - 40, 0):line_start + mat.start()]
            char40_after = doc_text[line_start + mat.end():line_start + mat.end() + 40]
        else:
            char40_before = line[max(mat.start()-40, 0):mat.start()]
            char40_after = line[mat.end():mat.end()+40]
        if EFFECTIVE_PAT.search(char40_before) or \
           EFFECTIVE_PAT.search(char40_after):
            result.append((mat.start(), mat.end(), mat.group(), 'effectivedate'))
        else:
            result.append((mat.start(), mat.end(), mat.group(), 'date'))

    for mat in DATE_PAT2.finditer(line):
        if doc_text:
            char40_before = doc_text[max(line_start + mat.start() - 40, 0):line_start + mat.start()]
            char40_after = doc_text[line_start + mat.end():line_start + mat.end() + 40]
        else:
            char40_before = line[max(mat.start()-40, 0):mat.start()]
            char40_after = line[mat.end():mat.end()+40]
        if EFFECTIVE_PAT.search(char40_before) or \
           EFFECTIVE_PAT.search(char40_after):
            result.append((mat.start(), mat.end(), mat.group(), 'effectivedate'))
        else:
            result.append((mat.start(), mat.end(), mat.group(), 'date'))

    for mat in DATE_PAT4.finditer(line):
        if doc_text:
            char40_before = doc_text[max(line_start + mat.start() - 40, 0):line_start + mat.start()]
            char40_after = doc_text[line_start + mat.end():line_start + mat.end() + 40]
        else:
            char40_before = line[max(mat.start()-40, 0):mat.start()]
            char40_after = line[mat.end():mat.end()+40]
        if EFFECTIVE_PAT.search(char40_before) or \
           EFFECTIVE_PAT.search(char40_after):
            result.append((mat.start(), mat.end(), mat.group(), 'effectivedate'))
        else:
            result.append((mat.start(), mat.end(), mat.group(), 'date'))

    # remove duplicates
    out_list2 = mathutils.remove_subsumed(result)

    return out_list2


# pylint: disable=too-many-locals
def extract_dates(filepath):
    """Return list of parties (lists of (start, inclusive-end) offsets)."""

    # Find the party line in the file
    party_line_ox = None
    before_lines = []
    offset = 0
    with open(filepath) as fin:
        for line in fin:
            tags = line.split('\t')[0].split('|')
            if 'party_line' in tags:
                after_first_bracket = ''.join(line.split('[')[1:])
                between_brackets = ''.join(after_first_bracket.split(']')[:-1])
                party_line_ox = offset, offset+len(between_brackets), between_brackets
                break
            after_first_bracket = ''.join(line.split('[')[1:])
            between_brackets = ''.join(after_first_bracket.split(']')[:-1])
            line2 = offset, offset+len(between_brackets), between_brackets
            before_lines.append(line2)
            offset = offset + len(between_brackets) + 2
        if not party_line_ox and len(before_lines) > 100:
            before_lines = before_lines[:100]

    # print("party line: [{}]".format(party_line_ox))
    # print('len(before_lines) = {}'.format(len(before_lines)))

    before_dates = []
    for line_start, unused_line_end, xline in before_lines:
        found_dates = extract_dates_v2(xline, line_start, doc_text='')
        if found_dates:
            before_dates.extend(found_dates)
    # print('before_dates: {}'.format(before_dates))

    if not before_dates and not party_line_ox:
        return None

    # Extract parties and return their offsets
    unused_party_start, unused_party_end, party_line = party_line_ox
    dates = extract_dates_from_party_line(party_line)
    return before_dates + dates


def extract_offsets(paras_attr_list, paras_text):
    """Return list of parties (lists of (start, inclusive-end) offsets)."""

    # logging.info('extract_offsets: len(paras_text) = {}'.format(len(paras_text)))
    # Grab lines from the file
    before_lines, start_end_partyline = extract_before_and_party_line(paras_attr_list)

    partyline_dates = []
    if start_end_partyline:
        partyline_start, unused_partyline_end, partyline = start_end_partyline
        # print("partyline ({}, {})".format(partyline_start, partyline_end))
        # print("[{}]".format(partyline))

        # Extract parties and return their offsets
        partyline_dates = extract_dates_from_party_line(partyline)
        # logging.info("partyline dates: {}".format(partyline_dates))
        if partyline_dates:
            adjusted_dates = []
            for date_ox in partyline_dates:
                start, end, unused_date_st, date_type = date_ox
                adjusted_dates.append((partyline_start + start, partyline_start + end, date_type))
            partyline_dates = adjusted_dates
    # print('partyline_dates: {}'.format(partyline_dates))

    before_dates = []
    for line_start, unused_line_end, xline in before_lines:
        found_dates = extract_dates_v2(xline, line_start, doc_text=paras_text)
        if found_dates:
            for date_ox in found_dates:
                start, end, unused_date_st, date_type = date_ox
                before_dates.append((line_start + start, line_start + end, date_type))
    # print('before_dates: {}'.format(before_dates))
    # x1 = before_dates[0]
    # print("paras_text: [{}]".format(paras_text[x1[0]:x1[1]]))

    if not before_dates and not partyline_dates:
        return None

    # we want first effective date and date, no more
    out_list = []
    # if effective date is mentioned in before_dates, that's effective date for the doc.
    # in party line, the effective date can be effective date for master doc
    xx_effective_dates = [date_ox for date_ox in before_dates
                          if date_ox[2] == 'effectivedate']
    if not xx_effective_dates:
        xx_effective_dates = [date_ox for date_ox in partyline_dates
                              if date_ox[2] == 'effectivedate']
    if xx_effective_dates:
        out_list.append(xx_effective_dates[0])

    xx_dates = [date_ox for date_ox in partyline_dates if date_ox[2] == 'date']
    if not xx_dates:
        xx_dates = [date_ox for date_ox in before_dates if date_ox[2] == 'date']
    if xx_dates:
        out_list.append(xx_dates[0])

    # logging.info("out_list: {}".format(out_list))
    return out_list


# pylint: disable=too-few-public-methods
class DateAnnotator:

    def __init__(self, provision):
        self.provision = provision

    # pylint: disable=no-self-use
    def extract_provision_offsets(self, paras_with_attrs, paras_text):
        return extract_offsets(paras_with_attrs, paras_text)


# pylint: disable=too-few-public-methods
class NoDefaultDate(object):

    # pylint: disable=no-self-use
    def replace(self, **fields) -> Dict:
        return fields


def get_last_day_of_month(year: int, month: int) -> int:
    """Returns the number of days in a month."""
    _, num_day = calendar.monthrange(year, month)
    return num_day


# pylint: disable=too-few-public-methods
class DateNormalizer:

    def __init__(self) -> None:
        self.label = 'date_norm'

    # if using fuzzy-with_tokens
    # Tuple[datetime.datetime, Tuple]
    # pylint: disable=no-self-use
    def parse_date(self, line: str) -> Optional[Dict]:
        orig_line = line.replace('\n', '|')
        line = re.sub(r'first', '1st', line)
        # fixing OCR errors for "L" for "1" or "O" for '0"
        if 'l' in line:
            line = re.sub(r'(\d)l', r'\g<1>1', line)
            line = re.sub(r'l(\d)', r'1\g<1>', line)
        if 'o' in line:
            line = re.sub(r'(\d)[oO]', r'\g<1>0', line)
            line = re.sub(r'[oO](\d)', r'0\g<1>', line)

        try:
            # set dayfirst=True for UK dates, revisit later
            norm = parser.parse(line, fuzzy=True, default=NoDefaultDate())
        except ValueError:
            print("x2523, failed to parse [{}] as a date".format(text))
            return None
        except:
            print("x2524, failed to parse2 [{}] as a date".format(text))
            return None

        if re.search('end of', orig_line, re.I) and \
           norm.get('month') and norm.get('year') and \
           not norm.get('day'):
            norm['day'] = get_last_day_of_month(norm.get('year'),
                                                norm.get('month'))

        year_st, month_st, day_st = 'XXXX', 'XX', 'XX'
        year_val = norm.get('year')
        month_val = norm.get('month')
        day_val = norm.get('day')
        if year_val:
            year_st = '{:04d}'.format(year_val)
        if month_val:
            month_st = '{:02d}'.format(month_val)
        if day_val:
            day_st = '{:02d}'.format(day_val)
        norm_st = {"norm": 'date:{}-{}-{}'.format(year_st, month_st, day_st)}
        return norm_st


    def enrich(self, sample: Dict) -> None:
        text = sample['text']
        date_dict = self.parse_date(text)

        if not date_dict:
            sample['reject'] = True
            return

        sample.update(date_dict)
