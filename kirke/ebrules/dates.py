import re
import logging
from kirke.utils import txtreader

def extract_party_line(paras_attr_list):
    lines = []
    offset = 0
    start_end_list = []
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
    lines = []
    offset = 0
    start_end_list = []
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
DATE_AS_OF_PAT = re.compile(r"as of (.*?) by\b", re.IGNORECASE)
DIGIT_PAT = re.compile(r'\d')

def extract_dates_from_party_line(line):
    result = []
    for mat in DATE_AS_OF_PAT.finditer(line):
        if DIGIT_PAT.search(mat.group(1)):   # hand written date
            char40_before = line[max(mat.start()-40, 0):mat.start()]
            char40_after = line[mat.end():mat.end()+40]
            print('char40_before = [{}]'.format(char40_before))
            print('char40_after = [{}]'.format(char40_after))        
            if (EFFECTIVE_PAT.search(char40_before) or
                EFFECTIVE_PAT.search(char40_after)):            
                result.append((mat.start(1), mat.end(1), mat.group(1), 'effectivedate_auto'))
            else:
                result.append((mat.start(1), mat.end(1), mat.group(1), 'date'))
    return result

MONTH_LIST = ['January', 'February', 'March', 'April', 'May',
              'June', 'July', 'August', 'September', 'October',
              'November', 'December']
MONTH_ABBR_LIST = ['Jan.?', 'Feb.?', 'Mar.?', 'Apr.?',
                   'Jun.?', 'Jul.?', 'Sep.?', 'Sept.?', 'Oct.?',
                   'Nov.?', 'Dec.?']
ALL_MONTH_LIST = MONTH_LIST + MONTH_ABBR_LIST

ALL_MONTH_PAT = '|'.join(ALL_MONTH_LIST)

DATE_PAT_ST = '(' + ALL_MONTH_PAT + r')\s+\d{1,2},?\s+\d{2,4}'
# DATE_PAT_ST = '(' + ALL_MONTH_PAT + r')'
print('DATE_PAT_ST = "{}"'.format(DATE_PAT_ST))
                         
DATE_PAT1 = re.compile(DATE_PAT_ST, re.IGNORECASE)

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
        if (EFFECTIVE_PAT.search(char40_before) or
            EFFECTIVE_PAT.search(char40_after)):
            result.append((mat.start(), mat.end(), mat.group(), 'effectivedate_auto'))
        else:
            result.append((mat.start(), mat.end(), mat.group(),  'date'))
    return result


def extract_dates(filepath):
    """Return list of parties (lists of (start, inclusive-end) offsets)."""

    doc_text = txtreader.loads(filepath)
    # Find the party line in the file
    party_line_ox = None
    before_lines = []
    offset = 0
    with open(filepath) as f:
        for i, line in enumerate(f):
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
    for line_start, line_end, xline in before_lines:
        found_dates = extract_dates_v2(xline, line_start, doc_text='')
        if found_dates:
            before_dates.extend(found_dates)
    print('before_dates: {}'.format(before_dates))

    if not before_dates and not party_line_ox:
        return None

    # Extract parties and return their offsets
    party_start, party_end, party_line = party_line_ox
    dates = extract_dates_from_party_line(party_line)
    return before_dates + dates


def extract_offsets(paras_attr_list, paras_text):
    """Return list of parties (lists of (start, inclusive-end) offsets)."""

    logging.info('extract_offsets: len(paras_text) = {}'.format(len(paras_text)))
    # Grab lines from the file
    before_lines, start_end_partyline = extract_before_and_party_line(paras_attr_list)

    partyline_dates = []
    if start_end_partyline:
        partyline_start, partyline_end, partyline = start_end_partyline
        # print("partyline ({}, {})".format(partyline_start, partyline_end))
        # print("[{}]".format(partyline))

        # Extract parties and return their offsets
        partyline_dates = extract_dates_from_party_line(partyline)
        out_list = []
        # logging.info("partyline dates: {}".format(partyline_dates))
        if partyline_dates:
            adjusted_dates = []
            for date_ox in partyline_dates:
                start, end, date_st, date_type = date_ox
                adjusted_dates.append((partyline_start + start, partyline_start + end, date_type))
            partyline_dates = adjusted_dates
    #for date_ox in partyline_dates:
    #    date_start, date_end, date_st = date_ox
    #    out_list.append((partyline_start + date_start, partyline_start + date_end, date_st))

    before_dates = [] 
    for line_start, line_end, xline in before_lines:
        found_dates = extract_dates_v2(xline, line_start, doc_text=paras_text)
        if found_dates:
            for date_ox in found_dates:
                start, end, date_st, date_type = date_ox
                before_dates.append((line_start + start, line_start + end, date_type))
    # print('before_dates: {}'.format(before_dates))

    if not before_dates and not partyline_dates:
        return None

    # Extract parties and return their offsets
    out_list = partyline_dates + before_dates
        
    # logging.info("out_list: {}".format(out_list))
    return out_list


class DateAnnotator:

    def __init__(self, provision):
        self.provision = 'date'

    def extract_provision_offsets(self, paras_with_attrs, paras_text):
        return extract_offsets(paras_with_attrs, paras_text)
        
