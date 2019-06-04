import logging
import re
from typing import Dict, List, Match, Optional, Pattern, Tuple

from kirke.utils import ebantdoc4, ebsentutils, strutils

from kirke.utils import text2int

from kirke.utils.text2int import remove_num_words_join_hyphen


# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# pylint: disable=line-too-long
# (?:^| |\()
# want to avoid "rst", which is "rs" + "t" where "t" is just trillion
CURRENCY_PAT_ST = r'((\bUSD\b|\bEUR\b|\bGBP\b|\bCNY\b|\bJPY\b|\bINR\b|\bRupees?\b|\bRs\b\.?)|[\$€£円¥₹]) *({})'.format(text2int.numeric_regex_st)
CURRENCY_PAT = re.compile(CURRENCY_PAT_ST, re.I)


CURRENCY_SYMBOL_PAT_ST = r'((USD|EUR|GBP|CNY|JPY|INR|Rs|[dD]ollars?|u\.\s*s\.\s*dollars?|' \
                         r'[eE]uros?|[pP]ounds?|[yY]uans?|[yY]ens?|[rR]upees?)\b|[\$€£円¥₹])'
CURRENCY_SYMBOL_PAT = re.compile(CURRENCY_SYMBOL_PAT_ST)

# print('\nCURRENCY_PAT_ST')
# print(CURRENCY_PAT_ST)

# pylint: disable=line-too-long
# NUMBER_PAT = re.compile(r'(^|\s)\(?(-?({}))\)?[,\.:;]?(\s|$)'.format(text2int.numeric_regex_st), re.I)
# NUMBER_PAT = re.compile(r'((^|\s)\(?(-?([0-9]+([,\.][0-9]{3})*[,\.]?[0-9]*|[,\.][0-9]+))\)?[,\.:;]?(\s|$))' +
#                         r'|({})'.format(text2int.numeric_words_regex_st))

# standard floating point number
# https://www.regular-expressions.info/floatingpoint.html
# NUM_PAT_ST = '[-+]?[0-9]*\.?[0-9]+'
# (?:^| |\()
# NUM_PAT_ST = r'(([-+]?\b[0-9,\.]*[0-9]+)|' + \
#              r'\b({}))\b'.format(text2int.numeric_words_regex_st)

NUM_PAT_ST = r'(((?<=\s)|(?<=^)|(?<=\()|(?<=\[)|(?<=\<))({}))\b'.format(text2int.numeric_regex_st)

NUMBER_PAT = re.compile(NUM_PAT_ST, re.I)

# print('\nNUM_PAT_ST')
# print(NUM_PAT_ST)

# TODO
# WARNING, this is no longer used due to backtracking take took long
# a new version of extract_percents() is used.  Not this regex.
# TO_FIX
# pylint: disable=line-too-long
PERCENT_PAT_ST = r'({})\s*(%|percent)'.format(text2int.numeric_regex_st_with_b)
PERCENT_PAT = re.compile(PERCENT_PAT_ST, re.I)

# print('\nPERCENT_PAT_ST:')
# print(PERCENT_PAT_ST)


# pylint: disable=too-many-return-statements
def normalize_currency_unit(line: str) -> str:
    lc_line = line.lower()
    if lc_line in set(['$', 'usd', 'dollar', 'dollars']) or \
       re.search(r'u\.\s*s\.\s*dollars?', lc_line):
        return 'USD'
    elif lc_line in set(['€', 'eur', 'euro', 'euros']):
        return 'EUR'
    elif lc_line in set(['£', 'gbp', 'pound', 'pounds']):
        return 'GBP'
    elif lc_line in set(['円', 'cny', 'yuan', 'yuans']):
        return 'CNY'
    elif lc_line in set(['¥', 'jpy', 'yen', 'yens']):
        return 'JPY'
    elif lc_line in set(['₹', 'inr', 'rupee', 'rupees', 'rs', 'rs.']):
        return 'INR'

    return 'UNKNOWN_CURRENCY'


def currency_to_norm_dict(cx_mat: Match, line: str) -> Dict:
    # print('  currency cx_mat group: {} {} [{}]'.format(cx_mat.start(), cx_mat.end(), cx_mat.group()))
    # for gi, group in enumerate(cx_mat.groups(), 1):
    #    print("    cx_mat.group #{}: [{}]".format(gi, cx_mat.group(gi)))
    norm_unit = normalize_currency_unit(cx_mat.group(1))
    norm_value = text2int.extract_number(cx_mat.group(3)).get('value', -1)
    norm_dict = {'norm': {'unit': norm_unit,
                          'value': norm_value},
                 'text': line[cx_mat.start():cx_mat.end()],
                 'start': cx_mat.start(),
                 'end': cx_mat.end()}
    return norm_dict


def currency_to_norm_dict_symbol(prev_num_start: int,
                                 prev_num_end: int,
                                 currency_end: int,
                                 line: str) -> Dict:
    num_st = line[prev_num_start:prev_num_end]
    norm_value = text2int.extract_number(num_st).get('value', -1)
    norm_unit = normalize_currency_unit(line[prev_num_end:currency_end].strip())
    norm_dict = {'norm': {'unit': norm_unit,
                          'value': norm_value},
                 'text': line[prev_num_start:currency_end],
                 'start': prev_num_start,
                 'end': currency_end}
    return norm_dict


def extract_currencies(line: str) -> List[Dict]:
    norm_line = remove_num_words_join_hyphen(line)
    result = []
    mat_list = CURRENCY_PAT.finditer(norm_line)
    for mat in mat_list:
        norm_dict = currency_to_norm_dict(mat, line)
        result.append(norm_dict)

    # Handle 'XXX dollars' using simplified regex to avoid nasty backtracking.
    #
    # '$XXX dollar' will trigger both prevoius and this regex.
    # For now, I am OK with capturing both and providing two positive
    # results.  In future, we can user overlap function to remove one
    # of them.
    number_dict_list = extract_numbers(norm_line)
    mat_list = CURRENCY_SYMBOL_PAT.finditer(norm_line)
    number_idx = 0
    for mat in mat_list:
        prev_num_start, prev_num_end, number_idx = find_prev_start_end_in_dict_list(number_dict_list,
                                                                                    number_idx,
                                                                                    mat.start(),
                                                                                    line)
        if prev_num_start != -1:
            norm_dict = currency_to_norm_dict_symbol(prev_num_start, prev_num_end, mat.end(), line)
            result.append(norm_dict)

    return result


PERCENT_SYMBOL_PAT_ST = r'(%|\bpercent\b)'
PERCENT_SYMBOL_PAT = re.compile(PERCENT_SYMBOL_PAT_ST, re.I)


def find_prev_start_end_in_dict_list(number_dict_list: List[Dict],
                                     number_idx: int,
                                     percent_start: int,
                                     line: str) -> Tuple[int, int, int]:
    list_len = len(number_dict_list)
    idx = number_idx
    ok_start, ok_end = -1, -1
    while idx < list_len:
        ndict = number_dict_list[idx]
        dstart, dend = ndict['start'], ndict['end']
        if dend < percent_start:
            ok_start, ok_end = dstart, dend
        elif dend == percent_start:
            ok_start, ok_end = dstart, dend
            return ok_start, ok_end, idx + 1
        else:
            if ok_end != -1 and \
               not line[ok_end:percent_start].strip():
                return ok_start, ok_end, idx
            return -1, -1, idx
        idx += 1

    # last one
    if ok_end != -1 and \
       not line[ok_end:percent_start].strip():
        return ok_start, ok_end, idx

    return -1, -1, list_len


def percent_to_norm_dict(prev_num_start: int,
                         prev_num_end: int,
                         percent_end: int,
                         line: str) -> Dict:
    norm_value = -1
    num_st = line[prev_num_start:prev_num_end]
    norm_value = text2int.extract_number(num_st).get('value', -1)
    norm_dict = {'norm': {'unit': '%',
                          'value': norm_value},
                 'text': line[prev_num_start:percent_end],
                 'start': prev_num_start,
                 'end': percent_end}
    return norm_dict


def extract_percents(line: str) -> List[Dict]:
    norm_line = remove_num_words_join_hyphen(line)
    result = []
    number_dict_list = extract_numbers(line)
    mat_list = PERCENT_SYMBOL_PAT.finditer(norm_line)

    number_idx = 0
    for mat in mat_list:
        prev_num_start, prev_num_end, number_idx = find_prev_start_end_in_dict_list(number_dict_list,
                                                                                    number_idx,
                                                                                    mat.start(),
                                                                                    line)
        if prev_num_start != -1:
            norm_dict = percent_to_norm_dict(prev_num_start, prev_num_end, mat.end(), line)
            result.append(norm_dict)
    return result


def number_to_norm_dict(cx_mat: Match, line: str, offset: int = -1) -> Dict:
    # print('  number cx_mat group: {} {} [{}]'.format(cx_mat.start(), cx_mat.end(), cx_mat.group()))
    # for gi, group in enumerate(cx_mat.groups(), 1):
    #     print("    numb cx_mat.group #{}: [{}]".format(gi, cx_mat.group(gi)))
    norm_value = -1
    if cx_mat.group():
        norm_value = text2int.extract_number(cx_mat.group()).get('value', -1)

    adjusted_offset = 0
    if offset != -1:
        adjusted_offset = offset
    norm_dict = {'norm': {'value': norm_value},
                 'text': line[adjusted_offset + cx_mat.start():adjusted_offset + cx_mat.end()],
                 'start': adjusted_offset + cx_mat.start(),
                 'end': adjusted_offset + cx_mat.end()}
    return norm_dict


D_D_SPLIT_REGEX = re.compile(r'(?<=\d)\s+(?=\d)')

def num_num_split(line: str, offset: int = 0) -> List[Tuple[int, int, str]]:
    line = line.strip()
    chunks = list(D_D_SPLIT_REGEX.finditer(line))
    if chunks:
        result = []  # type: List[Tuple[int, int, str]]
        prev = 0
        for chunk in chunks:
            result.append((offset + prev, offset + chunk.start(), line[prev:chunk.start()]))
            prev = chunk.end()
        result.append((offset + prev, offset + len(line), line[prev:]))
        return result
    return [(offset, offset + len(line), line)]

# following numbers are not valid
# 'm'  'b', 't'
INVALID_NUM_REGEX = re.compile(r'(\s*\b[mbt]\-\S+\s*|^and\s*|(?<=\d)\s+and\s+(?=\d)|,(?=\d{4})|(?<=\d)\s*\-\s*(?=\d))')

MATCH_ALL_REGEX = re.compile(r'^.*$')

D_DASH_D_REGEX = re.compile(r'\d+\-\d+')

# is line always just one word, or can be multiple words?
# it seems to be possible to be multiple words
def is_invalid_number_word(word: str) -> bool:
    # a date, 02-03
    if D_DASH_D_REGEX.search(word):
        return True

    norm_st = text2int.normalize_comma_period(word)
    # sechead, 2.3.3
    if norm_st.count('.') > 1:
        return True
    return False

NUM_MBT_REGEX = re.compile(r'^\d+[mbt]$', re.I)

def is_invalid_number_phrase(line: str) -> bool:
    if line.lower() in set(['b', 'm', 't']):
        return True

    if NUM_MBT_REGEX.search(line):
        return True

    return False


def invalid_num_split(mat: Match) -> List[Tuple[Match, int]]:
    line = mat.group()
    if is_invalid_number_phrase(line):
        return []

    # should not start with 'and' and all 'mbt-*' are not valid numbers
    # so they are removed
    split_chunks = list(INVALID_NUM_REGEX.finditer(line))
    if split_chunks:
        adjusted_offset = mat.start()
        result = []  # type: List[Tuple[Match, int]]
        prev = 0
        for split_chunk in split_chunks:
            chunk_st = line[prev:split_chunk.start()]
            tmp_mat = NUMBER_PAT.search(chunk_st)
            if tmp_mat and chunk_st and not is_invalid_number_word(chunk_st):
                result.append((tmp_mat, adjusted_offset + prev))
            prev = split_chunk.end()
        tmp_mat = NUMBER_PAT.search(line[prev:])
        if tmp_mat and line[prev:] and not is_invalid_number_word(line[prev:]):
            result.append((tmp_mat, adjusted_offset + prev))
        return result

    return [(mat, -1)]


def extract_numbers(line: str) -> List[Dict]:
    norm_line = remove_num_words_join_hyphen(line)
    result = []
    mat_list = NUMBER_PAT.finditer(norm_line)
    # some mat in list might have mutliple intergers, such as '2 3'
    # 'better_mat_list' will store the real numeric mat
    mat_offset_list = []  # type: List[Tuple[Match, int]]
    for mat in mat_list:
        # Due ot our preference to parse English expressions mixed
        # with numbers, '2 4' might be captured in the match part.
        # We separate those numbers here.

        num_num_span_list = num_num_split(mat.group(), mat.start())
        if len(num_num_span_list) > 1:
            for offset_start, unused_offset_end, span_st in num_num_span_list:
                # string is 'm-3'
                # re.search(r'^[mtb]\-', span_st):
                if re.search(r'^[mtb]\-', span_st):
                    continue

                mat2 = NUMBER_PAT.search(span_st)
                if mat2:
                    mat_offset_list.append((mat2, offset_start))
        else:
            # get rid of 'm-2'
            # remove_invalid_num_spans = invalid_num_split(mat)
            mat_offset_list.extend(invalid_num_split(mat))

    for mat, mat_start in mat_offset_list:
        # print('mat = {}, mat_start = {}'.format(mat, mat_start))
        # norm_st = text2int.normalize_comma_period(mat.group())
        # 2.3.4, or section head
        # 2018-01-01 or date
        if not mat or is_invalid_number_phrase(mat.group()) or \
           is_invalid_number_word(mat.group()):
            continue

        norm_dict = number_to_norm_dict(mat, line, mat_start)
        result.append(norm_dict)
    return result


# pylint: disable=too-few-public-methods
class RegexContextGenerator:

    # pylint: disable=too-many-arguments
    def __init__(self,
                 num_prev_words: int,
                 num_post_words: int,
                 center_regex: Pattern,
                 candidate_type: str,
                 length_min: int = 0,
                 group_num: int = 1) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words
        self.center_regex = center_regex
        self.candidate_type = candidate_type
        self.length_min = length_min
        self.group_num = group_num


    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def get_candidates_from_text(self,
                                 nl_text: str,
                                 group_id: int = 0,
                                 # pylint: disable=line-too-long
                                 label_ant_list_param: Optional[List[ebsentutils.ProvisionAnnotation]] = None,
                                 label_list_param: Optional[List[bool]] = None,
                                 label: Optional[str] = None):
        # pylint: disable=line-too-long
        label_ant_list, label_list = [], []  # type: List[ebsentutils.ProvisionAnnotation], List[bool]
        if label_ant_list_param is not None:
            label_ant_list = label_ant_list_param
        if label_list_param is not None:
            label_list = label_list_param

        candidates = [] # type: List[Dict]
        group_id_list = [] # type: List[int]

        match_dict_list = []  # type: List[Dict]
        if self.candidate_type == 'NUMBER':
            match_dict_list = extract_numbers(nl_text)
        elif self.candidate_type == 'CURRENCY':
            match_dict_list = extract_currencies(nl_text)
        elif self.candidate_type == 'PERCENT':
            match_dict_list = extract_percents(nl_text)
        else:
            matches = self.center_regex.finditer(nl_text)
            for match in matches:
                match_start, match_end = match.span(self.group_num)
                match_str = match.group(self.group_num)
                match_dict_list.append({'start': match_start,
                                        'end': match_end,
                                        'text': match.group()})

        """
        matches = self.center_regex.finditer(nl_text)
        for match in matches:
            match_start, match_end = match.span(self.group_num)
            match_str = match.group(self.group_num)
        """

        for match_dict in match_dict_list:
            match_start, match_end = match_dict['start'], match_dict['end']
            match_str = match_dict['text']

            norm_dict = match_dict.get('norm', {})

            is_label = ebsentutils.check_start_end_overlap(match_start,
                                                           match_end,
                                                           label_ant_list)
            prev_n_words, prev_spans = strutils.get_prev_n_clx_tokens(nl_text,
                                                                      match_start,
                                                                      self.num_prev_words)
            post_n_words, post_spans = strutils.get_post_n_clx_tokens(nl_text,
                                                                      match_end,
                                                                      self.num_post_words)
            new_bow = '{} {} {}'.format(' '.join(prev_n_words),
                                        match_str,
                                        ' '.join(post_n_words))

            #update span based on window size
            new_start = match_start
            new_end = match_end
            if prev_spans:
                new_start = prev_spans[0][0]
            if post_spans:
                new_end = post_spans[-1][-1]

            """
            norm_dict = {}  # type: Dict
            if self.candidate_type == 'CURRENCY':
                norm_dict = currency_to_norm_dict(match, nl_text)
            elif self.candidate_type == 'NUMBER':
                # need to check for valid number because the extractor is
                # quite liberal.  Will catch sechead, such has 1.1.2
                norm_st = text2int.normalize_comma_period(match.group())
                if norm_st.count('.') >= 2:
                    continue
                norm_dict = number_to_norm_dict(match, nl_text)
            elif self.candidate_type == 'PERCENT':
                norm_dict = percent_to_norm_dict(match, nl_text)
            """

            a_candidate = {'candidate_type': self.candidate_type,
                           'bow_start': new_start,
                           'bow_end': new_end,
                           'text': new_bow,
                           'start': match_start,
                           'end': match_end,
                           'prev_n_words': ' '.join(prev_n_words),
                           'post_n_words': ' '.join(post_n_words),
                           'chars': match_str}
            if norm_dict:
                a_candidate['norm'] = norm_dict

            candidates.append(a_candidate)
            group_id_list.append(group_id)
            if is_label:
                a_candidate['label_human'] = label
                label_list.append(True)
            else:
                label_list.append(False)

        # remove any candidate that is >= min_length
        filtered_candidates = []  # type: List[Dict]
        filtered_label_list = []  # type: List[bool]
        filtered_group_id_list = []  # type: List[int]
        for candidate, cand_label, cand_group_id in zip(candidates,
                                                        label_list,
                                                        group_id_list):
            if len(candidate['chars']) >= self.length_min:
                filtered_candidates.append(candidate)
                filtered_label_list.append(cand_label)
                filtered_group_id_list.append(cand_group_id)
        return filtered_candidates, filtered_group_id_list, filtered_label_list

    # pylint: disable=too-many-locals
    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                label: Optional[str] = None) \
                                -> List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                              List[Dict],
                                              List[bool],
                                              List[int]]]:

        if 'length_min' not in self.__dict__:
            self.length_min = 0
        if 'group_num' not in self.__dict__:
            self.group_num = 1
        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):  # these are ebantdoc4
            label_list = []   # type: List[bool]

            #creates list of ants for a specific provision
            ant_list = antdoc.prov_annotation_list
            label_ant_list = []  # type: List[ebsentutils.ProvisionAnnotation]
            for ant in ant_list:
                if ant.label == label:
                    label_ant_list.append(ant)

            #gets text based on document type
            if antdoc.doc_format in set([ebantdoc4.EbDocFormat.html,
                                         ebantdoc4.EbDocFormat.html_nodocstruct,
                                         ebantdoc4.EbDocFormat.other]):
                nl_text = antdoc.text
            else:
                nl_text = antdoc.get_nl_text()

            if group_id % 10 == 0:
                logger.debug('RegexContextGenerator.documents_to_candidates(), group_id = %d',
                             group_id)

            candidates, group_id_list, label_list = self.get_candidates_from_text(nl_text,
                                                                                  group_id=group_id,
                                                                                  label_ant_list_param=label_ant_list,
                                                                                  label_list_param=label_list,
                                                                                  label=label)
            result.append((antdoc, candidates, label_list, group_id_list))
        return result
