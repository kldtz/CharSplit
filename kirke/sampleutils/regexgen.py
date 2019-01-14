import logging
import re
from typing import Dict, List, Optional, Pattern, Tuple

from kirke.utils import ebantdoc4, ebsentutils, strutils

from kirke.utils.text2int import extract_number

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    elif lc_line in set(['₹', 'inr', 'rupee', 'rupees', 'rs']):
        return 'INR'

    return 'UNKNOWN_CURRENCY'

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
        matches = self.center_regex.finditer(nl_text)
        for match in matches:
            match_start, match_end = match.span(self.group_num)
            match_str = match.group(self.group_num)
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

            norm_dict = {}
            if self.candidate_type == 'CURRENCY':
                cx_mat = self.center_regex.search(match.group())
                # print('\ncx_mat group: {} {} [{}]'.format(cx_mat.start(), cx_mat.end(), cx_mat.group()))
                # for gi, group in enumerate(cx_mat.groups()):
                #     print("  cx_mat.group #{}: [{}]".format(gi+1, cx_mat.group(gi+1)))
                norm_unit = 'USD'
                norm_value = -1
                if cx_mat.group(3):
                    norm_unit = normalize_currency_unit(cx_mat.group(3))
                    norm_value = extract_number(cx_mat.group(5)).get('value', -1)
                elif cx_mat.group(9):
                    norm_unit = normalize_currency_unit(cx_mat.group(13))
                    norm_value = extract_number(cx_mat.group(9)).get('value', -1)
                norm_dict = {'unit': norm_unit,
                             'value': norm_value}
            elif self.candidate_type == 'NUMBER':
                cx_mat = self.center_regex.search(match.group())
                print('\nnum cx_mat group: {} {} [{}]'.format(cx_mat.start(), cx_mat.end(), cx_mat.group()))
                for gi, group in enumerate(cx_mat.groups()):
                    print("  num cx_mat.group #{}: [{}]".format(gi+1, cx_mat.group(gi+1)))
                # mat_text = re.sub('[\.,]$', '', cx_mat.group().strip())
                # if len(list(re.finditer(r'\.', mat_text))) >= 2:  # this is sectionhead 7.2.1
                #    # this is NOT a number
                #    print('skipping, not a number')
                #    continue
                norm_value = -1
                if cx_mat.group(2):
                    norm_value = extract_number(cx_mat.group(2)).get('value', -1)
                elif cx_mat.group(5):
                    norm_value = extract_number(cx_mat.group(5)).get('value', -1)

                norm_dict = norm_value
            elif self.candidate_type == 'PERCENT':
                cx_mat = self.center_regex.search(match.group())
                print('\nperc cx_mat group: {} {} [{}]'.format(cx_mat.start(), cx_mat.end(), cx_mat.group()))
                for gi, group in enumerate(cx_mat.groups()):
                    print("  perc cx_mat.group #{}: [{}]".format(gi+1, cx_mat.group(gi+1)))
                norm_value = -1
                if cx_mat.group(4):
                    norm_value = extract_number(cx_mat.group(4)).get('value', -1)
                norm_dict = {'unit': '%',
                             'value': norm_value}


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
