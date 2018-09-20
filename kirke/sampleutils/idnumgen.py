import logging
import re
from typing import Dict, List, Match, Pattern, Tuple

from kirke.utils import ebantdoc4, ebsentutils, strutils

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_idnum_words_mergeable(dicta: Dict, dictb: Dict, line: str) -> bool:
    diff_str = line[dicta['end']:dictb['start']]
    return bool(re.match('^ {1,3}$', diff_str) or not diff_str)


def merge_adjacent_idnum_words(alist: List[Dict],
                               line: str) -> List[Dict]:
    if not alist:
        return alist
    cur_dict = alist[0]
    result = [cur_dict]
    for next_dict in alist[1:]:
        if is_idnum_words_mergeable(cur_dict, next_dict, line):
            # merge_two(cur_dict, next_dict)
            cur_dict['end'] = next_dict['end']
            cur_dict['chars'] = line[cur_dict['start']:cur_dict['end']]
        else:
            cur_dict = next_dict
            result.append(cur_dict)
    return result


def match_to_idnum_word(mat: Match,
                        group_num: int) -> Dict:
    match_start, match_end = mat.span(group_num)
    match_str = mat.group(group_num)

    # clean up the string if special character is at the end.  Currently
    # none of the matat_str will have nose characters except for ";" or ":"
    if match_str.endswith(',') or match_str.endswith(';') or \
       match_str.endswith(':') or match_str.endswith('.'):
        match_str = match_str[:-1]
        match_end -= 1
    if match_str.endswith(')') and not '(' in match_str:
        match_str = match_str[:-1]
        match_end -= 1
    if match_str.startswith('(') and not ')' in match_str:
        match_str = match_str[1:]
        match_start += 1

    idnum_word = {'start': match_start,
                  'end': match_end,
                  'chars': match_str}
    return idnum_word


PHONE_PREFIX_PAT = re.compile(r'(phone|ph|telephone|tel|mobile|land\s?line|office|home):', re.I)

def extract_idnum_list(line: str,
                       idnum_word_pat: Pattern,
                       group_num: int = 1,
                       is_join: bool = False,
                       length_min: int = 1) -> List[Dict]:

    mat_list = idnum_word_pat.finditer(line)

    candidates = [match_to_idnum_word(mat, group_num)
                  for mat in mat_list]
    if is_join:
        candidates = merge_adjacent_idnum_words(candidates, line)

    for cand in candidates:
        cand_str = cand['chars']
        prefix_mat = PHONE_PREFIX_PAT.match(cand_str)
        if prefix_mat:
            mat_len = len(prefix_mat.group())
            cand['chars'] = cand_str[mat_len:]
            cand['start'] = cand['start'] + mat_len

    # remove any candidate with len < length_min
    candidates = [cand for cand in candidates if len(cand['chars']) >= length_min]

    return candidates


# pylint: disable=too-few-public-methods
class IdNumContextGenerator:

    # pylint: disable=too-many-arguments
    def __init__(self,
                 num_prev_words: int,
                 num_post_words: int,
                 regex_pat: Pattern,
                 candidate_type: str,
                 is_join: bool = False,
                 length_min: int = 1,
                 group_num: int = 1) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words
        self.regex_pat = regex_pat
        self.candidate_type = candidate_type
        self.is_join = is_join
        self.length_min = length_min
        self.group_num = group_num

    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
    def get_candidates_from_text(self,
                                 nl_text: str,
                                 group_id: int,
                                 # pylint: disable=line-too-long
                                 label_ant_list: List[ebsentutils.ProvisionAnnotation],
                                 label: str = '') \
                                 -> Tuple[List[Dict],
                                          List[bool],
                                          List[int]]:
        candidates = [] # type: List[Dict]
        label_list = [] # type: List[bool]
        group_id_list = [] # type: List[int]
        idnum_list = extract_idnum_list(nl_text,
                                        idnum_word_pat=self.regex_pat,
                                        group_num=self.group_num,
                                        is_join=self.is_join,
                                        length_min=self.length_min)
        for idnum_dict in idnum_list:
            match_start, match_end = idnum_dict['start'], idnum_dict['end']
            match_str = idnum_dict['chars']
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
            new_start, new_end = match_start, match_end
            if prev_spans:
                new_start = prev_spans[0][0]
            if post_spans:
                new_end = post_spans[-1][-1]

            a_candidate = {'candidate_type': self.candidate_type,
                           'bow_start': new_start,
                           'bow_end': new_end,
                           'text': new_bow,
                           'start': match_start,
                           'end': match_end,
                           'prev_n_words': ' '.join(prev_n_words),
                           'post_n_words': ' '.join(post_n_words),
                           'chars': match_str}
            candidates.append(a_candidate)

            group_id_list.append(group_id)
            if is_label:
                a_candidate['label_human'] = label
            label_list.append(is_label)

        return candidates, label_list, group_id_list


    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                label: str = '') \
                                -> List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                              List[Dict],
                                              List[bool],
                                              List[int]]]:
        if not hasattr(self, 'length_min'):
            self.length_min = 1
        if not hasattr(self, 'is_join'):
            self.is_join = False
        if not hasattr(self, 'group_num'):
            self.group_num = 1

        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):  # these are ebantdoc5

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

            candidates, cand_label_list, cand_group_id_list = \
                self.get_candidates_from_text(nl_text,
                                              group_id=group_id,
                                              label_ant_list=label_ant_list,
                                              label=label)

            result.append((antdoc, candidates, cand_label_list, cand_group_id_list))
        return result
