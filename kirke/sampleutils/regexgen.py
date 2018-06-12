import logging
import re
from typing import Dict, List, Pattern, Tuple
import copy
from kirke.utils import ebantdoc4, ebsentutils, strutils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pylint: disable=too-few-public-methods
class RegexContextGenerator:

    def __init__(self,
                 num_prev_words: int,
                 num_post_words: int,
                 center_regex: Pattern,
                 candidate_type: str,
                 join: bool = False,
                 length_min: int = 0,
                 group_num: int = 1) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words
        self.center_regex = center_regex
        self.candidate_type = candidate_type
        self.join = join
        self.length_min = length_min
        self.group_num = group_num

    # pylint: disable=too-many-locals
    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                label: str = None) -> List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                                               List[Dict],
                                                               List[bool],
                                                               List[int]]]:

        if 'length_min' not in self.__dict__:
            self.length_min = 0
        if 'join' not in self.__dict__:
            self.join = False
        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):  # these are ebantdoc4
            candidates = []  # type: List[Dict]
            label_list = []   # type: List[bool]
            group_id_list = []  # type: List[int]

            #creates list of ants for a specific provision
            ant_list = antdoc.prov_annotation_list
            label_ant_list = []
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

            #finds all matches in the text and adds window around each as a candidate
            matches = self.center_regex.finditer(nl_text, re.I)
            for match in matches:
                match_start, match_end = match.span(group_num)
                match_str = match.group(group_num)
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
                if match_str.endswith(',') or match_str.endswith(';') or match_str.endswith(':'):
                    match_str = match_str[:-1]
                    match_end -= 1
                if match_str.endswith(')') and not '(' in match_str:
                    match_str = match_str[:-1]
                    match_end -= 1
                if match_str.startswith('(') and not ')' in match_str:
                    match_str = match_str[1:]
                    match_start += 1
                if len(match_str) > self.length_min:
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
                        label_list.append(True)
                    else:
                        label_list.append(False)
            if self.join:
                merge_candidates = []
                merge_labels = []
                merge_groups = []
                i = 0
                while i < len(candidates):
                    skip = True
                    new_candidate = copy.deepcopy(candidates[i])
                    while skip and i+1 < len(candidates):
                        diff = candidates[i+1]['start'] - new_candidate['end']
                        diff_str = nl_text[new_candidate['end']:candidates[i+1]['start']]
                        if (diff_str.isspace() or not diff_str) and diff < 3:
                            new_candidate['end'] = candidates[i+1]['end']
                            new_candidate['chars'] = nl_text[new_candidate['start']:new_candidate['end']]
                            i += 1
                        else:
                            merge_candidates.append(new_candidate)
                            merge_labels.append(label_list[i])
                            merge_groups.append(group_id_list[i])
                            i += 1
                            skip = False
                    if i == len(candidates) - 1:
                        skip = False
                        merge_candidates.append(candidates[i])
                        merge_labels.append(label_list[i])
                        merge_groups.append(group_id_list[i])
                        i += 1
                candidates = merge_candidates
                label_list = merge_labels
                group_id_list = merge_groups
            result.append((antdoc, candidates, label_list, group_id_list))
        return result
