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

    def merge_candidates(self, cands: List[Dict], nl_text: str) -> Dict:
        # doesn't need to be merged
        if len(cands) == 1:
            return cands[0]

        # else concat the list of candidates to a single dictionary
        else:
            new_cand = {'candidate_type': self.candidate_type,
                        'bow_start': cands[0]['bow_start'],
                        'bow_end': cands[-1]['bow_end'],
                        'start': cands[0]['start'],
                        'end': cands[-1]['end'],
                        'prev_n_words': cands[0]['prev_n_words'],
                        'post_n_words': cands[-1]['post_n_words']}
            new_cand['text'] = nl_text[new_cand['bow_start']:new_cand['bow_end']]
            new_cand['chars'] = nl_text[new_cand['start']:new_cand['end']]
            return new_cand

    def get_candidates_from_text(self, nl_text: str, group_id: int, label_ant_list: List[str], label_list: List[bool]):
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
        # joins adjacent candidates that are only separated by whitespace
        if self.join:
            candidates_to_merge = []
            new_candidates = []
            merge_labels = []
            merge_groups = []
            i = 0
            while i < len(candidates):
                skip = True
                candidates_to_merge.append(candidates[i])
                # looks ahead until it fails the requirement
                while skip and i+1 < len(candidates):
                    diff = candidates[i+1]['start'] - candidates_to_merge[-1]['end']
                    diff_str = nl_text[candidates_to_merge[-1]['end']:candidates[i+1]['start']]
                    if (diff_str.isspace() or not diff_str) and diff < 3:
                        candidates_to_merge.append(candidates[i+1])
                        i += 1
                    else:
                        skip = False

                    # merges candidates that pass the requirement
                merged_cands = self.merge_candidates(candidates_to_merge, nl_text)
                new_candidates.append(merged_cands)
                candidates_to_merge = []
                merge_labels.append(label_list[i])
                merge_groups.append(group_id_list[i])
                i += 1

            candidates = new_candidates
            label_list = merge_labels
            group_id_list = merge_groups
        # remove any candidate that is >= min_length
        filtered_candidates = []  # type: List[Dict]
        filtered_label_list = []  # type: List[bool]
        filtered_group_id_list = []  # type: List[int]
        for candidate, cand_label, group_id in zip(candidates,
                                              label_list,
                                              group_id_list):
            if len(candidate['chars']) >= self.length_min:
                filtered_candidates.append(candidate)
                filtered_label_list.append(cand_label)
                filtered_group_id_list.append(group_id)
        return filtered_candidates, filtered_group_id_list, filtered_label_list

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
        if 'group_num' not in self.__dict__:
            self.group_num = 1
        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):  # these are ebantdoc4
            label_list = []   # type: List[bool]

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

            candidates, group_id_list, label_list = self.get_candidates_from_text(nl_text, group_id, label_ant_list, label_list)
            result.append((antdoc, candidates, label_list, group_id_list))
        return result

    def extract_doc_candidates(regex_pat: Pattern,
                               group_num: int,
                               atext: str,
                               candidate_type: str,
                               num_prev_words: int,
                               num_post_words: int,
                               min_length: int,
                               is_join: bool) -> List[Dict]:

        candidates, group_id_list, label_list = self.get_candidates_from_text(nl_text, group_id, label_list)
        return candidates










