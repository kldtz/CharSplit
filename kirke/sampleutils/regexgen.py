import logging
import re
from typing import Dict, List, Pattern, Tuple

from kirke.utils import ebantdoc3, ebsentutils, strutils

# pylint: disable=too-few-public-methods
class RegexContextGenerator:

    def __init__(self,
                 num_prev_words: int,
                 num_post_words: int,
                 center_regex: Pattern,
                 candidate_type: str) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words
        self.center_regex = center_regex
        self.candidate_type = candidate_type

    # pylint: disable=too-many-locals
    def documents_to_candidates(self,
                             antdoc_list: List[ebantdoc3.EbAnnotatedDoc3],
                             label: str = None)  -> List[Tuple[ebantdoc3.EbAnnotatedDoc3,
                                                               List[Dict],
                                                               List[bool],
                                                               List[int]]]:

        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc3.EbAnnotatedDoc3, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):  # these are ebantdoc3
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
            if antdoc.doc_format in set([ebantdoc3.EbDocFormat.html,
                                         ebantdoc3.EbDocFormat.html_nodocstruct,
                                         ebantdoc3.EbDocFormat.other]):
                nl_text = antdoc.text
            else:
                nl_text = antdoc.nl_text

            if group_id % 10 == 0:
                logging.info('RegexContextGenerator.documents_to_candidates(), group_id = %d',
                             group_id)

            #finds all matches in the text and adds window around each as a candidate
            matches = self.center_regex.finditer(nl_text, re.I)
            for match in matches:
                match_start, match_end = match.span(1)
                match_str = match.group(1)
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

                a_candidate = {'candidate_type': self.candidate_type,
                            'bow_start': new_start,
                            'bow_end': new_end,
                            'text': new_bow,
                            'start': match_start,
                            'end': match_end,
                            'prev_n_words': ' '.join(prev_n_words),
                            'post_n_words': ' '.join(post_n_words)}
                candidates.append(a_candidate)
                group_id_list.append(group_id)
                if is_label:
                    a_candidate['label_human'] = label
                    label_list.append(True)
                else:
                    label_list.append(False)
            result.append((antdoc, candidates, label_list, group_id_list))
        return result
