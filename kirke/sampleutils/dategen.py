import logging
from typing import Dict, List, Optional, Tuple

from kirke.ebrules import dates
from kirke.utils import ebantdoc5, ebsentutils, strutils

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# pylint: disable=too-few-public-methods
class DateSpanGenerator:

    def __init__(self, num_prev_words: int, num_post_words: int, candidate_type: str) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words
        self.candidate_type = candidate_type

    # pylint: disable=too-many-arguments, too-many-locals
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
        matches = []  # List[Tuple[int, int]]
        offset = 0
        for line in nl_text.split('\n'):
            if line:
                tmp_matches = dates.extract_std_dates(line)
                for tmp_start, tmp_end in tmp_matches:
                    matches.append((offset + tmp_start, offset + tmp_end))
            offset += len(line) + 1
        doc_len = len(nl_text)
        for mat_i, (match_start, match_end) in enumerate(matches):
            match_str = nl_text[match_start:match_end]
            is_label = ebsentutils.check_start_end_overlap(match_start,
                                                           match_end,
                                                           label_ant_list)

            prev_n_words, prev_spans = \
                strutils.get_prev_n_clx_tokens(nl_text,
                                               match_start,
                                               self.num_prev_words)
            post_n_words, post_spans = \
                strutils.get_post_n_clx_tokens(nl_text,
                                               match_end,
                                               self.num_post_words)

            prev_15_words = ['PV15_' + wd for wd in prev_n_words[-15:]]
            post_15_words = ['PS15_' + wd for wd in post_n_words[:15]]
            prev_10_words = ['PV10_' + wd for wd in prev_n_words[-10:]]
            post_10_words = ['PS10_' + wd for wd in post_n_words[:10]]
            prev_5_words = ['PV5_' + wd for wd in prev_n_words[-5:]]
            post_5_words = ['PS5_' + wd for wd in post_n_words[:5]]
            prev_2_words = ['PV2_' + wd for wd in prev_n_words[-2:]]
            post_2_words = ['PS2_' + wd for wd in post_n_words[:2]]
            prev_n_words_plus = prev_n_words + ['EOLN'] + prev_15_words + ['EOLN'] + \
                                prev_10_words + ['EOLN'] + prev_5_words + ['EOLN'] + prev_2_words
            post_n_words_plus = post_n_words + ['EOLN'] + post_15_words + ['EOLN'] + \
                                post_10_words + ['EOLN'] + post_5_words + ['EOLN'] + post_2_words

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

            candidate_percentage10 = min((mat_i + 1) / 10.0, 1.0)
            a_candidate = {'candidate_type': self.candidate_type,
                           'bow_start': new_start,
                           'bow_end': new_end,
                           'text': new_bow,
                           'start': match_start,
                           'end': match_end,
                           'chars': match_str,
                           'prev_n_words': ' '.join(prev_n_words_plus),
                           'post_n_words': ' '.join(post_n_words_plus),
                           'candidate_percent10': candidate_percentage10,
                           'doc_percent': match_start / doc_len}
            candidates.append(a_candidate)
            group_id_list.append(group_id)
            if is_label:
                a_candidate['label_human'] = label
                label_list.append(True)
            else:
                label_list.append(False)
        return candidates, group_id_list, label_list

    # pylint: disable=too-many-locals
    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc5.EbAnnotatedDoc],
                                label: Optional[str] = None) \
                                -> List[Tuple[ebantdoc5.EbAnnotatedDoc,
                                              List[Dict],
                                              List[bool],
                                              List[int]]]:
        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc5.EbAnnotatedDoc, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):

            label_list = []   # type: List[bool]

            #creates list of ants for a specific provision
            ant_list = antdoc.prov_annotation_list
            label_ant_list = []
            for ant in ant_list:
                if ant.label == label:
                    label_ant_list.append(ant)

            #gets text based on document type
            if antdoc.doc_format in set([ebantdoc5.EbDocFormat.html,
                                         ebantdoc5.EbDocFormat.html_nodocstruct,
                                         ebantdoc5.EbDocFormat.other]):
                nl_text = antdoc.text
            else:
                nl_text = antdoc.get_nl_text()

            if group_id % 10 == 0:
                logger.debug('DateSpanGenerator.documents_to_candidates(), group_id = %d', group_id)

            candidates, group_id_list, label_list = self.get_candidates_from_text(nl_text,
                                                                                  group_id=group_id,
                                                                                  label_ant_list_param=label_ant_list,
                                                                                  label_list_param=label_list,
                                                                                  label=label)

            result.append((antdoc, candidates, label_list, group_id_list))
        return result
