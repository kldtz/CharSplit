import logging
from typing import Optional, Dict, List, Tuple

from kirke.ebrules import dates
from kirke.nlputil import dates_jp
from kirke.utils import ebantdoc4, ebsentutils, strutils

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# pylint: disable=too-few-public-methods
class DateSpanGenerator:

    def __init__(self, num_prev_words: int, num_post_words: int, candidate_type: str) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words
        self.candidate_type = candidate_type

    # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
    def get_candidates_from_text(self,
                                 nl_text: str,
                                 doc_lang: str,
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
        matches = []  # List[Tuple[int, int, str]]
        offset = 0
        for para_line in nl_text.split('\n\n'):
            if para_line:
                if doc_lang == 'ja':
                    datedict_list = dates_jp.extract_dates(para_line)
                else:
                    datedict_list = dates.extract_std_dates(para_line)

                for datedict in datedict_list:
                    tmp_start, tmp_end = datedict['start'], datedict['end']
                    matches.append((offset + tmp_start, offset + tmp_end, datedict['norm']))
            offset += len(para_line) + 2
        doc_len = len(nl_text)
        for mat_i, (match_start, match_end, norm_date_st) in enumerate(matches):
            match_str = nl_text[match_start:match_end]
            is_label = ebsentutils.check_start_end_overlap(match_start,
                                                           match_end,
                                                           label_ant_list)

            if doc_lang in set(['zh', 'ja']):
                prev_n_words, prev_spans = \
                    strutils.get_prev_n_chars_as_tokens(nl_text,
                                                        match_start,
                                                        self.num_prev_words)
                post_n_words, post_spans = \
                    strutils.get_post_n_chars_as_tokens(nl_text,
                                                        match_end,
                                                        self.num_post_words)
            else:
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
            prev_n_words_plus = prev_n_words + ['EOLN'] + prev_15_words + ['EOLN'] + prev_10_words + ['EOLN'] + prev_5_words + ['EOLN'] + prev_2_words
            post_n_words_plus = post_n_words + ['EOLN'] + post_15_words + ['EOLN'] + post_10_words + ['EOLN'] + post_5_words + ['EOLN'] + post_2_words

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
                           'doc_percent': match_start / doc_len,
                           'norm': norm_date_st}
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
                                antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                label: str = None) -> List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                                                 List[Dict],
                                                                 List[bool],
                                                                 List[int]]]:
        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):

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
                logger.debug('DateSpanGenerator.documents_to_candidates(), group_id = %d', group_id)

            candidates, group_id_list, label_list = self.get_candidates_from_text(nl_text,
                                                                                  doc_lang=antdoc.doc_lang,
                                                                                  group_id=group_id,
                                                                                  label_ant_list_param=label_ant_list,
                                                                                  label_list_param=label_list,
                                                                                  label=label)

            result.append((antdoc, candidates, label_list, group_id_list))
        return result
