import logging
from typing import Dict, List, Tuple

from kirke.ebrules import dates
from kirke.utils import ebantdoc3, ebsentutils, strutils


# pylint: disable=too-few-public-methods
class DateSpanGenerator:

    def __init__(self, num_prev_words: int, num_post_words: int, sample_type: str) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words
        self.sample_type = sample_type

    # pylint: disable=too-many-locals
    def documents_to_samples(self,
                             antdoc_list: List[ebantdoc3.EbAnnotatedDoc3],
                             label: str = None) -> Tuple[List[Dict], List[bool], List[int]]:
        samples = []  # type: List[Dict]
        label_list = []   # type: List[bool]
        group_id_list = []  # type: List[int]

        for group_id, antdoc in enumerate(antdoc_list):

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
                logging.info('DateSpanGenerator.documents_to_samples(), group_id = %d', group_id)

            #finds all matches in the text and adds window around each as a candidate
            matches = dates.extract_std_dates(nl_text)
            doc_len = len(nl_text)
            for mat_i, (match_start, match_end) in enumerate(matches):
                match_str = nl_text[match_start:match_end]
                is_label = ebsentutils.check_start_end_overlap(match_start,
                                                               match_end,
                                                               label_ant_list)

                prev_n_words, prev_spans = \
                    strutils.get_lc_prev_n_words_with_quote_nl(nl_text,
                                                               match_start,
                                                               self.num_prev_words)
                post_n_words, post_spans = \
                    strutils.get_lc_post_n_words_with_quote_nl(nl_text,
                                                               match_end,
                                                               self.num_post_words)
                """
                prev_n_words, prev_spans = \
                    strutils.get_lc_prev_n_words(nl_text,
                                                 match_start,
                                                 self.num_prev_words)
                post_n_words, post_spans = \
                    strutils.get_lc_post_n_words(nl_text,
                                                 match_end,
                                                 self.num_post_words)
                """

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
                a_sample = {'sample_type': self.sample_type,
                            'start': new_start,
                            'end': new_end,
                            'text': new_bow,
                            'match_start': match_start,
                            'match_end': match_end,
                            'prev_n_words': ' '.join(prev_n_words),
                            'post_n_words': ' '.join(post_n_words),
                            'candidate_percent10': candidate_percentage10,
                            'doc_percent': match_start / doc_len}
                samples.append(a_sample)
                group_id_list.append(group_id)
                if is_label:
                    a_sample['label_human'] = label
                    label_list.append(True)
                else:
                    label_list.append(False)

        return samples, label_list, group_id_list
