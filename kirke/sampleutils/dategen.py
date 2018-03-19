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
                             label: str = None) -> List[Tuple[ebantdoc3.EbAnnotatedDoc3,
                                                              List[Dict],
                                                              List[bool],
                                                              List[int]]]:
        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc3.EbAnnotatedDoc3, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):

            samples = []  # type: List[Dict]
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
                logging.info('DateSpanGenerator.documents_to_samples(), group_id = %d', group_id)


            # Finds all matches in the text and adds window around each as a candidate
            # We must parse line by line, otherwise, some regex capture too much.  That
            # will cause major issue with date normalization.  The reason is probably
            # because \s capture line break.
            matches = []  # List[Tuple[int, int]]
            offset = 0
            for line in nl_text.split('\n'):
                if line:
                    tmp_matches = dates.extract_std_dates(line)
                    for tmp_start, tmp_end in tmp_matches:
                        matches.append((offset + tmp_start, offset + tmp_end))
                offset += len(line) + 1
            
            #matches = dates.extract_std_dates(nl_text)  # List[Tuple[int, int]]
            
            # 'matches' is already sorted
            # matches = sorted(matches)
            doc_len = len(nl_text)
            for mat_i, (match_start, match_end) in enumerate(matches):
                match_str = nl_text[match_start:match_end]
                is_label = ebsentutils.check_start_end_overlap(match_start,
                                                               match_end,
                                                               label_ant_list)

                # change num_prev_word, num_post_word to 12, 12 get this very close to optimal 0.772
                # now it's 0.771
                prev_n_words, prev_spans = \
                    strutils.get_prev_n_clx_tokens(nl_text,
                                                   match_start,
                                                   self.num_prev_words-1)
                post_n_words, post_spans = \
                    strutils.get_post_n_clx_tokens(nl_text,
                                                   match_end,
                                                   self.num_post_words-1)
                """

                prev_n_words, prev_spans = \
                    strutils.get_prev_n_words_with_quote(nl_text,
                                                         match_start,
                                                         self.num_prev_words,
                                                         is_lower=True,
                                                         is_quote=True)
                post_n_words, post_spans = \
                    strutils.get_post_n_words_with_quote(nl_text,
                                                         match_end,
                                                         self.num_post_words,
                                                         is_lower=True,
                                                         is_quote=True)
                """

                # Adding both lc and original-case words lowers 0.07% F1.
                # OK, the code is not correct since using set() messes up
                # the 2-gram for CountVector.
                # It basically increase FP without any other benefits in
                # FN or TP.
                # lc_prev_n_words = [wd.lower() for wd in prev_n_words]
                # lc_post_n_words = [wd.lower() for wd in post_n_words]
                # prev_n_words = set(prev_n_words + lc_prev_n_words)
                # post_n_words = set(post_n_words + lc_post_n_words)
                # Using original-case has same F1.

                # add first 4 words surround as addition features.  Improved.  :-)
                prev_4_words = ['PV4_' + wd for wd in prev_n_words[-4:]]
                post_4_words = ['PS4_' + wd for wd in post_n_words[:4]]
                # to deal with n-gram of 2, added 'EOLN' to not mix
                # prev_4_words with others
                prev_n_words_plus = prev_n_words + ['EOLN'] + prev_4_words
                post_n_words_plus = post_n_words + ['EOLN'] + post_4_words
                print("prev_n_words:\t{}".format(prev_n_words))
                print("post_n_words:\t{}".format(post_n_words))

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
                            'prev_n_words': ' '.join(prev_n_words_plus),
                            'post_n_words': ' '.join(post_n_words_plus),
                            'candidate_percent10': candidate_percentage10,
                            'doc_percent': match_start / doc_len}
                samples.append(a_sample)
                group_id_list.append(group_id)
                if is_label:
                    a_sample['label_human'] = label
                    label_list.append(True)
                else:
                    label_list.append(False)

            result.append((antdoc, samples, label_list, group_id_list))
        return result
