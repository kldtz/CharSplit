
from typing import Dict, List, Optional, Tuple

from kirke.utils import antutils, ebantdoc4, strutils



# pylint: disable=too-few-public-methods
class LineSpanGenerator:

    def __init__(self, num_prev_words: int, num_post_words: int) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words

    # pylint: disable=too-many-locals
    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                label: Optional[str] = None) \
                                -> List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                              List[Dict],
                                              List[bool],
                                              List[int]]]:
        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
        # each candidate is the date regex +
        for group_id, antdoc in enumerate(antdoc_list):  # these are ebantdoc4
            candidates = []  # type: List[Dict]
            label_list = []   # type: List[bool]
            group_id_list = []  # type: List[int]

            # get only ant for this particular label
            label_ant_list = antdoc.get_provision_annotations(label)

            if antdoc.doc_format in set([ebantdoc4.EbDocFormat.html,
                                         ebantdoc4.EbDocFormat.html_nodocstruct,
                                         ebantdoc4.EbDocFormat.other]):
                nl_text = antdoc.text
            else:
                nl_text = antdoc.get_nl_text()

            # if group_id % 10 == 0:
            #    print("LineSpanGenerator.documents_to_candidates(), group_id = {}". \
            #        format(group_id))

            lines = nl_text.split('\n')
            offset = 0
            notempty_line_seq = 0

            for line in lines:
                # there are lines with just spaces, or non-breaking spaces
                if not line or not line.strip():  # skip
                    offset += len(line) + 1
                    continue

                line_len = len(line)
                start = offset
                end = offset + line_len

                prev_n_words, _ = strutils.get_lc_prev_n_words(nl_text,
                                                               start,
                                                               self.num_prev_words)
                post_n_words, _ = strutils.get_lc_post_n_words(nl_text,
                                                               end,
                                                               self.num_post_words)
                is_label = antutils.check_start_end_overlap(start,
                                                            end,
                                                            label_ant_list)
                a_candidate = {'candidate_type': 'line',
                               'start': start,
                               'end': end,
                               'line_seq': notempty_line_seq,
                               'text': nl_text[start:end],
                               'prev_n_words': ' '.join(prev_n_words),
                               'post_n_words': ' '.join(post_n_words)}
                notempty_line_seq += 1

                if is_label:
                    a_candidate['label_human'] = label
                    label_list.append(True)
                    # print('candidate = {}'.format(a_candidate))
                else:
                    label_list.append(False)
                    # print('candidate = {}'.format(a_candidate['text']))
                candidates.append(a_candidate)
                group_id_list.append(group_id)

                offset += len(line) + 1  # for eoln

            result.append((antdoc, candidates, label_list, group_id_list))
        return result
