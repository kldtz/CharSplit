import logging
from typing import Dict, List, Tuple

from kirke.ebrules import dates
from kirke.utils import ebantdoc3, ebsentutils, strutils

    
class LineSpanGenerator:
    
    def __init__(self, num_prev_words: int, num_post_words: int) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words

    def documents_to_samples(self,
                             antdoc_list: List[ebantdoc3.EbAnnotatedDoc3],
                             label: str=None) -> Tuple[List[Dict], List[bool], List[int]]:
        samples = []  # type: List[Dict]
        label_list = []   # type: List[bool]
        group_id_list = []  # type: List[int]


        # each sample is the date regex +
        for group_id, antdoc in enumerate(antdoc_list):  # these are ebantdoc3
            
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
                logging.info("LineSpanGenerator.documents_to_samples(), group_id = {}".format(group_id))

            lines = nl_text.split('\n')
            offset = 0
            notempty_line_seq = 0
            
            #creates dict for candidate information
            for line in lines:
                if not line:  # skip
                    offset += len(line) + 1
                    continue

                line_len = len(line)
                start = offset
                end = offset + line_len

                prev_n_words = strutils.get_lc_prev_n_words(nl_text, start, self.num_prev_words)
                post_n_words = strutils.get_lc_post_n_words(nl_text, end, self.num_post_words)

                is_label = ebsentutils.check_start_end_overlap(start,
                                                               end,
                                                               label_ant_list)
                a_sample = {'sample_type': 'line',
                            'start': start,
                            'end': end,
                            'line_seq': notempty_line_seq,
                            'text': nl_text[start:end],
                            'prev_n_words': ' '.join(prev_n_words),
                            'post_n_words': ' '.join(post_n_words)}
                notempty_line_seq += 1
                
                #updates boolean list for whether candidate is pos for provision?
                if is_label:
                    a_sample['label_human'] = label
                    label_list.append(True)
                    # print('sample = {}'.format(a_sample))
                else:
                    label_list.append(False)
                    # print('sample = {}'.format(a_sample['text']))
                samples.append(a_sample)
                group_id_list.append(group_id)                            
                            
                offset += len(line) + 1  # for eoln

        return samples, label_list, group_id_list