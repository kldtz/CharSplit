import logging
import re, regex
from typing import Dict, List, Tuple
from kirke.ebrules import dates, addresses
from kirke.utils import ebantdoc3, ebsentutils, strutils

class AddrContextGenerator:
    def __init__(self, num_prev_words: int, num_post_words: int) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words

    def documents_to_samples(self,
                             antdoc_list: List[ebantdoc3.EbAnnotatedDoc3],
                             label: str=None) -> Tuple[List[Dict], List[bool], List[int]]:
        samples = []  # type: List[Dict]
        label_list = []   # type: List[bool]
        group_id_list = []  # type: List[int]

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
                logging.info("AddrContextGenerator.documents_to_samples(), group_id = {}".format(group_id))
           
            #loads address keywords 
            all_keywords = addresses.addr_keywords()
            split_text = nl_text.split()

            #finds all addresses in the text and adds window around each as a candidate
            for addr in addresses.find_addresses(nl_text, all_keywords):
                addr_start, addr_end, addr_st = addr
                is_label = ebsentutils.check_start_end_overlap(addr_start,
                                                               addr_end,
                                                               label_ant_list)
                prev_n_words, prev_spans = strutils.get_lc_prev_n_words(nl_text, addr_start, self.num_prev_words)
                post_n_words, post_spans = strutils.get_lc_post_n_words(nl_text, addr_end, self.num_post_words)
                new_bow = '{} {} {}'.format(' '.join(prev_n_words), addr_st, ' '.join(post_n_words))
                bow_start = addr_start
                bow_end = addr_end
                #update span based on window size
                if prev_spans:
                    bow_start = prev_spans[0][0]
                if post_spans:
                    bow_end = post_spans[-1][-1]
                a_sample = {'sample_type': 'addr',
                            'start': bow_start,
                            'end': bow_end,
                            'text': new_bow,
                            'addr_start': addr_start,
                            'addr_end': addr_end,
                            'prev_n_words': ' '.join(prev_n_words),
                            'post_n_words': ' '.join(post_n_words),
                            'has_addr': True}
                samples.append(a_sample)
                group_id_list.append(group_id)
                if is_label:
                    a_sample['label_human'] = label
                    label_list.append(True)
                else:
                    label_list.append(False)
        return samples, label_list, group_id_list
