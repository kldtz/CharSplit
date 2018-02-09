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
                logging.info("AddrContextGenerator.documents_to_samples(), group_id = {}".format(group_id))
            
            all_keywords = addresses.all_constituencies()
            split_text = nl_text.split()
            #print([x[2] for x in self.find_constituencies(nl_text, all_keywords)])
            for addr in addresses.find_constituencies(nl_text, all_keywords):
                new_start, new_end, addr_st = addr
                is_label = ebsentutils.check_start_end_overlap(new_start,
                                                               new_end,
                                                               label_ant_list)
                prev_n_words, prev_spans = strutils.get_lc_prev_n_words(nl_text, new_start, self.num_prev_words)
                post_n_words, post_spans = strutils.get_lc_post_n_words(nl_text, new_end, self.num_post_words)
                new_bow = '{} {} {}'.format(' '.join(prev_n_words), addr_st, ' '.join(post_n_words))
                #print("\t>>>>>", new_bow)
                if prev_spans:
                    new_start = prev_spans[0][0]
                if post_spans:
                    new_end = post_spans[-1][-1]
                a_sample = {'sample_type': 'addr',
                            'start': new_start,
                            'end': new_end,
                            'text': new_bow,
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
            '''
            lines = nl_text.split('\n')
            offset = 0
            notempty_line_seq = 0
            prev_start = 0
            #creates dict for candidate information
            for i in range(0, len(lines)):
                line = lines[i]
                line_len = len(line)
                start = offset
                end = offset + line_len
                new_bow = line.strip()
                x = None
                if not line:  # skip
                    offset += len(line) + 1
                    continue
                is_label = ebsentutils.check_start_end_overlap(start,
                                                               end,
                                                               label_ant_list)
                found_adresses = self.find_constituencies(line, all_keywords)
                if not found_adresses:
                    new_start = start
                    new_end = end
                    prev_n_words, prev_spans = strutils.get_lc_prev_n_words(nl_text, new_start, self.num_prev_words)
                    post_n_words, post_spans = strutils.get_lc_post_n_words(nl_text, new_end, self.num_post_words)
                    a_sample = {'sample_type': 'addr',
                                'start': new_start,
                                'end': new_end,
                                'line_seq': notempty_line_seq,
                                'text': new_bow,
                                'prev_n_words': ' ' .join(prev_n_words),
                                'post_n_words': ' '.join(post_n_words),
                                'has_addr': False}
                    samples.append(a_sample)
                    group_id_list.append(group_id)
                    if is_label:
                        a_sample['label_human'] = label
                        label_list.append(True)
                    else:
                        label_list.append(False)
                else:
                    for x in found_adresses: 
                        ad_start = x[0]
                        ad_end = x[1]
                        new_start = ad_start + start
                        new_end = ad_end + start
                        prev_n_words, prev_spans = strutils.get_lc_prev_n_words(nl_text, new_start, self.num_prev_words)
                        post_n_words, post_spans = strutils.get_lc_post_n_words(nl_text, new_end, self.num_post_words)
                        new_bow = '{} {} {}'.format(' '.join(prev_n_words), x[2], ' '.join(post_n_words))
                        #new_bow = '{} {}'.format(' '.join(prev_n_words), ' '.join(post_n_words))
                        if prev_spans:
                            new_start = prev_spans[0][0]
                        else:
                            new_start = ad_start
                        if post_spans:
                            new_end = post_spans[-1][-1]
                        else:
                            new_end = ad_end
                        a_sample = {'sample_type': 'addr',
                                    'start': new_start,
                                    'end': new_end,
                                    'line_seq': notempty_line_seq,
                                    'text': new_bow,
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
                notempty_line_seq += 1
                offset += len(line) + 1  # for eoln
            '''
        return samples, label_list, group_id_list
