import logging
import re
from typing import Dict, List, Tuple
from kirke.ebrules import dates, addresses
from kirke.utils import ebantdoc3, ebsentutils, strutils

class AddrContextGenerator:
    def __init__(self, num_prev_words: int, num_post_words: int) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words

    def find_constituencies(self, text, constituencies):
        s = ''
        text = text.replace ("\n", " ")
        for word in text.split():
            word = re.sub(r'[,\.]+$|\-', "", word)
            if word.isdigit() or word in constituencies:
                s += '1'
            else:
                s += '0'
        matches = re.finditer(r'(1+0?0?(1+0?0?)*1+)', s)
        all_spans = [match.span(1) for match in matches]
        ads = []
        for ad_start, ad_end in all_spans:
            list_address = text.split()[ad_start:ad_end]
            ad_st = " ".join(list_address) 
            address_prob = addresses.classify(ad_st)
            if address_prob >= 0.5 and len(text.split()[ad_start:ad_end]) > 3:
                pred_start,_ = re.search(list_address[0], text).span()
                _, pred_end = re.search(list_address[-1], text[pred_start:]).span()
                ads.append([pred_start, pred_end, text[pred_start:pred_end]])
        return ads

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
                                'post_n_words': ' '.join(post_n_words)}
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
                                    'post_n_words': ' '.join(post_n_words)}
                        samples.append(a_sample)
                        group_id_list.append(group_id)
                        if is_label:
                            a_sample['label_human'] = label
                            label_list.append(True)
                        else:
                            label_list.append(False)
                notempty_line_seq += 1
                #updates boolean list for whether candidate is pos for provision?
                offset += len(line) + 1  # for eoln
        return samples, label_list, group_id_list