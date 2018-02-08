import logging
import re, regex
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

        #replaces words with consistuency information
        for word in text.split():
            word = re.sub(r'[,\.]+$|\-', "", word)
            if word.isdigit() or word in constituencies:
                s += '1'
            else:
                s += '0'

        #matches based on sequence of constituencies
        matches = re.finditer(r'(1+0?0?(1+0?0?){1,3}1+)', s)
        all_spans = [match.span(1) for match in matches]
        ads = []

        #checks matches to eliminate things that aren't addresses
        for ad_start, ad_end in all_spans:
            list_address = text.split()[ad_start:ad_end]
            ad_st = " ".join(list_address) 
            address_prob = addresses.classify(ad_st)
            if address_prob >= 0.5 and len(text.split()[ad_start:ad_end]) > 3:
                ad_st = re.sub('[\(\.\)]', '', ad_st)

                #finds address in text so we can recover the exact indices
                for found in regex.finditer('(?e)(?:'+ad_st+'){e<=3}', text):
                    pred_start, pred_end = found.span()
                    ads_list = [pred_start, pred_end, text[pred_start:pred_end]]
                    if ads_list not in ads:
                        ads.append(ads_list)
        return ads

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
            all_keywords = addresses.all_constituencies()
            split_text = nl_text.split()

            #finds all addresses in the text and adds window around each as a candidate
            for addr in self.find_constituencies(nl_text, all_keywords):
                new_start, new_end, addr_st = addr
                is_label = ebsentutils.check_start_end_overlap(new_start,
                                                               new_end,
                                                               label_ant_list)
                prev_n_words, prev_spans = strutils.get_lc_prev_n_words(nl_text, new_start, self.num_prev_words)
                post_n_words, post_spans = strutils.get_lc_post_n_words(nl_text, new_end, self.num_post_words)
                new_bow = '{} {} {}'.format(' '.join(prev_n_words), addr_st, ' '.join(post_n_words))
                #update span based on window size
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
        
        return samples, label_list, group_id_list
