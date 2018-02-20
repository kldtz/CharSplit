from typing import Dict, List, Tuple

class RegexContextGenerator:

    def __init__(self, num_prev_words: int, num_post_words: int, center_regex: String, sample_type: String) -> None:
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words
        self.center_regex = center_regex
        self.sample_type = sample_type

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
                logging.info("ContextGenerator.documents_to_samples(), group_id = {}".format(group_id))

            matches = re.finditer(re.escape(self.center_regex), nl_text, re.I)
            #finds all matches in the text and adds window around each as a candidate
            for match in matches:
                match_start, match_end = match.span()
                match_str = match.group()
                is_label = ebsentutils.check_start_end_overlap(match_start,
                                                               match_end,
                                                               label_ant_list)
                prev_n_words, prev_spans = strutils.get_lc_prev_n_words(nl_text, match_start, self.num_prev_words)
                post_n_words, post_spans = strutils.get_lc_post_n_words(nl_text, match_end, self.num_post_words)
                new_bow = '{} {} {}'.format(' '.join(prev_n_words), match_str, ' '.join(post_n_words))
                
                #update span based on window size
                new_start = match_start
                new_end = match_end
                if prev_spans:
                    new_start = prev_spans[0][0]
                if post_spans:
                    new_end = post_spans[-1][-1]
                
                a_sample = {'sample_type': self.sample_type,
                            'start': new_start,
                            'end': new_end,
                            'text': new_bow,
                            'match_start': match_start
                            'match_end': match_end
                            'prev_n_words': ' '.join(prev_n_words),
                            'post_n_words': ' '.join(post_n_words)}
                samples.append(a_sample)
                group_id_list.append(group_id)
                if is_label:
                    a_sample['label_human'] = label
                    label_list.append(True)
                else:
                    label_list.append(False)
        return samples, label_list, group_id_list
