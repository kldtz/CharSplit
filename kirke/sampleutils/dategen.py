
from kirke.ebrules import dates
from kirke.utils import ebantdoc3, ebsentutils, strutils

    
class DateSpanGenerator:
    
    def __init__(self, num_prev_words, num_post_words):
        self.num_prev_words = num_prev_words
        self.num_post_words = num_post_words

    def documents_to_samples(self,
                             antdoc_list,
                             label=None):
        samples = []  # dict
        group_id_list = []  # int
        label_list = []   # booleans

        # each sample is the date regex +
        for group_id, antdoc in enumerate(antdoc_list):  # these are ebantdoc3

            ant_list = antdoc.prov_annotation_list
            label_ant_list = []
            for ant in ant_list:
                if ant.label == label:
                    label_ant_list.append(ant)

            if antdoc.doc_format in set([ebantdoc3.EbDocFormat.html,
                                         ebantdoc3.EbDocFormat.html_nodocstruct,
                                         ebantdoc3.EbDocFormat.other]):
                nl_text = antdoc.text
            else:
                nl_text = antdoc.nl_text

            if group_id % 10 == 0:
                print("DateSpanGenerator.documents_to_samples(), group_id = {}".format(group_id))

            lines = nl_text.split('\n')
            offset = 0
            for line in lines:
                date_se_list = dates.extract_std_dates(line)
                if date_se_list:
                    for start, end in date_se_list:
                        start = offset + start
                        end = offset + end

                        prev_n_words = strutils.get_lc_prev_n_words(nl_text, start, self.num_prev_words)
                        post_n_words = strutils.get_lc_post_n_words(nl_text, end, self.num_post_words)

                        is_label = ebsentutils.check_start_end_overlap(start,
                                                                       end,
                                                                       label_ant_list)
                        a_sample = {'sample_type': 'date',
                                    'start': start,
                                    'end': end,
                                    'text': nl_text[start:end],
                                    'prev_n_words': ' '.join(prev_n_words),
                                    'post_n_words': ' '.join(post_n_words)}

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
        
        # the data is fake right now
        return samples, label_list, group_id_list
