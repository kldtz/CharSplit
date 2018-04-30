import logging
import re
from typing import Dict, List, Pattern, Tuple
from operator import itemgetter
from itertools import groupby
from kirke.utils import ebantdoc4, ebsentutils, strutils

# pylint: disable=too-few-public-methods
class SectionGenerator:

    def __init__(self,
                 candidate_type: str) -> None:
        self.candidate_type = candidate_type

    # pylint: disable=too-many-locals
    def documents_to_candidates(self,
                             antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                             label: str = None)  -> List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                                               List[Dict],
                                                               List[bool],
                                                               List[int]]]:

        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):
            print(">>>>", antdoc.file_id, ">>>>")
            candidates = []  # type: List[Dict]
            label_list = []   # type: List[bool]
            group_id_list = []  # type: List[int]

            #creates list of ants for a specific provision
            ant_list = antdoc.prov_annotation_list
            label_ant_list = []
            for ant in ant_list:
                if ant.label == label:
                    label_ant_list.append(ant)

            nl_text = antdoc.get_nlp_text()

            if group_id % 10 == 0:
                logging.info('ContextGenerator.documents_to_candidates(), group_id = %d',
                             group_id)
            #finds all matches in the text and adds window around each as a candidate
            all_indices = antdoc.para_indices 
            all_secheads = antdoc.sections
            try:
                current_sechead = all_secheads[0][0][2]
            except:
                continue
            candidate_text = nl_text[all_indices[0][0][1].start:all_indices[0][-1][1].end]
            candidate_start = all_indices[0][0][1].start
            candidate_end = all_indices[0][0][1].end
            raw_start = all_indices[0][0][0].start
            raw_end = all_indices[0][0][0].end
            for i in range(1, len(all_indices)):
                if all_secheads[i]:
                    sechead = all_secheads[i][0][2]
                else:
                    sechead = current_sechead
                match_start = all_indices[i][0][1].start
                match_end = all_indices[i][0][1].end
                raw_indices = all_indices[i][0][0]
                para_text = nl_text[match_start:match_end].strip()
                #print(sechead, '///', current_sechead)
                if (sechead != current_sechead) or (i == len(all_indices)-1):
                    #print("\t", candidate_text)
                    if len(candidate_text.split()) > 10:
                        is_label = ebsentutils.check_start_end_overlap(raw_start,
                                                                       raw_end,
                                                                       label_ant_list)
                        #update span based on window size
                        a_candidate = {'candidate_type': self.candidate_type,
                                       'bow_start': candidate_start,
                                       'bow_end': candidate_end,
                                       'text': candidate_text,
                                       'start': raw_start,
                                       'end': raw_end,
                                       'section': sechead}
                        candidates.append(a_candidate)
                        group_id_list.append(group_id)
                        if is_label:
                            a_candidate['label_human'] = label
                            label_list.append(True)
                        else:
                            label_list.append(False)
                        print("<<<<", candidate_text, "<<<<")
                    candidate_text = para_text
                    candidate_start = match_start
                    candidate_end = match_end
                    current_sechead = sechead
                    raw_start = raw_indices.start
                    raw_end = raw_indices.end
                else:
                    candidate_text = candidate_text + " " + para_text
                    candidate_end = match_end
                    raw_end = raw_indices.end
            result.append((antdoc, candidates, label_list, group_id_list))
        return result
