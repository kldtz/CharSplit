import logging
import re
from typing import Dict, List, Optional, Tuple

from kirke.utils import ebantdoc4, ebsentutils


# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# pylint: disable=too-few-public-methods
class ParagraphGenerator:

    def __init__(self,
                 candidate_type: str) -> None:
        self.candidate_type = candidate_type

    # pylint: disable=too-many-arguments, too-many-locals
    def get_candidates_from_ebantdoc(self,
                                     antdoc: ebantdoc4.EbAnnotatedDoc4,
                                     group_id: int = 0,
                                     # pylint: disable=line-too-long
                                     label_ant_list_param: Optional[List[ebsentutils.ProvisionAnnotation]] = None,
                                     label_list_param: Optional[List[bool]] = None,
                                     label: Optional[str] = None):

        # pylint: disable=line-too-long
        label_ant_list, label_list = [], []  # type: List[ebsentutils.ProvisionAnnotation], List[bool]
        if label_ant_list_param is not None:
            label_ant_list = label_ant_list_param
        if label_list_param is not None:
            label_list = label_list_param

        candidates = []  # type: List[Dict]
        group_id_list = []  # type: List[int]

        nl_text = antdoc.get_nlp_text()

        #finds all matches in the text and adds window around each as a candidate
        i = 0
        #sorted_paras = sorted(antdoc.para_indices, key=lambda x: x[0][0].start)
        while i < len(antdoc.para_indices):
            para = antdoc.para_indices[i]
            match_start = para[0][1].start
            match_end = para[-1][1].end
            raw_start = para[0][0].start
            raw_end = para[-1][0].end
            para_text = nl_text[match_start:match_end].strip()
            skipping = True
            span_list = [x[0] for x in para]
            while skipping and i+1 < len(antdoc.para_indices):
                para_next = antdoc.para_indices[i+1]
                next_start = para_next[0][1].start
                next_end = para_next[-1][1].end
                next_raw_end = para_next[-1][0].end
                next_text = nl_text[next_start:next_end].strip()
                para_end_punct = re.search(r'[;:]', para_text[-10:])
                next_end_punct = re.search(r'[;\.A-z]', next_text[-10:])
                preamble = re.search(r'(now,? +therefore)|(definitions)', para_text[:50], re.I)
                if not preamble:
                    preamble = re.search(r'(as +follows[:;])|(defined +terms)', para_text, re.I)
                try:
                    # the try-except will take care of the None case
                    # pylint: disable=line-too-long
                    next_start_punct = re.search(r'(\([A-z]+\) *)?([A-z])', next_text).group()[-1].islower()  # type: ignore
                except AttributeError:
                    next_start_punct = False
                if ((not preamble) and para_end_punct and next_end_punct and len(para_text.split()) > 1) or (not next_text) or next_start_punct:
                    if next_raw_end > raw_end:
                        para_text = para_text + " " + next_text
                        match_end = next_end
                        raw_end = next_raw_end
                        span_list.extend([x[0] for x in antdoc.para_indices[i+1]])
                    i += 1
                else:
                    skipping = False
            if len(para_text.split()) > 5 and raw_end > raw_start:
                is_label = ebsentutils.check_start_end_overlap(raw_start,
                                                               raw_end,
                                                               label_ant_list)
                #update span based on window size
                a_candidate = {'candidate_type': self.candidate_type,
                               'bow_start': match_start,
                               'bow_end': match_end,
                               'text': para_text,
                               'start': raw_start,
                               'end': raw_end,
                               'unused_span_list':span_list}
                candidates.append(a_candidate)
                group_id_list.append(group_id)
                if is_label:
                    a_candidate['label_human'] = label
                    label_list.append(True)
                else:
                    label_list.append(False)
            i += 1
        return candidates, group_id_list, label_list

    # pylint: disable=too-many-locals, too-many-statements
    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                label: str = None)  -> List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                                                  List[Dict],
                                                                  List[bool],
                                                                  List[int]]]:

        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):
            label_list = []   # type: List[bool]

            #creates list of ants for a specific provision
            ant_list = antdoc.prov_annotation_list
            label_ant_list = []
            for ant in ant_list:
                if ant.label == label:
                    label_ant_list.append(ant)

            if group_id % 10 == 0:
                logger.info('ContextGenerator.documents_to_candidates(), group_id = %d',
                            group_id)
            candidates, group_id_list, label_list = self.get_candidates_from_ebantdoc(antdoc,
                                                                                      group_id=group_id,
                                                                                      label_ant_list_param=label_ant_list,
                                                                                      label_list_param=label_list,
                                                                                      label=label)
            result.append((antdoc, candidates, label_list, group_id_list))
        return result