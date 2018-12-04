import logging
from typing import Dict, List, Optional, Tuple

from kirke.utils import ebantdoc4, ebsentutils
from kirke.docstruct import fromtomapper

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SentenceGenerator:

    def __init__(self,
                 candidate_type: str,
                 group_num: int = 1) -> None:
        self.candidate_type = candidate_type
        self.group_num = group_num

    # pylint: disable=too-many-arguments, unused-argument
    def get_candidates_from_ebantdoc(self,
                                     ebantdoc: ebantdoc4.EbAnnotatedDoc4,
                                     group_id: int = 0,
                                     # pylint: disable=line-too-long
                                     label_ant_list_param: Optional[List[ebsentutils.ProvisionAnnotation]] = None,
                                     label_list_param: Optional[List[bool]] = None,
                                     label: Optional[str] = None):

        candidates = [] # type: List[Dict]
        group_id_list = [] # type: List[int]

        # candidates are sentences from corenlp stored in the attrvec list
        for attrvec in ebantdoc.get_attrvec_list():
            match_start = attrvec.start
            match_end = attrvec.end
            match_str = ebantdoc.get_nlp_text()[match_start:match_end]

            a_candidate = {'candidate_type': self.candidate_type,
                           'bow_start': match_start,
                           'bow_end': match_end,
                           'text': attrvec.bag_of_words,
                           'start': match_start,
                           'end': match_end,
                           'prev_n_words': '',
                           'post_n_words': '',
                           'chars': match_str,
                           'attrvec': attrvec}
            candidates.append(a_candidate)
            group_id_list.append(group_id)

        return candidates, group_id_list, label_list_param

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
        for group_id, antdoc in enumerate(antdoc_list):  # these are ebantdoc4
            label_list = []   # type: List[bool]

            #creates list of ants for a specific provision
            ant_list = antdoc.prov_annotation_list
            label_ant_list = []  # type: List[ebsentutils.ProvisionAnnotation]
            for ant in ant_list:
                if ant.label == label:
                    label_ant_list.append(ant)

            if group_id % 10 == 0:
                logger.debug('SentenceGenerator.documents_to_candidates(), group_id = %d',
                             group_id)
            candidates, group_id_list, label_list = self.get_candidates_from_ebantdoc(antdoc,
                                                                                      group_id=group_id,
                                                                                      label_ant_list_param=label_ant_list,
                                                                                      label_list_param=label_list,
                                                                                      label=label)
            # map offsets from nlp_text to raw_text 
            fromto_mapper = fromtomapper.FromToMapper('an offset mapper',
                                                      antdoc.get_nlp_sx_lnpos_list(),
                                                      antdoc.get_origin_sx_lnpos_list())
            # this is an in-place modification
            fromto_mapper.adjust_fromto_offsets(candidates)

            for candidate in candidates:
                is_label = ebsentutils.check_start_end_overlap(candidate['start'],
                                                               candidate['end'],
                                                               label_ant_list)
                if is_label:
                    candidate['label_human'] = label
                    label_list.append(True)
                else:
                    label_list.append(False)
            result.append((antdoc, candidates, label_list, group_id_list))
        return result
