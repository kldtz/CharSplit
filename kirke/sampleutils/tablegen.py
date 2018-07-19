import logging
from operator import itemgetter
import re
from typing import Dict, List, Pattern, Tuple

from kirke.utils import ebantdoc5, ebsentutils, strutils
from kirke.abbyxml import tableutils


# pylint: disable=too-few-public-methods
class TableGenerator:

    def __init__(self,
                 candidate_type: str) -> None:
        self.candidate_type = candidate_type

    # pylint: disable=too-many-locals
    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc5.EbAnnotatedDoc],
                                label: str) \
                                -> List[Tuple[ebantdoc5.EbAnnotatedDoc,
                                              List[Dict],
                                              List[bool],
                                              List[int]]]:

        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc5.EbAnnotatedDoc, List[Dict], List[bool], List[int]]]
        for group_id, antdoc in enumerate(antdoc_list):
            candidates = []  # type: List[Dict]
            label_list = []   # type: List[bool]
            group_id_list = []  # type: List[int]

            # creates list of ants for a specific provision
            ant_list = antdoc.prov_annotation_list
            label_ant_list = []
            for ant in ant_list:
                if ant.label == label:
                    label_ant_list.append(ant)
            doc_text = antdoc.get_text()

            if group_id % 10 == 0:
                logging.info('TableGenerator.documents_to_candidates(), group_id = %d',
                             group_id)
                
            for abby_table in antdoc.abby_table_list:
                table_start, table_end = tableutils.get_pbox_text_offset(abby_table)

                table_text = doc_text[table_start:table_end]
                span_list = tableutils.get_pbox_text_span_list(abby_table, doc_text)

                for span_seq, (start, end) in enumerate(span_list):
                    print("  tablegen.span #{}, ({}, {}): [{}]".format(span_seq,
                                                                       start, end,
                                                                       doc_text[start:end]))

                span_dict_list = [{'start': start,
                                   'end': end} for start,end in span_list]
                # looks ahead to see if it should merge the next paragraph

                is_label = ebsentutils.check_start_end_overlap(table_start,
                                                               table_end,
                                                               label_ant_list)
                a_candidate = {'candidate_type': self.candidate_type,
                               'text': table_text,
                               'start': table_start,
                               'end': table_end,
                               'span_list':span_dict_list}
                candidates.append(a_candidate)
                group_id_list.append(group_id)
                if is_label:
                    a_candidate['label_human'] = label
                    label_list.append(True)
                else:
                    label_list.append(False)
            result.append((antdoc, candidates, label_list, group_id_list))
        return result
