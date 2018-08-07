import logging
from typing import Dict, List, Optional, Tuple

from kirke.utils import ebantdoc5, ebsentutils
from kirke.abbyyxml import tableutils

def find_prev_sechead(start: int,
                      sechead_list: List[Tuple[int, int, str, int]]) \
                      -> Optional[Tuple[int, int, str, int]]:
    print("find_prev_sechead, table_start = {}".format(start))
    prev_sechead_tuple = None
    for sechead_tuple in sechead_list:
        shead_start, unused_shead_end, unused_shead_st, unused_shead_page_num = sechead_tuple
        if start < shead_start:
            return prev_sechead_tuple
        prev_sechead_tuple = sechead_tuple
    return prev_sechead_tuple


def find_prev_exhibit_in_page(start: int,
                              page_num: int,
                              sechead_list: List[Tuple[int, int, str, int]]) \
                              -> Optional[Tuple[int, int, str, int]]:
    prev_exhibit_tuple = None
    for sechead_tuple in sechead_list:
        shead_start, unused_shead_end, shead_st, shead_page_num = sechead_tuple
        if shead_page_num == page_num and \
           'exhibit' in shead_st.lower():
            prev_exhibit_tuple = sechead_tuple
        if start < shead_start:
            return prev_exhibit_tuple
        if page_num < shead_page_num:
            return prev_exhibit_tuple
    return prev_exhibit_tuple


def is_in_exhibit_section(start: int,
                          page_num: int,
                          sechead_list: List[Tuple[int, int, str, int]]) \
                          -> Optional[Tuple[int, int, str, int]]:
    """Returning the sechead_tuple so that we know where the exhibit
       is, for debugging purpose"""
    sechead_tuple = find_prev_exhibit_in_page(start,
                                              page_num,
                                              sechead_list)
    if not sechead_tuple:
        sechead_tuple = find_prev_sechead(start,
                                          sechead_list)
        if sechead_tuple:
            unused_shead_start, unused_shead_end, shead_st, unused_shead_page_num = \
                sechead_tuple
            if 'exhibit' in shead_st:
                return sechead_tuple
    return sechead_tuple


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
            doc_len = len(doc_text)

            if group_id % 10 == 0:
                logging.info('TableGenerator.documents_to_candidates(), group_id = %d',
                             group_id)

            sechead_list = antdoc.sechead_list
            for sechead_count, xsechead_tuple in enumerate(sechead_list):
                print("== sechead #{}: {}".format(sechead_count, xsechead_tuple))

            for table_count, abbyy_table in enumerate(antdoc.abbyy_table_list):
                table_start, table_end = tableutils.get_pbox_text_offset(abbyy_table)

                table_text = doc_text[table_start:table_end]
                span_list = tableutils.get_pbox_text_span_list(abbyy_table, doc_text)


                print('\n\n==================================================')
                print('ABBYY table count #{}, page_num = {}'.format(table_count, abbyy_table.page_num))

                print("  is_abbyy_original: {}".format(abbyy_table.is_abbyy_original))
                print("  sechead: {}".format(find_prev_sechead(table_start, sechead_list)))
                print("  is_in_exhibit: {}".format(is_in_exhibit_section(table_start,
                                                                         abbyy_table.page_num,
                                                                         sechead_list)))
                print("  perc doc: {:.2f}%".format(100.0 * table_start / doc_len))

                for span_seq, (start, end) in enumerate(span_list):

                    print("  tablegen.span #{}, ({}, {}): [{}]".format(span_seq,
                                                                       start, end,
                                                                       doc_text[start:end]))




                span_dict_list = [{'start': start,
                                   'end': end} for start, end in span_list]
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
