import logging
from typing import Dict, List, Optional, Tuple

from kirke.utils import ebantdoc5, ebsentutils, strutils
from kirke.abbyyxml import tableutils

def find_prev_sechead(start: int,
                      sechead_list: List[Tuple[int, int, str, int]]) \
                      -> Optional[Tuple[int, int, str, int]]:
    # print("find_prev_sechead, table_start = {}".format(start))
    prev_sechead_tuple = None
    for sechead_tuple in sechead_list:
        shead_start, unused_shead_end, unused_shead_st, unused_shead_page_num = sechead_tuple
        if start <= shead_start:
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
        if start <= shead_start:
            return prev_exhibit_tuple
        if shead_page_num == page_num and \
           'exhibit' in shead_st.lower():
            prev_exhibit_tuple = sechead_tuple
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
    if sechead_tuple:
        return sechead_tuple

    # no previous 'exhibit' sechead found
    sechead_tuple = find_prev_sechead(start,
                                      sechead_list)
    if sechead_tuple:
        unused_shead_start, unused_shead_end, shead_st, unused_shead_page_num = \
            sechead_tuple
        if 'exhibit' in shead_st:
            return sechead_tuple

    return None

def get_before_table_text(table_start: int,
                          sechead_tuple: Optional[Tuple[int, int, str, int]],
                          exhibit_tuple: Optional[Tuple[int, int, str, int]],
                          prev_table_end: int,
                          doc_text: str) -> str:
    if sechead_tuple and exhibit_tuple:
        if sechead_tuple[1] <= exhibit_tuple[1]:
            last_tuple = exhibit_tuple
        else:
            last_tuple = sechead_tuple
    elif sechead_tuple:
        last_tuple = sechead_tuple
    elif exhibit_tuple:
        last_tuple = exhibit_tuple
    else:
        return ''

    unused_shead_start, shead_end, unused_shead_st, unused_shead_page_num = \
        last_tuple

    if prev_table_end != -1 and \
       prev_table_end > shead_end:
        shead_end = prev_table_end

    # only up to 1000 char are returned
    print('  before table text len = {}'.format(table_start - shead_end))
    if table_start > shead_end and table_start - shead_end < 1000:
        text = doc_text[shead_end:table_start].replace('\n', ' ')
        return text

    return ''


# pylint: disable=too-few-public-methods
class TableGenerator:

    def __init__(self,
                 candidate_type: str) -> None:
        self.candidate_type = candidate_type

    # pylint: disable=too-many-locals, too-many-statements
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

            # a section might have multiple tables, so the pretext should start from
            # either the sechead or the last table_end
            prev_table_end = -1
            for table_count, abbyy_table in enumerate(antdoc.abbyy_table_list):
                table_start, table_end = tableutils.get_pbox_text_offset(abbyy_table)

                table_text = doc_text[table_start:table_end].strip()
                span_list = tableutils.get_pbox_text_span_list(abbyy_table, doc_text)

                # rate table related features
                num_currency = table_text.count('$')
                has_currency = num_currency > 0
                num_number = strutils.count_numbers(table_text)
                has_number = num_number > 0
                num_word = len(table_text.split())
                num_nonnum_word = max(num_word - num_number, 0)
                is_num_nonnum_word_le10 = num_nonnum_word <= 10
                is_num_nonnum_word_le20 = num_nonnum_word <= 20
                num_word_div_100 = num_word / 100.0
                num_nonnum_word_div_100 = num_nonnum_word / 100
                perc_number_word = num_number / num_word

                print('\n\n==================================================')
                print('ABBYY table count #{}, page_num = {}, table_start = {}'.format(table_count,
                                                                                      abbyy_table.page_num,
                                                                                      table_start))

                print("  is_abbyy_original: {}".format(abbyy_table.is_abbyy_original))
                table_sechead = find_prev_sechead(table_start, sechead_list)
                sechead_text = ''
                if table_sechead:
                    sechead_text = table_sechead[2]

                is_table_in_exhibit = is_in_exhibit_section(table_start,
                                                            abbyy_table.page_num,
                                                            sechead_list)
                doc_percent = table_start / doc_len
                pre_table_text = get_before_table_text(table_start,
                                                       table_sechead,
                                                       is_table_in_exhibit,
                                                       prev_table_end,
                                                       doc_text).strip()
                len_pre_table_text = len(pre_table_text)
                num_rows = len(abbyy_table.ab_rows)
                num_cols = abbyy_table.get_num_cols()

                print("  sechead: {}".format(table_sechead))
                print("  is_in_exhibit: {}".format(is_table_in_exhibit))
                print('  before_table text: [{}]'.format(pre_table_text))
                print("  doc_percent: {:.2f}%".format(100.0 * doc_percent))
                print("  num_number: {}".format(num_number))
                print("  num_currency: {}".format(num_currency))
                print("  num_word: {}".format(num_word))
                print("  num_word_div_100: {}".format(min(num_word_div_100, 1.0)))
                print("  num_nonnum_word: {}".format(num_nonnum_word))
                print("  is_num_nonnum_word_le_10: {}".format(is_num_nonnum_word_le10))
                print("  is_num_nonnum_word_le_10: {}".format(is_num_nonnum_word_le20))
                print("  num_nonnum_word_div_100: {}".format(min(num_nonnum_word_div_100, 1.0)))
                print("  perc_num_word: {}".format(perc_number_word))
                print("  len_before_table_text: {}".format(len_pre_table_text))
                print("  num_rows: {}".format(num_rows))
                print("  num_cols: {}".format(num_cols))

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
                               'span_list': span_dict_list,
                               'pre_table_text': pre_table_text,
                               'len_pre_table_text': len_pre_table_text,
                               'doc_percent': doc_percent,
                               'is_abbyy_original': abbyy_table.is_abbyy_original,
                               'is_in_exhibit': is_table_in_exhibit,
                               'sechead_text': sechead_text,
                               'num_word': num_word,
                               'num_currency': num_currency,
                               'num_number': num_number,
                               'has_currency': has_currency,
                               'has_number': has_number,
                               'num_nonnum_word': num_nonnum_word,
                               'is_num_nonnum_word_le10': is_num_nonnum_word_le10,
                               'is_num_nonnum_word_le20': is_num_nonnum_word_le20,
                               'num_word_div_100': num_word_div_100,
                               'num_nonnum_word_div_100': num_nonnum_word_div_100,
                               'perc_number_word': perc_number_word,
                               'num_rows': num_rows,
                               'num_cols': num_cols}
                candidates.append(a_candidate)
                group_id_list.append(group_id)
                if is_label:
                    a_candidate['label_human'] = label
                    label_list.append(True)
                else:
                    label_list.append(False)

                prev_table_end = table_end

            result.append((antdoc, candidates, label_list, group_id_list))
        return result
