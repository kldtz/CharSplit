import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from kirke.docstruct.secheadutils import SecHeadTuple
from kirke.utils import ebantdoc4, engutils, ebsentutils, strutils
from kirke.abbyyxml import tableutils
from kirke.abbyyxml.pdfoffsets import AbbyyTableBlock

# on 2018/10/03, the precision and recall number for
# data-rate-table, 8 documents (no goldwest.txt)
# are fp=7, fn=1, tp=8
# precision=0.53, recall=0.89, f=0.67
# I am not putting this into unit test suite yet,
# because table extraction is still unstable.

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IS_DEBUG_TABLE = True
IS_DEBUG_INVALID_TABLE = False

def find_prev_sechead(start: int,
                      sechead_list: List[SecHeadTuple]) \
                      -> Optional[SecHeadTuple]:
    # print("find_prev_sechead, table_start = {}".format(start))
    prev_sechead_tuple = None
    for sechead_tuple in sechead_list:
        shead_start, unused_shead_end, unused_shead_prefix, \
            unused_shead_st, unused_shead_page_num = sechead_tuple
        if start <= shead_start:
            return prev_sechead_tuple
        prev_sechead_tuple = sechead_tuple
    return prev_sechead_tuple


def find_prev_exhibit_in_page(start: int,
                              page_num: int,
                              sechead_list: List[SecHeadTuple]) \
                              -> Optional[SecHeadTuple]:
    prev_exhibit_tuple = None
    for sechead_tuple in sechead_list:
        shead_start, unused_shead_end, unused_shead_prefix, \
            shead_st, shead_page_num = sechead_tuple
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
                          sechead_list: List[SecHeadTuple]) \
                          -> Optional[SecHeadTuple]:
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
        unused_shead_start, unused_shead_end, shead_prefix, \
            shead_st, unused_shead_page_num = sechead_tuple
        if 'exhibit' in shead_st.lower() or \
           'exhibit' in shead_prefix.lower():
            return sechead_tuple

    return None

def get_before_table_text(table_start: int,
                          sechead_tuple: Optional[SecHeadTuple],
                          exhibit_tuple: Optional[SecHeadTuple],
                          prev_table_end: int,
                          doc_text: str) \
                          -> Tuple[str, Tuple[int, int]]:
    if sechead_tuple and exhibit_tuple:
        if sechead_tuple.end <= exhibit_tuple.end:
            last_tuple = exhibit_tuple
        else:
            last_tuple = sechead_tuple
    elif sechead_tuple:
        last_tuple = sechead_tuple
    elif exhibit_tuple:
        last_tuple = exhibit_tuple
    else:
        return '', (-1, -1)

    unused_shead_start, shead_end, unused_shead_prefix, \
        unused_shead_st, unused_shead_page_num = last_tuple

    if prev_table_end != -1 and \
       prev_table_end > shead_end:
        shead_end = prev_table_end

    # only up to 1000 char are returned
    # print('  before table text len = {}'.format(table_start - shead_end))
    if table_start > shead_end and table_start - shead_end < 1000:
        text = doc_text[shead_end:table_start].replace('\n', ' ').strip()

        shead_end, table_start = strutils.text_strip(start=shead_end,
                                                     end=table_start,
                                                     text=doc_text)
        return text, (shead_end, table_start)

    return '', (-1, -1)


def is_invalid_table(table_candidate: Dict,
                     # pylint: disable=unused-argument
                     ab_table_block: AbbyyTableBlock) -> bool:
    """Return False if a table is not valid.

    This is another check on tables, similar to abbyyxml.tableutils.is_invalid_table().
    Here we use more content-based information from the table, such as
    the words.
    """
    # reject really bad table here
    if table_candidate['num_word'] >= 10 and \
       table_candidate['perc_bad_word'] >= 0.75:
        if IS_DEBUG_INVALID_TABLE:
            print("tablegen.py, table rejected because too many weird words")
        return True

    return False


# pylint: disable=too-few-public-methods
class TableGenerator:

    def __init__(self,
                 candidate_type: str) -> None:
        self.candidate_type = candidate_type

    # pylint: disable=too-many-locals, too-many-statements
    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                label: str) \
                                -> List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                              List[Dict],
                                              List[bool],
                                              List[int]]]:

        # pylint: disable=line-too-long
        result = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
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
                logger.info('TableGenerator.documents_to_candidates(), group_id = %d',
                            group_id)

            sechead_list = antdoc.sechead_list
            if IS_DEBUG_TABLE:
                for sechead_count, xsechead_tuple in enumerate(sechead_list):
                    print("== sechead #{}: {}".format(sechead_count, xsechead_tuple))

            doc_abbyy_table_list = []  # type: List[AbbyyTableBlock]
            doc_table_cand_list = []  # type: List[Dict]

            invalid_tables = list(antdoc.invalid_table_list)  # type: List[AbbyyTableBlock]

            # a section might have multiple tables, so the pretext should start from
            # either the sechead or the last table_end
            prev_table_end = -1
            for table_count, pbox_table in enumerate(antdoc.pbox_table_list):

                span_list = pbox_table.span_list
                # assume only 1 table per page
                abbyy_table = pbox_table.abbyy_table_list[0]
                table_start, table_end = span_list[0][0], span_list[-1][1]

                table_page_num, bot_left_point, top_right_point = pbox_table.bltr_list[0]

                left_x, bot_y = bot_left_point
                right_x, top_y = top_right_point

                table_text = pbox_table.abbyy_text
                table_text = fix_rate_table_text(table_text)

                if IS_DEBUG_TABLE:
                    print('\n\n==================================================')
                    print("-------- table txt:")
                    print(table_text.replace('\n', ' || '))

                # rate table related features
                num_number, num_currency, num_percent, \
                    num_phone_number, num_date, num_alpha_word, \
                    num_alphanum_word, num_bad_word, num_word, \
                    table_text_alphanum = \
                        strutils.remove_number_types(table_text)
                has_number = num_number > 0
                has_currency = num_currency > 0
                has_percent = num_percent > 0
                has_phone_number = num_phone_number > 0
                has_date = num_date > 0

                is_num_alpha_word_le10 = num_alpha_word <= 10
                is_num_alpha_word_le20 = num_alpha_word <= 20
                num_word_div_100 = num_word / 100.0
                num_alpha_word_div_100 = num_alpha_word / 100

                perc_number_word = num_number / num_word
                perc_currency_word = num_currency / num_word
                perc_percent_word = num_percent / num_word
                perc_phone_word = num_phone_number / num_word
                perc_date_word = num_date / num_word
                perc_alpha_word = num_alpha_word / num_word
                perc_alphanum_word = num_alphanum_word / num_word
                perc_bad_word = (num_bad_word + num_alphanum_word) / num_word

                infer_attr_dict = abbyy_table.infer_attr_dict
                is_header, is_footer = False, False
                if infer_attr_dict.get('header'):
                    is_header = True
                if infer_attr_dict.get('footer'):
                    is_footer = True

                if IS_DEBUG_TABLE:
                    print()
                    print('ABBYY table count #{}, page_num = {}, table_start = {}'.format(table_count,
                                                                                          table_page_num,
                                                                                          table_start))

                    print("  is_abbyy_original: {}".format(abbyy_table.is_abbyy_original))

                table_sechead = find_prev_sechead(table_start, sechead_list)
                sechead_text = ''
                out_sechead_dict = {}
                if table_sechead:
                    if table_sechead.head_st:
                        sechead_text = table_sechead.head_st
                    else:
                        # will take prefix is no head_st found
                        sechead_text = table_sechead.head_prefix
                    out_sechead_dict['start'] = table_sechead.start
                    out_sechead_dict['end'] = table_sechead.end
                    out_sechead_dict['text'] = sechead_text

                lc_sechead_text = sechead_text.lower()
                # remove useless and confusing secheads
                if 'appendix' in lc_sechead_text or \
                   'section' in lc_sechead_text or \
                   'exhibit' in lc_sechead_text or \
                   'article' in lc_sechead_text:
                    sechead_text = ''

                is_table_in_exhibit = is_in_exhibit_section(table_start,
                                                            table_page_num,
                                                            sechead_list)
                doc_percent = table_start / doc_len
                pre_table_text, pre_table_se_tuple = get_before_table_text(table_start,
                                                                           table_sechead,
                                                                           is_table_in_exhibit,
                                                                           prev_table_end,
                                                                           doc_text)
                pre_table_text_dict = {}
                if pre_table_text:
                    pre_table_text_dict['text'] = pre_table_text
                    pre_table_text_dict['start'] = pre_table_se_tuple[0]
                    pre_table_text_dict['end'] = pre_table_se_tuple[1]
                len_pre_table_text = len(pre_table_text)
                num_rows = len(abbyy_table.ab_rows)
                num_cols = abbyy_table.get_num_cols()
                row_header_text = abbyy_table.get_row(0).get_text()

                # approximate number of sentences
                num_period_cap = engutils.num_letter_period_cap(table_text)
                has_dollar_div = re.search(r'\$\s*[\d,\.]*\/[A-Z][a-zA-Z]+', table_text) != None

                if IS_DEBUG_TABLE:
                    print("  table_text_alphanum: [{}]".format(table_text_alphanum))
                    print("  sechead: [{}]".format(table_sechead))
                    print("  row header_text: [{}]".format(row_header_text.replace('\n', '|')))
                    print("  is_in_exhibit: {}".format(is_table_in_exhibit))
                    print('  before_table text: [{}]'.format(pre_table_text))
                    print("  doc_percent: {:.2f}%".format(100.0 * doc_percent))
                    print("  num_number: {}".format(num_number))
                    print("  num_currency: {}".format(num_currency))
                    print("  num_percent: {}".format(num_percent))
                    print("  num_phone_number: {}".format(num_phone_number))
                    print("  num_date: {}".format(num_date))
                    print("  num_alpha_word: {}".format(num_alpha_word))
                    print("  num_alphanum_word: {}".format(num_alphanum_word))
                    print("  num_bad_word: {}".format(num_bad_word))
                    print("  num_word: {}".format(num_word))
                    print("  num_word_div_100: {}".format(min(num_word_div_100, 1.0)))
                    print("  is_num_alpha_word_le_10: {}".format(is_num_alpha_word_le10))
                    print("  is_num_alpha_word_le_20: {}".format(is_num_alpha_word_le20))
                    print("  num_alpha_word_div_100: {}".format(min(num_alpha_word_div_100, 1.0)))
                    print("  perc_number_word: {}".format(perc_number_word))
                    print("  perc_currency_word: {}".format(perc_currency_word))
                    print("  perc_percent_word: {}".format(perc_percent_word))
                    print("  perc_phone_word: {}".format(perc_phone_word))
                    print("  perc_date_word: {}".format(perc_date_word))
                    print("  perc_alpha_word: {}".format(perc_alpha_word))
                    print("  perc_alphanum_word: {}".format(perc_alphanum_word))
                    print("  perc_bad_word: {}".format(perc_bad_word))
                    print("  len_before_table_text: {}".format(len_pre_table_text))
                    print("  num_rows: {}".format(num_rows))
                    print("  num_cols: {}".format(num_cols))
                    print("  num_period_cap: {}".format(num_period_cap))
                    print("  has_dollar_div: {}".format(has_dollar_div))
                    row_header_text2 = fix_rate_table_text(row_header_text)
                    if row_header_text2 != row_header_text:
                        print("  fixed_row header_text: {}".format(row_header_text2))

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

                out_table_json = tableutils.table_block_to_json(abbyy_table)

                out_table_json['start'] = table_start
                out_table_json['end'] = table_end
                out_table_json['x_left'] = left_x
                out_table_json['x_right'] = right_x
                out_table_json['y_top'] = top_y
                out_table_json['y_bottom'] = bot_y
                out_table_json['span_list'] = span_dict_list
                if out_sechead_dict:
                    out_table_json['section_head'] = out_sechead_dict['text']

                if pre_table_text_dict:
                    out_table_json['pre_table_text'] = pre_table_text_dict['text']

                a_candidate = {'candidate_type': self.candidate_type,
                               'text': '\n'.join([sechead_text,
                                                  table_text]),
                               'text_alphanum': '\n'.join([sechead_text,
                                                           table_text_alphanum]),
                               'start': table_start,
                               'end': table_end,
                               # 'left_x': left_x,
                               # 'bottom_y': bot_y,
                               # 'right_x': right_x,
                               # 'top_y': top_y,
                               'span_list': span_dict_list,
                               'pre_table_text': pre_table_text,
                               'len_pre_table_text': len_pre_table_text,
                               'doc_percent': doc_percent,
                               'is_abbyy_original': abbyy_table.is_abbyy_original,
                               'is_in_exhibit': is_table_in_exhibit,
                               'sechead_text': sechead_text,
                               'num_number': num_number,
                               'num_currency': num_currency,
                               'num_percent': num_percent,
                               'num_phone_number': num_phone_number,
                               'num_date': num_date,
                               'has_number': has_number,
                               'has_currency': has_currency,
                               'has_percent': has_percent,
                               'has_phone_number': has_phone_number,
                               'has_date': has_date,
                               'num_word': num_word,
                               'num_alpha_word': num_alpha_word,
                               'num_alphanum_word': num_alphanum_word,
                               'num_bad_word': num_bad_word,
                               'is_num_alpha_word_le10': is_num_alpha_word_le10,
                               'is_num_alpha_word_le20': is_num_alpha_word_le20,
                               'num_word_div_100': num_word_div_100,
                               'num_alpha_word_div_100': num_alpha_word_div_100,
                               'perc_number_word': perc_number_word,
                               'perc_currency_word': perc_currency_word,
                               'perc_percent_word': perc_percent_word,
                               'perc_phone_word': perc_phone_word,
                               'perc_date_word': perc_date_word,
                               'perc_alpha_word': perc_alpha_word,
                               'perc_alphanum_word': perc_alphanum_word,
                               'perc_bad_word': perc_bad_word,
                               'num_rows': num_rows,
                               'num_cols': num_cols,
                               'num_period_cap': num_period_cap,
                               'has_dollar_div': has_dollar_div,
                               'row_header_text': row_header_text,
                               'json': out_table_json}

                if is_header:
                    a_candidate['is_header'] = True
                    a_candidate['json']['is_header'] = True
                if is_footer:
                    a_candidate['is_footer'] = True
                    a_candidate['json']['is_footer'] = True

                # Verify that thee table is valid
                # "not is_label" make sure that is a table is annotated
                # for a provision, it is kept regardless of is_valid or not.
                if not is_label and is_invalid_table(a_candidate, abbyy_table):
                    invalid_tables.append(abbyy_table)
                    continue

                candidates.append(a_candidate)
                group_id_list.append(group_id)
                if is_label:
                    a_candidate['label_human'] = label
                    label_list.append(True)
                else:
                    label_list.append(False)

                prev_table_end = table_end

                doc_abbyy_table_list.append(abbyy_table)
                doc_table_cand_list.append(a_candidate)


            if IS_DEBUG_TABLE:
                tableutils.save_cand_tables_to_html_file(antdoc.file_id,
                                                         doc_table_cand_list,
                                                         extension='.table.html')

                tableutils.save_tables_to_html_file(antdoc.file_id,
                                                    invalid_tables,
                                                    extension='.invalid.table.html')

                tableutils.save_tables_to_html_file(antdoc.file_id,
                                                    antdoc.abbyy_signature_list,
                                                    extension='.sign.html')

                tableutils.save_tables_to_html_file(antdoc.file_id,
                                                    antdoc.abbyy_address_list,
                                                    extension='.addr.html')

            result.append((antdoc, candidates, label_list, group_id_list))
        return result


def fix_rate_table_text(text: str) -> str:
    text = re.sub(r'\b(r)\s+(ate)\b', r'\1\2', text, flags=re.I)
    return text

# pylint: disable=pointless-string-statement
r"""
def fix_rate_table_text(text: str) -> str:
    text = text.replace('$', ' dollar_symbol ')
    # fix a missspelling
    text = re.sub(r'\b(r)\s+(ate)\b', r'\1\2', text, flags=re.I)
    return text
"""
