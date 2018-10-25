import re
from typing import List, Optional, Tuple

from kirke.docstruct import linepos, partyutils, secheadutils
from kirke.docstruct.docutils import PLineAttrs
from kirke.utils import engutils

# pylint: disable=line-too-long
WITNESS_PAT = re.compile(r'(w i t n e s s e t h|witnesseth|recitals?\:?|r e c i t a l( s)?(\s*\:)?)',
                         re.IGNORECASE)
# 'background statement'
WHEREAS_PAT = re.compile(r'^\s*(whereas|background)', re.IGNORECASE)


def mark_attrs(para_attr_list: List[Tuple[str, List[str]]],
               begin_idx: int,
               end_idx: int,
               attr: str,
               ignore_if_tag: Optional[str] = None):
    if end_idx <= begin_idx:
        return
    for i, (line, attr_list) in enumerate(para_attr_list):
        if i >= begin_idx and i < end_idx:
            if line and (not ignore_if_tag or (ignore_if_tag and not ignore_if_tag in attr_list)):
                attr_list.append(attr)
        elif i >= end_idx:
            return

def mark_title_attrs(para_attr_list: List[Tuple[str, List[str]]],
                     begin_idx: int,
                     end_idx: int,
                     lc_party_line: str):
    if end_idx <= begin_idx:
        return
    for i, (line, attr_list) in enumerate(para_attr_list):
        if i >= begin_idx and i < end_idx:
            if line and re.sub(r'\s+', ' ', line.lower()) in re.sub(r'\s+', ' ', lc_party_line):
                attr_list.append('title')
        elif i >= end_idx:
            return


# already found 'toc' marker
def mark_toc_aux(para_attr_list: List[Tuple[str, List[str]]]):
    found_eng_i = -1
    num_sechead = 0
    found_party_line_i = -1
    num_line = len(para_attr_list)
    for i, (line, attr_list) in enumerate(para_attr_list):

        if i == num_line -1:  # last line, when toc is at the end
            found_eng_i = i
            break

        # maybe party line
        if 'party_line' in attr_list or \
           ('yes_eng' in attr_list and \
            'date_line' in attr_list):
            found_party_line_i = i
            break

        # TODO, jshaw, still not sure if this is needed.
        # if not line:
        #    continue

        # 'Date: xxx'
        # 'Tenant: xxx'
        words = line.split(' ')
        if words and words[0].endswith(':'):
            found_eng_i = i
            break

        if 'not_eng' in attr_list or 'sechead' in attr_list or 'toc' in attr_list:
            num_sechead += 1
            continue

        if 'yes_eng' in attr_list and len(line) > 120:
            found_eng_i = i
            break

    if found_party_line_i >= 0:
        for i, (line, attr_list) in enumerate(para_attr_list):
            if i < found_party_line_i:
                attr_list.append('toc70')
        return found_party_line_i

    if found_eng_i > 0 and num_sechead / float(found_eng_i) > 0.9:
        # now go find the last sechead
        i = found_eng_i - 1
        found_last_sechead = -1
        while i > 0:
            line, attr_list = para_attr_list[i]
            if 'sechead' in attr_list:
                found_last_sechead = i
                break
            i -= 1
        if found_last_sechead != -1:
            # everything before is toc
            for i, (line, attr_list) in enumerate(para_attr_list):
                if i <= found_last_sechead:
                    attr_list.append('toc7')
        return found_last_sechead
    return -1


def mark_toc(para_attr_list: List[Tuple[str, List[str]]]) -> Tuple[int, int]:
    for i, (unused_line, attr_list) in enumerate(para_attr_list):
        if 'toc' in attr_list:
            toc_start_idx = i
            toc_last_idx = mark_toc_aux(para_attr_list[i+1:])
            return toc_start_idx, i + 1 + toc_last_idx
    return -1, -1


def find_previous_sechead(para_attr_list: List[Tuple[str, List[str]]],
                          idx: int) \
                          -> int:
    while idx >= 0:
        unused_line, attr_list = para_attr_list[idx]
        if 'sechead' in attr_list:
            return idx
        idx -= 1
    return -1


def find_previous_notempty_line(para_attr_list: List[Tuple[str, List[str]]],
                                idx: int) \
                                -> int:
    idx -= 1
    while idx >= 0:
        line, unused_attr_list = para_attr_list[idx]
        if line:
            return idx
        idx -= 1
    return -1


def maybe_adjust_toc_last_recital(para_attr_list: List[Tuple[str, List[str]]],
                                  toc_last_idx: int,
                                  party_line_idx: int) \
                                  -> int:
    prev_sechead_idx = find_previous_notempty_line(para_attr_list, party_line_idx)
    if prev_sechead_idx != -1:
        line, attr_list = para_attr_list[prev_sechead_idx]

        if 'preamble' in line.lower():
            if 'toc' in attr_list:
                attr_list.remove('toc')
                toc_last_idx = prev_sechead_idx - 1
                return toc_last_idx
    # return the original idx
    return toc_last_idx


# toc_last_idx = mabye_adjust_toc_last(para_attr_list, toc_last_idx + 1, party_line_idx)
def maybe_adjust_toc_last(para_attr_list: List[Tuple[str, List[str]]],
                          toc_last_idx: int,
                          party_line_idx: int):

    prev_sechead_idx = find_previous_notempty_line(para_attr_list, party_line_idx)

    # pylint: disable=too-many-nested-blocks
    if prev_sechead_idx != -1:
        line, attr_list = para_attr_list[prev_sechead_idx]
        party_line, _ = para_attr_list[party_line_idx]
        lc_party_line_150 = party_line.lower()[:150]

        if line.lower() in lc_party_line_150:
            if 'toc' in attr_list:
                attr_list.remove('toc')
            attr_list.append('title')
            toc_last_idx = prev_sechead_idx - 1

            found_more_title = True
            while found_more_title and toc_last_idx > 0:
                # try to merge the next prev line as title again
                prev_sechead_idx = find_previous_notempty_line(para_attr_list, toc_last_idx)
                if prev_sechead_idx != -1:
                    line, attr_list = para_attr_list[prev_sechead_idx]

                    if line.lower() in lc_party_line_150:
                        if 'toc' in attr_list:
                            attr_list.remove('toc')
                        attr_list.append('title')
                        toc_last_idx = prev_sechead_idx - 1
                        found_more_title = True
                    else:
                        found_more_title = False
                else:
                    found_more_title = False

            return toc_last_idx
    return toc_last_idx

# return (toc_start_idx, toc_end_idx)
def find_sechead_toc(para_attr_list: List[Tuple[str, List[str]]]):
    # find more than 5 consecutive sechead in the first 40 non-empty lines
    num_nonempty_line = 0
    is_prev_sechead = False
    num_consecutive_sechead = 0
    consecutive_start_idx = 0  # TODO, not certain of this default init value
    max_consecutive_sechead = 0
    max_consecutive_start_idx = -1
    consecutive_start_idx = -1
    for line_idx, (line, attr_list) in enumerate(para_attr_list):
        if line:
            if 'sechead' in attr_list:
                if is_prev_sechead:
                    num_consecutive_sechead += 1
                    if num_consecutive_sechead > max_consecutive_sechead:
                        max_consecutive_sechead = num_consecutive_sechead
                        max_consecutive_start_idx = consecutive_start_idx
                else:
                    num_consecutive_sechead = 1
                    consecutive_start_idx = line_idx
                is_prev_sechead = True
            # handle '(e)' which are comb_sechead
            elif secheadutils.is_line_sechead_prefix_only(line):
                # print("is_line_sechead_prefix_only({})".format(line))
                continue
            else:
                num_consecutive_sechead = 0
                is_prev_sechead = False
            num_nonempty_line += 1
            if max_consecutive_sechead >= 4:
                break
            if num_nonempty_line > 40:
                return -1, -1
    # print("max_consecutive_sechead = {}".format(max_consecutive_sechead))
    if max_consecutive_sechead >= 4:
        toc_start_idx = max_consecutive_start_idx
        toc_last_idx = mark_toc_aux(para_attr_list[max_consecutive_start_idx:])
        return toc_start_idx, max_consecutive_start_idx + toc_last_idx
    return -1, -1

# this is called by eblearn/lineannotator.py
# pylint: disable=too-many-locals, too-many-statements
def lineinfos_paras_to_attr_list(lineinfos_paras: List[Tuple[List[Tuple[linepos.LnPos,
                                                                        linepos.LnPos]],
                                                             PLineAttrs]],
                                 doc_text: str) \
                                 -> List[Tuple[str, List[str]]]:
    para_attr_list = []  # type: List[Tuple[str, List[str]]]
    prev_out_line = ''
    # found_witness = False   # never changed.
    found_toc = False
    toc_last_idx, party_line_idx, witness_start_idx = -1, -1, -1
    date_line_idx, first_eng_para_idx = -1, -1
    lc_party_line = ''
    num_line = len(lineinfos_paras)
    num_long_english_line = 0

    for line_idx, (se_list, pline_attrs) in enumerate(lineinfos_paras):
        attr2_list = []
        # print("se_list: {}".format(se_list))
        line_st_list = []
        for from_se_ln, unused_to_se_ln in se_list:
            fstart, fend, unused_fln = from_se_ln.to_tuple()
            line_st_list.append(doc_text[fstart:fend])
        line = ' '.join(line_st_list)
        # print("line: [{}]".format(line))
        is_english = engutils.classify_english_sentence(line)
        if is_english:
            attr2_list.append('yes_eng')
            if len(line) > 350:
                num_long_english_line += 1
        else:
            attr2_list.append('not_eng')

        if engutils.is_skip_template_line(line):
            attr2_list.append('skip_as_template')

        if engutils.is_date_line(line):
            attr2_list.append('date_line')
            if date_line_idx != -1:
                date_line_idx = line_idx

        # this is redundant in regards to num_long_english_line
        # based on tests
        # if line.strip() and \
        #    len(line) > 200 and \
        #   not ('toc' in attr_list or 'toc7' in attr_list):
        #    line_num_notoc_empty += 1

        if engutils.has_date(line):
            attr2_list.append('has_date')

        if first_eng_para_idx == -1 and \
           'yes_eng' in attr2_list and \
           'skip_as_template' not in attr2_list and \
           not re.search(r'the\s+Securities\s+and\s+Exchange\s+Commission',
                         line, re.I) and \
           len(line) > 110:
            attr2_list.append('first_eng_para')
            first_eng_para_idx = line_idx

        # print('num_english_line = {}, nun_sechead = {}'.format(num_english_line,
        #                                                        num_sechead), end='')
        # print("num_date = {}".format(num_date))
        # pylint: disable=too-many-boolean-expressions
        if party_line_idx == -1 and \
           'skip_as_template' not in attr2_list and \
           partyutils.is_party_line(line, num_long_english_line) and \
           (first_eng_para_idx == -1 or \
            num_long_english_line < 2 or  # add some breadthing room for cover page
            found_toc or \
            # it was 10 before
            abs(first_eng_para_idx - line_idx) < 40):
            # print("adding party line jjjjj, [{}]".format(line))
            # print("is party? {}".format(partyutils.is_party_line(line)))
            attr2_list.append('party_line')
            party_line_idx = line_idx
            lc_party_line = line.lower()

        sechead_attr = pline_attrs.sechead
        if sechead_attr:
            unused_sechead_type, prefix_num, head, unused_split_idx = sechead_attr
            tmp_outline = '[{}]\t[{}]'.format(prefix_num, head)
            if tmp_outline != prev_out_line:
                if prefix_num == 'toc':
                    attr2_list.append('toc')
                    found_toc = True
                    toc_idx = line_idx
                attr2_list.append('sechead')
                prev_out_line = tmp_outline
            # it's possible to have mutlipe adjacents toc lines
            elif tmp_outline == prev_out_line and prefix_num == 'toc':
                attr2_list.append('toc')

        #if not found_witness:
        if WITNESS_PAT.match(line) or WHEREAS_PAT.search(line):
            attr2_list.append('preamble')
            if witness_start_idx == -1:
                witness_start_idx = line_idx

        para_attr_list.append((line, attr2_list))

    # identify TOC to find begin of party_line
    if found_toc:
        toc_start_idx, toc_last_idx = mark_toc(para_attr_list)
        if toc_idx / float(num_line) < 0.4:
            mark_attrs(para_attr_list, 0, toc_start_idx, 'first_page')
    else:
        toc_start_idx, toc_last_idx = find_sechead_toc(para_attr_list)
        if toc_start_idx != -1:
            found_toc = True
            mark_attrs(para_attr_list, 0, toc_start_idx, 'first_page')


    # print('party_line_idx = {}'.format(party_line_idx))
    # print('first_eng_para_idx = {}'.format(first_eng_para_idx))

    # in case there the last line of TOC, 'xxx agreement" is a part of the party_line,
    # "This xxx agreement"
    if party_line_idx != -1:
        toc_last_idx = maybe_adjust_toc_last(para_attr_list, toc_last_idx, party_line_idx)

    # adjust toc_last_idx iff previous line is 'preamble'
    if party_line_idx != -1:
        toc_last_idx = maybe_adjust_toc_last_recital(para_attr_list,
                                                     toc_last_idx,
                                                     party_line_idx)
    elif first_eng_para_idx != -1:
        toc_last_idx = maybe_adjust_toc_last_recital(para_attr_list,
                                                     toc_last_idx,
                                                     first_eng_para_idx)

    if toc_last_idx != -1 and party_line_idx != -1:
        mark_attrs(para_attr_list, toc_last_idx + 1, party_line_idx, 'maybe_title')
    elif party_line_idx != -1 and abs(party_line_idx - first_eng_para_idx) < 5:  # no toc
        mark_attrs(para_attr_list, 0, party_line_idx, 'first_page')
        mark_title_attrs(para_attr_list, 0, party_line_idx, lc_party_line)
    elif witness_start_idx != -1:  # no toc
        mark_attrs(para_attr_list, 0, witness_start_idx, 'first_page', ignore_if_tag='toc')
    elif first_eng_para_idx != -1:
        first_eng_sechead = find_previous_sechead(para_attr_list, first_eng_para_idx - 1)
        mark_attrs(para_attr_list, 0, first_eng_sechead, 'first_page', ignore_if_tag='toc')

    return para_attr_list
