import os
import re
import sys
from typing import List

from kirke.utils import mathutils, strutils

from kirke.docstruct import secheadutils

from kirke.docstruct import footerutils, partyutils
from kirke.utils import txtreader, engutils
                                       
DEBUG_MODE = False

def htmltxt_to_lineinfos_with_attrs(file_name, lineinfo_fname=None, is_combine_line=True):
    lineinfo_list = []
    split_idx = -1
    to_offset = 0

    prev_output_line = ''
    # These are for handling inconsistent ways html text split section number and heading.
    # But handling this issue causes a lot of complications in extraction section head
    prev_nonempty_line, prev_line_idx = '', -1
    for start, end, line in txtreader.load_normalized_lines_with_offsets(file_name):

        attr_list = []
        is_pagenum_line = False
        if start != end:
            if footerutils.classify_line_page_number(line):
                is_pagenum_line = True
                prev_nonempty_line = ''
            elif strutils.is_dashed_line(line):
                prev_nonempty_line = ''
            else:
                sechead_type, prefix_num, sec_head, split_idx = \
                        secheadutils.extract_sechead_v4(line, prev_nonempty_line, prev_line_idx, is_combine_line=is_combine_line)

                if sechead_type:
                    attr_list.append((sechead_type, prefix_num, sec_head, split_idx))
                prev_nonempty_line, prev_line_idx = line, split_idx

            # attr_list is True iff it is a section head
            if attr_list:
                if split_idx == -1:
                    # print("\t\tcxx{}\t{}\t{}".format(start, end, attr_list))
                    # print("{}\t{}\t{}".format(start, end, line))
                    lineinfo_list.append(((start, end), (to_offset, to_offset + len(line)), line, attr_list))
                    to_offset += len(line) + 1
                    prev_output_line = line
                else:
                    first_line = line[:split_idx]
                    second_line = line[split_idx:]
                    attr_list.append('xsplit')
                    # print("\t\tcxx{}\t{}\t{}".format(start, start+split_idx, attr_list))
                    # print("{}\t{}\t{}".format(start, start+split_idx, first_line))
                    # print("{}\t{}\t{}".format(start+split_idx, end, second_line))
                    lineinfo_list.append(((start, start+split_idx), (to_offset, to_offset + len(first_line)),
                                          first_line, attr_list))
                    to_offset += len(first_line) + 1  # for eoln
                    # insert a line break
                    tmp_from_end = start + split_idx
                    tmp_to_end = to_offset + len(first_line)
                    lineinfo_list.append(((tmp_from_end, tmp_from_end), (tmp_to_end, tmp_to_end),
                                          '', []))
                    to_offset += len(first_line) + 1  # for line break eoln                    
                    
                    lineinfo_list.append(((start+split_idx, end), (to_offset, to_offset + len(second_line)),
                                          second_line, []))
                    to_offset += len(second_line) + 1
                    prev_output_line = second_line                    
            else:  # no attr_list, but maybe a page number
                # print("{}\t{}\t[{}]".format(start, end, line))
                if is_pagenum_line:
                    attr_list.append('pagenum')          
                lineinfo_list.append(((start, end), (to_offset, to_offset + len(line)),
                                      line, attr_list))
                to_offset += len(line) +1
                prev_output_line = line

        else:  # blank line, though spaces might have been removed
            if prev_output_line != '':
                lineinfo_list.append(((start, start), (to_offset, to_offset),
                                      '', []))
                to_offset += 1
                prev_output_line = ''

    doc_lines = [line for _, _, line, _ in lineinfo_list]
    doc_text = '\n'.join(doc_lines)

    return lineinfo_list, doc_text
    
def has_sechead_attr(attr_list):
    for attr in attr_list:
        if attr == 'pagenum':
            pass
        elif attr == 'xsplit':
            pass
        else:
            return True
    return False

def get_sechead_attr(attr_list):
    for attr in attr_list:
        if attr == 'pagenum':
            pass
        elif attr == 'xsplit':
            pass
        else:
            return attr
    return []

# This distribute sechead to all the lines after it
# and remove pagenum.
# TODO, Should add footer and header in the future.
# But such info only available in PDF files.
def lineinfos_to_paras(lineinfos):
    # make a list of iterators,
    # will be easier to remove pagenum
    tmp_list = list(lineinfos)  
        
    len_tmp_list = len(tmp_list)
    omit_line_set = set([])
    cur_attr = []
    prev_line = ''
    tmp2_list = []
    prev_notempty_line, prev_attr_list = 'Not Empty Line.', []
    gap_span_list = []
    prefix = 'fake_prefix'
    for i, linfo in enumerate(tmp_list):
        (start, end), (_, _), line, attr_list = linfo
        # print('line #{}\t[{}]'.format(i, line))

        if attr_list:
            for attr in attr_list:
                # if there is a pagenum, remove the line between it, previous and the next
                if attr == 'pagenum':
                    omit_line_set.add(i)
                    gap_span_list.append((start, end))
                    omit_list = []
                    if i - 1 >= 0:
                        _, _, prev_line, _ = tmp_list[i-1]
                        if prev_line:
                            prev_notempty_line = prev_line
                        else:
                            omit_list.append(i-1)
                            if i - 2 >= 0:
                                _, _, prev_notempty_line, _ = tmp_list[i-2]
                            else:
                                prev_notempty_line = 'Not Empty Line.'
                    if i + 1 < len_tmp_list:
                        _, _, next_line, _ = tmp_list[i+1]
                        if next_line:
                            next_notempty_line = next_line
                        else:
                            omit_list.append(i+1)
                            if i + 2 < len_tmp_list:
                                _, _, next_notempty_line, _ = tmp_list[i+2]
                        ## if really end of sentence and have any line break
                        ## leave 1 empty line break in.
                        #if prev_notempty_line[-1] in set(['.', '?', '!', '_', ':']) and omit_list:
                        #    omit_line_set.add(omit_list[0])
                        #else:  # not end of sentence, remove all empty lines
                        #    omit_line_set |= set(omit_list)
                        # print('prev_notempty_line = [{}]'.format(prev_notempty_line))
                        # scccccc
                        # xxx failed
                        # diff dir-data/40213.clean.txt.lineinfo.paras /tmp/40213.clean.txt.lineinfo.paras
                        if omit_list and has_sechead_attr(prev_attr_list):
                            omit_line_set.add(omit_list[0])
                        elif (strutils.is_all_alphas(prev_notempty_line[-1]) or
                            prev_notempty_line[-1] in set([',', '-'])):
                            omit_line_set |= set(omit_list)
                        elif omit_list:
                            omit_line_set.add(omit_list[0])
                elif attr == 'xsplit':
                    pass
                else:
                    # "cur_attr" is how sechead info is distribute to the lines below it.
                    sec_type, prefix, head, _ = attr
                    cur_attr = [attr]
                    # omit_line_set.add(i)
        else:  # attr_list is empty
            # "interactive Intell SOW CNG 000 Child.pdf" failed the "not prev_line" test.
            # need to compute the ydiff, and somehow add line breaks to lineinfos (not in file, but only in memory).
            # if not prev_line and prefix == 'toc':  # we don't continue TOC, if previous prefix is toc
            if prefix == 'toc':  # there is no attribute, we don't continue 'toc'
                cur_attr = []

        if not prev_line and not line:
            omit_line_set.add(i)
            
        tmp2_list.append((start, end, line, cur_attr))
        prev_line = line
        prev_attr_list = attr_list

    result = []
    out_offset = 0
    for i, linfo in enumerate(tmp2_list):
        if i not in omit_line_set:
            start, end, line, attr_list = linfo
            sechead_attr = get_sechead_attr(attr_list)
            if sechead_attr:
                result.append(((start, end), (out_offset, out_offset + len(line)), line, [sechead_attr]))
            else:
                result.append(((start, end), (out_offset, out_offset + len(line)), line, []))
            out_offset += len(line) + 1
            
    doc_lines = [line for _, _, line, _ in result]
    doc_text = '\n'.join(doc_lines)

    return result, doc_text, gap_span_list


def paras_to_fromto_lists(para_list):
    alist = []
    for (from_start, from_end), (to_start, to_end), line, attr_list in para_list:
        alist.append((from_start, to_start))

    sorted_alist = sorted(alist)

    from_list = [a for a,b in sorted_alist]
    to_list = [b for a,b in sorted_alist]
    return from_list, to_list

witness_pat = re.compile(r'(w i t n e s s e t h|witnesseth|recitals?\:?|r e c i t a l( s)?(\s*\:)?)', re.IGNORECASE)
# 'background statement'
whereas_pat = re.compile(r'^\s*(whereas|background)', re.IGNORECASE)

def mark_attrs(para_attr_list, begin_idx, end_idx, attr, ignore_if_tag=None):
    if end_idx <= begin_idx:
        return
    for i, (line, attr_list) in enumerate(para_attr_list):
        if i >= begin_idx and i < end_idx:
            if line and (not ignore_if_tag or (ignore_if_tag and not ignore_if_tag in attr_list)):
                attr_list.append(attr)
        elif i >= end_idx:
            return

def mark_title_attrs(para_attr_list, begin_idx, end_idx, lc_party_line):
    if end_idx <= begin_idx:
        return
    for i, (line, attr_list) in enumerate(para_attr_list):
        if i >= begin_idx and i < end_idx:
            if line and re.sub(r'\s+', ' ', line.lower()) in re.sub(r'\s+', ' ',lc_party_line):
                attr_list.append('title')
        elif i >= end_idx:
            return


# already found 'toc' marker
def mark_toc_aux(para_attr_list):
    found_eng_i = -1
    num_sechead = 0
    num_line = len(para_attr_list)
    for i, (line, attr_list) in enumerate(para_attr_list):

        if i == num_line -1:  # last line, when toc is at the end
            found_eng_i = i
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


def mark_toc(para_attr_list):
    for i, (line, attr_list) in enumerate(para_attr_list):
        if 'toc' in attr_list:
            toc_start_idx = i
            toc_last_idx = mark_toc_aux(para_attr_list[i+1:])
            return toc_start_idx, i + 1 + toc_last_idx
    return -1, -1


def find_previous_sechead(para_attr_list, idx):
    while idx >= 0:
        line, attr_list = para_attr_list[idx]
        if 'sechead' in attr_list:
            return idx
        idx -= 1
    return -1


def find_previous_notempty_line(para_attr_list, idx):
    idx -= 1
    while idx >= 0:
        line, attr_list = para_attr_list[idx]
        if line:
            return idx
        idx -= 1
    return -1


def maybe_adjust_toc_last_recital(para_attr_list, toc_last_idx,  party_line_idx):
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


# toc_last_idx = mabye_adjust_toc_last(para_attr_list, toc_last_idx + 1,  party_line_idx)
def maybe_adjust_toc_last(para_attr_list, toc_last_idx, party_line_idx):

    prev_sechead_idx = find_previous_notempty_line(para_attr_list, party_line_idx)

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
def find_sechead_toc(para_attr_list):
    # find more than 5 consecutive sechead in the first 40 non-empty lines
    num_nonempty_line = 0
    is_prev_sechead = False
    num_consecutive_sechead = 0
    max_consecutive_sechead = 0
    max_consecutive_start_idx = -1
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


def lineinfos_paras_to_attr_list(lineinfos_paras):
    para_attr_list = []
    prev_out_line = ''
    found_witness = False   # never changed.
    found_toc = False
    toc_last_idx, party_line_idx, witness_start_idx = -1, -1, -1
    date_line_idx, first_eng_para_idx = -1, -1
    lc_party_line = ''
    num_line = len(lineinfos_paras)
    for line_idx, (_, _, line, attr_list) in enumerate(lineinfos_paras):
        attr2_list = []
        is_english = engutils.classify_english_sentence(line)
        if is_english:
            attr2_list.append('yes_eng')
        else:
            attr2_list.append('not_eng')

        if engutils.is_skip_template_line(line):
            attr2_list.append('skip_as_template')

        if engutils.is_date_line(line):
            attr2_list.append('date_line')
            if date_line_idx != -1:
                date_line_idx = line_idx

        if engutils.has_date(line):
            attr2_list.append('has_date')

        if (first_eng_para_idx == -1 and
            'yes_eng' in attr2_list and
            not 'skip_as_template' in attr2_list and
            len(line) > 110):
            attr2_list.append('first_eng_para')
            first_eng_para_idx = line_idx

        if (party_line_idx == -1 and
            not 'skip_as_template' in attr2_list and
            partyutils.is_party_line(line) and
            (first_eng_para_idx == -1 or
             found_toc or
             # it was 10 before
             abs(first_eng_para_idx - line_idx) < 40)):
            attr2_list.append('party_line')
            party_line_idx = line_idx
            lc_party_line = line.lower()

        if attr_list:
            sechead_attr = attr_list[0]
            sechead_type, prefix_num, head, split_idx = sechead_attr
            tmp_outline = '[{}]\t[{}]'.format(prefix_num, head)
            if tmp_outline != prev_out_line:
                if prefix_num == 'toc':
                    attr2_list.append('toc')
                    found_toc = True
                    toc_idx = line_idx
                attr2_list.append('sechead')
                prev_out_line = tmp_outline
            elif tmp_outline == prev_out_line and prefix_num == 'toc':  # it's possible to have mutlipe adjacents toc lines
                attr2_list.append('toc')
        #if not found_witness:
        if witness_pat.match(line) or whereas_pat.search(line):
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

    # in case there the last line of TOC, 'xxx agreement" is a part of the party_line, "This xxx agreement"
    if party_line_idx != -1:
        toc_last_idx = maybe_adjust_toc_last(para_attr_list, toc_last_idx,  party_line_idx)

    # adjust toc_last_idx iff previous line is 'preamble'
    if party_line_idx != -1:
        toc_last_idx = maybe_adjust_toc_last_recital(para_attr_list, toc_last_idx,  party_line_idx)
    elif first_eng_para_idx != -1:
        toc_last_idx = maybe_adjust_toc_last_recital(para_attr_list, toc_last_idx,  first_eng_para_idx)

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

# 'is_combine_line' indicates if the system combines line when doing sechead identification
# for HTML docs, this shoulbe True.  For PDF documents, this should be False.
def parse_document(file_name, work_dir, is_combine_line=True):
    debug_mode = True

    base_fname = os.path.basename(file_name)
    orig_doc_text = txtreader.loads(file_name)

    lineinfos_with_attrs, lineinfo_doc_text = htmltxt_to_lineinfos_with_attrs(file_name, is_combine_line=is_combine_line)
    if debug_mode:
        lineinfo_fname = '{}/{}.lineinfo.v1'.format(work_dir, base_fname).replace('.txt', '')
        with open(lineinfo_fname, 'wt') as fout:
            for _, _, line, attr_list in lineinfos_with_attrs:
                print("{}\t{}".format(attr_list, line[:50]), file=fout)
            # txtreader.dumps(lineinfo_doc_text, lineinfo_fname)
        print('wrote {}'.format(lineinfo_fname), file=sys.stderr)

    lineinfos_paras, paras_doc_text, gap_span_list = \
             lineinfos_to_paras(lineinfos_with_attrs)

    """
    tmp_lineinfo_fname = '{}/{}.lineinfo.tmp_paras'.format(work_dir, base_fname).replace('.txt', '')
    with open(tmp_lineinfo_fname, 'wt') as fout:
        # for _, _, line, attr_list in lineinfos_paras:
        #        print("{}\t{}".format(attr_list, line[:50]), file=fout)
        for x in lineinfos_paras:
            print("{}".format(x), file=fout)
        # txtreader.dumps(lineinfo_doc_text, lineinfo_fname)
        print('wrote {}'.format(tmp_lineinfo_fname), file=sys.stderr)
    """

    if debug_mode:
        paras_fname = '{}/{}.lineinfo.paras'.format(work_dir, base_fname).replace('.txt', '')
        txtreader.dumps(paras_doc_text, paras_fname)
        print('wrote {}'.format(paras_fname), file=sys.stderr)

        # TODO, remove this after debugging
        para_debug_fname = paras_fname.replace('.paras', '.paras.debug')
        with open(para_debug_fname, 'wt') as fout1:
            paras_attr_list = lineinfos_paras_to_attr_list(lineinfos_paras)
            for line, para_attrs in paras_attr_list:
                attrs_st = '|'.join([str(attr) for attr in para_attrs])
                print('\t'.join([attrs_st, '[{}]'.format(line)]), file=fout1)
        print('wrote {}'.format(para_debug_fname), file=sys.stderr)

        sechead_fname = paras_fname.replace('.paras', '.secheads')
        with open(sechead_fname, 'wt') as fout2:
            prev_out_line = ''
            for _, (to_start, to_end), line, attr_list in lineinfos_paras:
                if attr_list:
                    sechead_attr = attr_list[0]
                    to_sechead_st = paras_doc_text[to_start:to_end]
                    sechead_type, prefix_num, head, split_idx = sechead_attr
                    out_line = '<{}>\t{}\t{}\t{}'.format(to_sechead_st, prefix_num, head, split_idx)
                    tmp_outline = '[{}]\t[{}]'.format(prefix_num, head)
                    if tmp_outline != prev_out_line:
                        print(out_line, file=fout2)
                    prev_out_line = tmp_outline
        print('wrote %s' % (sechead_fname, ), file=sys.stderr)

    return lineinfos_paras, paras_doc_text, gap_span_list, orig_doc_text
