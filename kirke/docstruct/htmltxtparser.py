import os
import sys
# pylint: disable=unused-import
from typing import Any, List, Optional, Tuple


from kirke.docstruct import footerutils, htmldocutils, partyutils, secheadutils
from kirke.utils import ebsentutils, engutils, strutils, txtreader
from kirke.docstruct.docutils import PLineAttrs

from kirke.docstruct import linepos
from kirke.docstruct.secheadutils import SecHeadTuple

IS_DEBUG_MODE = False

# pylint: disable=too-few-public-methods
class HPLineAttrs:

    def __init__(self) -> None:
        self.sechead = None  # type: Optional[Tuple[str, str, str, int]]
        self.xsplit = False
        self.has_page_num = False

    def __str__(self) -> str:
        alist = []  # List[str]
        if self.sechead:
            alist.append('{}={}'.format('sechead', self.sechead))
        if self.xsplit:
            alist.append('{}={}'.format('xsplit', self.xsplit))
        if self.has_page_num:
            alist.append('{}={}'.format('has_page_num', self.has_page_num))

        return '|'.join(alist)

    def has_any_attrs(self) -> bool:
        return bool(self.sechead or self.xsplit or self.has_page_num)

# returning List[((from_start, from_end),
#                 (to_start, to_end)),
#                 text_span,
#                 attr_list)]
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def htmltxt_to_lineinfos_with_attrs(file_name: str,
                                    is_combine_line: bool = True) \
                                    -> Tuple[List[Tuple[Tuple[int, int],
                                                        Tuple[int, int],
                                                        str,
                                                        HPLineAttrs]],
                                             # sechead_list
                                             List[SecHeadTuple],
                                             str]:
    """Convert a text into lineinfos_with_attrs.

    Returns
       1. lineinfo_list.append(((start, end),
                                (to_offset, to_offset + len(line)),
                                line,
                                hpline_attrs))
       2. sechead_list
       3. text

    attr_list is usually a 'str', but sometimes (sechead_type, prefix_num, sec_head, split_idx)
    """
    # pylint: disable=line-too-long
    lineinfo_list = []  # type: List[Tuple[Tuple[int, int], Tuple[int, int], str, HPLineAttrs]]
    split_idx = -1
    to_offset = 0
    sechead_list = []  # type: List[SecHeadTuple]

    prev_output_line = ''
    # These are for handling inconsistent ways html text split section number and heading.
    # But handling this issue causes a lot of complications in extraction section head
    prev_nonempty_line, prev_line_idx = '', -1
    for start, end, line in txtreader.load_normalized_lines_with_offsets(file_name):
        hpline_attrs = HPLineAttrs()  # type: HPLineAttrs
        is_pagenum_line = False
        if start != end:
            if footerutils.classify_line_page_number(line):
                is_pagenum_line = True
                prev_nonempty_line = ''
            elif strutils.is_dashed_line(line):
                prev_nonempty_line = ''
            else:
                sechead_t4 = \
                    secheadutils.extract_sechead(line,
                                                 prev_line=prev_nonempty_line,
                                                 prev_line_idx=prev_line_idx,
                                                 is_combine_line=is_combine_line)
                # pylint: disable=pointless-string-statement
                """
                print("secheadutils.extract_sechead_v4(ln={}, prv={}, prv_idx={}, iscomb={})".format(line,
                                                                                                     prev_nonempty_line,
                                                                                                     prev_line_idx,
                                                                                                     is_combine_line))
                print("       sechead_type= {}, prefix_num= {}, sec_head= {}, split_idx= {}".format(sechead_type,
                                                                                                    prefix_num,
                                                                                                    sec_head,
                                                                                                    split_idx))
                """

                if sechead_t4:
                    hpline_attrs.sechead = sechead_tuple
                    unused_sechead_type, prefix_num, sec_head, split_idx = sechead_t4
                else:
                    split_idx = -1
                prev_nonempty_line, prev_line_idx = line, split_idx

            # attr_list is True iff it is a section head
            if hpline_attrs.sechead:
                if split_idx == -1:
                    # print("\t\tcxx{}\t{}\t{}".format(start, end, attr_list))
                    # print("{}\t{}\t{}".format(start, end, line))
                    lineinfo_list.append(((start, end),
                                          (to_offset, to_offset + len(line)),
                                          line,
                                          hpline_attrs))
                    to_offset += len(line) + 1
                    prev_output_line = line

                    if sec_head:
                        sechead_st = sec_head
                    else:
                        sechead_st = prefix_num
                        prefix_num = ''
                    out_sechead = SecHeadTuple(start,
                                               end,
                                               prefix_num,
                                               sechead_st,
                                               -1)
                    # print("html sechead_tuple: {}".format(out_sechead))
                    sechead_list.append(out_sechead)
                else:
                    first_line = line[:split_idx]
                    second_line = line[split_idx:]
                    hpline_attrs.xsplit = True
                    # print("\t\tcxx{}\t{}\t{}".format(start, start+split_idx, attr_list))
                    # print("{}\t{}\t{}".format(start, start+split_idx, first_line))
                    # print("{}\t{}\t{}".format(start+split_idx, end, second_line))
                    lineinfo_list.append(((start, start+split_idx),
                                          (to_offset, to_offset + len(first_line)),
                                          first_line,
                                          hpline_attrs))

                    # insert a line break
                    tmp_from_end = start + split_idx
                    tmp_to_end = to_offset + len(first_line) + 1   # line break
                    lineinfo_list.append(((tmp_from_end, tmp_from_end),
                                          (tmp_to_end, tmp_to_end),
                                          '',
                                          HPLineAttrs()))

                    to_offset += len(first_line) + 2  # for 2 eolns

                    lineinfo_list.append(((start+split_idx, end),
                                          (to_offset, to_offset + len(second_line)),
                                          second_line,
                                          HPLineAttrs()))
                    to_offset += len(second_line) + 1
                    prev_output_line = second_line

                    if sec_head:
                        sechead_st = sec_head
                    else:
                        sechead_st = prefix_num
                        prefix_num = ''
                    out_sechead = SecHeadTuple(start,
                                               start + split_idx,
                                               prefix_num,
                                               sechead_st,
                                               # first_line,
                                               -1)
                    # print("html sechead_tuple2: {}".format(out_sechead))
                    sechead_list.append(out_sechead)
            else:  # no attr_list, but maybe a page number
                # print("{}\t{}\t[{}]".format(start, end, line))
                if is_pagenum_line:
                    hpline_attrs.has_page_num = True
                lineinfo_list.append(((start, end),
                                      (to_offset, to_offset + len(line)),
                                      line,
                                      hpline_attrs))
                to_offset += len(line) +1
                prev_output_line = line

        else:  # blank line, though spaces might have been removed
            if prev_output_line != '':
                lineinfo_list.append(((start, start),
                                      (to_offset, to_offset),
                                      '',
                                      HPLineAttrs()))
                to_offset += 1
                prev_output_line = ''

    doc_lines = [line for unused_fromx, unuused_tox, line, unused_attrlist
                 in lineinfo_list]
    doc_text = '\n'.join(doc_lines)

    return lineinfo_list, sechead_list, doc_text


EMPTY_PLINE_ATTRS = PLineAttrs()

# This distribute sechead to all the lines after it
# and remove pagenum.
# TODO, Should add footer and header in the future.
# But such info only available in PDF files.
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def lineinfos_to_paras(lineinfos: List[Tuple[Tuple[int, int],
                                             Tuple[int, int],
                                             str,
                                             HPLineAttrs]]) \
    -> Tuple[List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]],
                        PLineAttrs]],
             str,
             List[Tuple[int, int]]]:
    # make a list of iterators,
    # will be easier to remove pagenum
    tmp_list = list(lineinfos)

    len_tmp_list = len(tmp_list)
    omit_line_set = set([])
    cur_attr = None  # type: Optional[Tuple[str, str, str, int]]
    # The 4th field is sechead.  This is intentional
    tmp2_list = []  # type: List[Tuple[int, int, str, Optional[Tuple[str, str, str, int]]]]
    prev_line = ''
    prev_notempty_line = 'Not Empty Line.'
    prev_hpline_attrs = HPLineAttrs()  # type: HPLineAttrs
    exclude_offsets = []  # type: List[Tuple[int, int]]
    prefix = 'fake_prefix'
    # pylint: disable=too-many-nested-blocks
    for i, linfo in enumerate(tmp_list):
        (start, end), (unused_to_start_2, unused_to_end_2), line, hpline_attrs = linfo
        # print('line #{}\t[{}]'.format(i, line))

        if not hpline_attrs.has_any_attrs():
            # "interactive Intell SOW CNG 000 Child.pdf" failed the "not prev_line" test.
            # need to compute the ydiff, and somehow add line breaks to lineinfos (not in file,
            # but only in memory).

            # if not prev_line and prefix == 'toc':  # we don't continue TOC, if
            # previous prefix is toc
            if prefix == 'toc':  # there is no attribute, we don't continue 'toc'
                cur_attr = None

        # if there is a pagenum, remove the line between it, previous and the next
        if hpline_attrs.has_page_num:
            omit_line_set.add(i)
            exclude_offsets.append((start, end))
            omit_list = []
            if i - 1 >= 0:
                unused_prev_from_se, unused_prev_to_se, prev_line, unused_prev_attrs = \
                    tmp_list[i-1]
                if prev_line:
                    prev_notempty_line = prev_line
                else:
                    omit_list.append(i-1)
                    if i - 2 >= 0:
                        # pylint: disable=line-too-long
                        unused_prev2_from_se, unused_prev2_to_se, tmp_prev_notempty_line, unused_x2 = tmp_list[i-2]
                        if tmp_prev_notempty_line:
                            prev_notempty_line = tmp_prev_notempty_line
                    else:
                        prev_notempty_line = 'Not Empty Line.'
            if i + 1 < len_tmp_list:
                # pylint: disable=line-too-long
                unused_nline_from_se, unused_nline_to_se, next_line, unused_nline_attrs = tmp_list[i+1]
                if next_line:
                    # next_notempty_line = next_line
                    pass
                else:
                    omit_list.append(i+1)
                    # if i + 2 < len_tmp_list:
                    #     _, _, next_notempty_line, _ = tmp_list[i+2]

                ## if really end of sentence and have any line break
                ## leave 1 empty line break in.
                #if prev_notempty_line[-1] in set(['.', '?', '!', '_', ':']) and omit_list:
                #    omit_line_set.add(omit_list[0])
                #else:  # not end of sentence, remove all empty lines
                #    omit_line_set |= set(omit_list)
                # print('prev_notempty_line = [{}]'.format(prev_notempty_line))
                # scccccc
                # xxx failed
                # pylint: disable=line-too-long
                # diff dir-data/40213.clean.txt.lineinfo.paras /tmp/40213.clean.txt.lineinfo.paras
                if omit_list and prev_hpline_attrs.sechead:
                    omit_line_set.add(omit_list[0])
                elif strutils.is_all_alphas(prev_notempty_line[-1]) or \
                     prev_notempty_line[-1] in set([',', '-']):
                    omit_line_set |= set(omit_list)
                elif omit_list:
                    omit_line_set.add(omit_list[0])
        # if hpline_attrs.xsplit:
        #     pass
        if hpline_attrs.sechead:
            # "cur_attr" is how sechead info is distribute to the lines below it.
            unused_sec_type, prefix, unused_head, _ = hpline_attrs.sechead
            cur_attr = hpline_attrs.sechead
            # omit_line_set.add(i)

        if not prev_line and not line:
            omit_line_set.add(i)

        tmp2_list.append((start, end, line, cur_attr))
        prev_line = line
        prev_hpline_attrs = hpline_attrs

    # We intentionally pass back PLineAttrs to keep consistency with
    # pdftxtparser's paras_attrs
    # pylint: disable=line-too-long
    doc_lines = []  # type: List[str]  # lines for nlp_text
    out_offset = 0
    non_empty_line_num = 0
    result = []  # type: List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]], PLineAttrs]]
    for i, linfo2 in enumerate(tmp2_list):

        if i not in omit_line_set:
            start, end, line, sechead_attr = linfo2
            # TODO, jshaw
            # The logic to handle non-empty_line_num is not exactly
            # the same as in pdftxtparser.  In pdftxtparser, not_empty_line_num
            # always increases even for page_num, which is a gap line.
            # In here, that seems to be not true.  When there is a gap line, the result should still
            # add it, with (same start, end=start, line_num=prev_line_num+1, gap=True)
            from_lpos = linepos.LnPos(start, end, non_empty_line_num)
            to_lpos = linepos.LnPos(out_offset, out_offset + len(line), non_empty_line_num)
            if line:
                non_empty_line_num += 1

            span_frto_list = [(from_lpos, to_lpos)]
            # print("span_frto_list: {}".format(span_frto_list))
            if sechead_attr:
                if line:
                    pline_attrs = PLineAttrs()
                    pline_attrs.sechead = sechead_attr
                    result.append((span_frto_list, pline_attrs))
                    doc_lines.append(line)
                else:
                    result.append((span_frto_list, EMPTY_PLINE_ATTRS))
                    doc_lines.append(line)
            else:
                # result.append(((start, end), (out_offset, out_offset + len(line)), line, []))
                result.append((span_frto_list, EMPTY_PLINE_ATTRS))
                doc_lines.append(line)
            out_offset += len(line) + 1

    # doc_lines = [line for _, line, _ in result]
    # the last '\n' is for the last line
    doc_text = '\n'.join(doc_lines) + '\n'

    return result, doc_text, exclude_offsets


class HTMLTextDoc:

    # pylint: disable=too-many-arguments
    def __init__(self,
                 file_name: str,
                 doc_text: str,
                 nlp_doc_text: str,
                 nlp_paras_with_attrs: List[Tuple[List[Tuple[linepos.LnPos,
                                                             linepos.LnPos]],
                                                  PLineAttrs]],
                 exclude_offsets: List[Tuple[int, int]],
                 sechead_list: List[SecHeadTuple]) \
                 -> None:
        self.file_name = file_name
        self.doc_text = doc_text
        self.nlp_doc_text = nlp_doc_text
        self.nlp_paras_with_attrs = nlp_paras_with_attrs
        self.exclude_offsets = exclude_offsets
        self.sechead_list = sechead_list


# 'is_combine_line' indicates if the system combines line when doing sechead identification
# for HTML docs, this shoulbe True.  For PDF documents, this should be False.
# pylint: disable=too-many-locals
def parse_document(file_name: str,
                   work_dir: str,
                   is_combine_line: bool,  # default to True before
                   nlptxt_file_name: Optional[str]) \
                   -> HTMLTextDoc:
    debug_mode = False
    base_fname = os.path.basename(file_name)
    orig_doc_text = txtreader.loads(file_name)

    lineinfos_with_attrs, sechead_list, unused_lineinfo_doc_text = \
        htmltxt_to_lineinfos_with_attrs(file_name, is_combine_line=is_combine_line)
    if debug_mode:
        lineinfo_fname = '{}/{}.lineinfo.v1'.format(work_dir, base_fname).replace('.txt', '')
        with open(lineinfo_fname, 'wt') as fout:
            for i, (from_se, to_se, line, hpline_attrs) in enumerate(lineinfos_with_attrs):
                print("line #{}\t{}\t{}\t{}\t[{}]".format(i, from_se, to_se, str(hpline_attrs),
                                                          line),
                      file=fout)
            # txtreader.dumps(lineinfo_doc_text, lineinfo_fname)
        print('wrote {}'.format(lineinfo_fname), file=sys.stderr)

    lineinfos_paras, paras_doc_text, exclude_offsets = \
             lineinfos_to_paras(lineinfos_with_attrs)

    if debug_mode:
        paras_fname = '{}/{}.lineinfo.paras'.format(work_dir, base_fname).replace('.txt', '')
        txtreader.dumps(paras_doc_text, paras_fname)
        print('wrote {}'.format(paras_fname), file=sys.stderr)

        se_para_debug_fname = paras_fname.replace('.paras', '.se.paras.debug')
        with open(se_para_debug_fname, 'wt') as fout10:
            for span_lnpos_list, para_attrs in lineinfos_paras:
                attrs_st = str(para_attrs)
                print('\t'.join([str(span_lnpos_list), attrs_st]), file=fout10)
        print('wrote {}'.format(se_para_debug_fname), file=sys.stderr)

        para_debug_fname = paras_fname.replace('.paras', '.paras.debug')
        with open(para_debug_fname, 'wt') as fout1:
            paras_attr_list = htmldocutils.lineinfos_paras_to_attr_list(lineinfos_paras,
                                                                        orig_doc_text)
            for line, para_attrs_xx in paras_attr_list:
                attrs_st = '|'.join([str(attr) for attr in para_attrs_xx])
                print('\t'.join([attrs_st, '[{}]'.format(line)]), file=fout1)
        print('wrote {}'.format(para_debug_fname), file=sys.stderr)

        sechead_fname = paras_fname.replace('.paras', '.secheads')
        with open(sechead_fname, 'wt') as fout2:

            prev_out_line = ''
            for span_frto_list, pline_attrs in lineinfos_paras:
                sechead_attr = pline_attrs.sechead

                to_se_list = [span_frto[1] for span_frto in span_frto_list]
                to_start = to_se_list[0].start    # to_se_list[0] = to_lpos
                to_end = to_se_list[-1].end

                if sechead_attr:
                    # sechead_attr = attr_list[0]
                    to_sechead_st = paras_doc_text[to_start:to_end]
                    unused_sechead_type, prefix_num, head, split_idx = sechead_attr
                    out_line = '<{}>\t{}\t{}\t{}'.format(to_sechead_st, prefix_num, head, split_idx)
                    tmp_outline = '[{}]\t[{}]'.format(prefix_num, head)
                    if tmp_outline != prev_out_line:
                        print(out_line, file=fout2)
                    prev_out_line = tmp_outline
        print('wrote %s' % (sechead_fname, ), file=sys.stderr)

    if nlptxt_file_name:
        txtreader.dumps(paras_doc_text, nlptxt_file_name)
        if IS_DEBUG_MODE:
            print("wrote {}".format(nlptxt_file_name), file=sys.stderr)

    html_text_doc = HTMLTextDoc(file_name,
                                doc_text=orig_doc_text,
                                nlp_doc_text=paras_doc_text,
                                nlp_paras_with_attrs=lineinfos_paras,
                                exclude_offsets=exclude_offsets,
                                sechead_list=sechead_list)

    # nlp_paras_with_attrs, nlp_doc_text, unused_gap_span_list, unused_
    # return lineinfos_paras, paras_doc_text, exclude_offsets, orig_doc_text
    return html_text_doc
