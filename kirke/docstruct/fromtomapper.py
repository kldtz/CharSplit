
import bisect
import logging
from typing import Any, Dict, List, Tuple, Union

from kirke.docstruct import linepos
from kirke.utils.ebsentutils import ProvisionAnnotation

def compute_fromto_offsets(start: int,
                           end: int,
                           from_sxlnpos_list: List[Tuple[int, linepos.LnPos]],
                           to_sxlnpos_list: List[Tuple[int, linepos.LnPos]]) -> Tuple[int, int]:
    from_start_list = [start for start, lnpos in from_sxlnpos_list]
    to_start_list = [start for start, lnpos in to_sxlnpos_list]

    result_start = find_offset_to(start, from_start_list, to_start_list)
    result_end = find_offset_to(end, from_start_list, to_start_list)

    return result_start, result_end
    

def find_offset_to(fromx: int,
                   from_list: List[int],
                   to_list: List[int]) -> int:
    return find_offset_to_binary(fromx, from_list, to_list)

# binary search version
## there is some error in binary_search version, result
## diff from find_offset_to_linear in certain cases???
def find_offset_to_binary(fromx: int,
                          from_list: List[int] ,
                          to_list: List[int]) -> int:

    # find rightmost value less than or equal to fromx
    found_i = bisect.bisect_right(from_list, fromx)
    if found_i:
        if fromx == from_list[found_i-1]:
            return to_list[found_i-1]
        diff = fromx - from_list[found_i-1]
        return to_list[found_i-1] + diff

    return -1


# linear version
def find_offset_to_linear(fromx: int,
                          from_list: List[int],
                          to_list: List[int]) -> int:
    found_i = -1
    for i, val in enumerate(from_list):
        if val >= fromx:
            found_i = i
            break

    if found_i != -1:
        if fromx == from_list[found_i]:
            return to_list[found_i]
        # we must be greater than from_list[found_i] before
        diff = fromx - from_list[found_i-1]
        return to_list[found_i-1] + diff

    return -1


def paras_to_fromto_lists(para_list: List[Tuple[List[Tuple[linepos.LnPos,
                                                           linepos.LnPos]],
                                                str,
                                                List[Tuple[Any]]]]) -> Tuple[List[Tuple[int, linepos.LnPos]],
                                                                             List[Tuple[int, linepos.LnPos]]]:
    alist = []
    # print("docutils.paras_to_fromto_lists()")
    for span_se_list, line, attr_list in para_list:
        # print("  span_se_list: {}".format(span_se_list))
        # for (from_lnpos, to_lnpos) in span_se_list:
        #    print("    from_lnpos = {}, to_lnpos = {}".format(from_lnpos, to_lnpos))

        # at this point from_lnpos is for original text
        # to_lnpos is for nlp text
        for (from_lnpos, to_lnpos) in span_se_list:
            # intentionally not use from_lnpos.end
            # using to_lnpos.end, just in case there is a gap, which migth cause two to_lnpos.start to
            # be the same
            alist.append((to_lnpos.start, to_lnpos.end, from_lnpos.start, from_lnpos, to_lnpos))

    # ordered by to_start, because this is where we will map from,
    # and it need be ordered
    sorted_alist = sorted(alist)

    from_list = [(b, c) for a, a2, b, c, d in sorted_alist]
    to_list = [(a, d) for a, a2, b, c, d in sorted_alist]
    return from_list, to_list


def find_index_diff(the_offset: int, from_list: List[int]) -> Tuple[int, int]:
    return find_index_diff_binary(the_offset, from_list)


# from_list is SORTED
# return (index: where the line is in the from_list,
#         diff: where the point is from the line.start)
# not tested
def find_index_diff_binary(the_offset: int, from_list: List[int]) -> Tuple[int, int]:

    # find rightmost value less than or equal to the_offset
    found_i = bisect.bisect_right(from_list, the_offset)
    if found_i:
        if the_offset == from_list[found_i-1]:
            return found_i - 1, 0
        diff = the_offset - from_list[found_i-1]
        return found_i - 1, diff

    logging.error('find_index_diff_binary({}, _) failed'.format(the_offset))
    return -1, -1


# from_list is SORTED
# return (index: where the line is in the from_list,
#         diff: where the point is from the line.start)
# not tested
def find_index_diff_linear(the_offset: int, from_list: List[int]) -> Tuple[int, int]:
    for i, from_start in enumerate(from_list):
        if from_start == the_offset:
            return i, 0
        elif from_start > the_offset:
            # we must be greater than from_list[i] before
            diff = the_offset - from_list[i - 1]
            return i - 1, diff
    logging.error('find_index_diff_linear({}, _) failed'.format(the_offset))        
    return -1, -1


# to take advantage of python typing, add the following class
# Otherwise, List[Tuple[linepos.LnPos, int], int] is not declarable in typing
# and Tuple[Tuple[linepos.LnPos, int], int] is not assignable in python
class StartLnPosDiff:

    def __init__(self, start_lnpos: Tuple[int, linepos.LnPos], diff: int) -> None:
        self.start_lnpos = start_lnpos
        self.diff = diff

    def __lt__(self, other):
        return self.start_lnpos < other.start_lnpos


# from_start_lnpos_list
# type example
# List[Tuple[int, linepos.LnPos]
# [(0, (0, 12, 0)),
#  (13, (13, 13)),
#  (14, (14, 81, 1))
#   first int, start_offste
#   2nd int, start offset
#   3nd int, 3nd offset
#   4th int, line sequence

        
class FromToMapper:

    def __init__(self,
                 name: str,
                 # from_start_lnpos_list: List[(int, linepos.LnPos)],
                 # to_start_lnpos_list: List[(int, linepos.LnPos)]):
                 from_start_lnpos_list: List,
                 to_start_lnpos_list: List) -> None:
        
        self.name = name
        
        # we don't trust other and make sure frstart_list is sorted.
        alist = []
        for from_sxlnpos, to_sxlnpos in zip(from_start_lnpos_list, to_start_lnpos_list):
            from_start, from_lnpos = from_sxlnpos
            to_start, to_lnpos = to_sxlnpos
            # using to_lnpos.end, just in case there is a gap, which migth cause two to_lnpos.start to
            # be the same
            alist.append((from_lnpos.start, from_lnpos.end, to_lnpos.start, from_sxlnpos, to_sxlnpos))

        # this is for binary search
        self.frstart_list = []  # type: List[int]
        # this is for mapping, for returning as the offset
        self.tostart_list = []  # type: List[int]
        self.from_start_lnpos_list = []  # type: List[Tuple[int, linepos.LnPos]]
        self.to_start_lnpos_list = []  # type: List[Tuple[int, linepos.LnPos]]

        for from_start, _, to_start, from_sxlnpos, to_sxlnpos in alist:
            self.frstart_list.append(from_start)
            self.tostart_list.append(to_start)
            self.from_start_lnpos_list.append(from_sxlnpos)
            self.to_start_lnpos_list.append(to_sxlnpos)

            # print("from_sxlnpos: {}".format(from_sxlnpos))
            # print("to_sxlnpos: {}".format(to_sxlnpos))


    def get_lnpos_list_se_offsets(self, from_start: int, from_end: int) -> Tuple[List[linepos.LnPos], int, int]:
        start_idx, start_diff = find_index_diff(from_start, self.frstart_list)

        end_idx, end_diff = find_index_diff(from_end, self.frstart_list)

        # print('from_start = {}, from_end = {}, start_idx = {}, end_idx = {}'.format(from_start, from_end, start_idx, end_idx))

        # TODO, jshaw, del
        #if from_start == 99:
        #    for i in range(start_idx, end_idx + 1):
        #        fxstart, fx_lnpos = self.from_start_lnpos_list[i]
        #        print("chosen from: {}".format(fx_lnpos))
        #
        #        zxstart, zx_lnpos = self.to_start_lnpos_list[i]
        #        print("mapped to: {}".format(zx_lnpos))

        # now get the to_list corresponding to all the chosen from_list
        # NOTE: must remove the empty line, otherwise might appear at the end if the from_list, and to_list has
        # mismatch.  The offset at begin and end of block will then mess thing up big time
        
        to_start_lnpos_ediff_list = []  # type: List[StartLnPosDiff]
        for i in range(start_idx, end_idx + 1):
            if (self.to_start_lnpos_list[i][1].is_gap or
                self.to_start_lnpos_list[i][1].start != self.to_start_lnpos_list[i][1].end):
                # need to set the potential_end_diff, just in case if the line got swapped to be the last line for end_offset
                potential_end_diff = self.to_start_lnpos_list[i][1].end - self.to_start_lnpos_list[i][1].start
                # Originally use a list instead of tuple for assign
                # Now, use StartLnPosDiff instead for typing
                to_start_lnpos_ediff_list.append(StartLnPosDiff(self.to_start_lnpos_list[i],
                                                                potential_end_diff))

        # if there is only 1 line, no chance of diff being different, skip
        # print("len(to_start_lnpos_ediff_list) = {}".format(len(to_start_lnpos_ediff_list)))
        if len(to_start_lnpos_ediff_list) > 1:
            # there is some chance if the from and to line got reordered
            # to_start_lnpos_ediff_list[0][1] = start_diff
            to_start_lnpos_ediff_list[-1].diff = end_diff
            
        # order the chosen to_list by its starts
        tmp_to_start_lnpos_ediff_list = to_start_lnpos_ediff_list
        to_start_lnpos_ediff_list = sorted(to_start_lnpos_ediff_list)

        # WARNING: jshaw
        # if lines are really out of order, the start_diff specification is not honored if it is
        # in the middle of the lines.  Our code doesn't check for incomplete line specification in a block
        # of lines when generating the offsets or spans.
        # normally, in those table operations, the whole lines are included, not in the middle.
        # We also don't want to apply this if there is only 1 line.
        if tmp_to_start_lnpos_ediff_list != to_start_lnpos_ediff_list:
            end_diff = to_start_lnpos_ediff_list[-1].diff   # adjust the offset for the last line

        to_lnpos_list = [sxlnposdiff.start_lnpos[1] for sxlnposdiff in to_start_lnpos_ediff_list]

        # the start and end diff really depends on the particular line beign first and last
        
        return to_lnpos_list, start_diff, end_diff
    

    def get_span_list(self, from_start: int, from_end: int) -> List[Dict]:

        # TODO, jshaw, del
        # print("==in fromtomapper.get_span_list({}, {}) ==============".format(from_start, from_end))
        # for (fstart, flnpos), (tstart, tlnpos) in zip(self.from_start_lnpos_list, self.to_start_lnpos_list):
        #    print("23 from({}, {})\tto({}, {})".format(fstart, flnpos, tstart, tlnpos))

        lnpos_list, start_diff, end_diff = \
            self.get_lnpos_list_se_offsets(from_start, from_end)

        #if from_start == 99:
        #    print("lnpos_list= {}, start_diff= {}, end_diff= {}".format(lnpos_list, start_diff, end_diff))

        #if not lnpos_list:
        #    print('\n\n')
        #    print("get_span_list(self, {}, {}) is empty".format(from_start, from_end))
        #    print('lnpos_list: {}'.format(lnpos_list))

        if len(lnpos_list) == 1:
            lnpos = lnpos_list[0]
            return [{'start': lnpos.start + start_diff,
                     'end': lnpos.start + end_diff}]

        result = [lnpos.to_dict() for lnpos in lnpos_list]

        # TODO, jshaw, del
        # if from_start == 99:
        #    print("result: {}".format(result))

        # adjust the start and end offsets
        start_lnpos = lnpos_list[0]
        result[0]['start'] = start_lnpos.start + start_diff
        end_lnpos = lnpos_list[-1]
        result[-1]['end'] = end_lnpos.start + end_diff

        # TODO, jshaw, we can do the collapsing based on line_num
        cur_lnpos_dict = result[0]
        prev_line_num = cur_lnpos_dict['line_num']
        merged_list = [cur_lnpos_dict]
        for lnpos_dict in result[1:]:
            if lnpos_dict.get('gap'):
                # prev_line_num will be -1
                pass
            elif (lnpos_dict['line_num'] == prev_line_num or
                lnpos_dict['line_num'] == prev_line_num + 1):
                cur_lnpos_dict['end'] = lnpos_dict['end']
            else:
                cur_lnpos_dict = lnpos_dict                
                merged_list.append(lnpos_dict)
            prev_line_num = lnpos_dict['line_num']

        # lnpos_dict['line_num'] no longer has correct value
        for lnpos_dict in merged_list:
            del lnpos_dict['line_num']

        return merged_list

    # this is destructive
    def adjust_fromto_offsets(self, ant_list: List) -> None:
        for antx in ant_list:
            # print("ant start = {}, end = {}".format(antx['start'], antx['end']))
            corenlp_start = antx['start']
            corenlp_end = antx['end']
            antx['corenlp_start'] = corenlp_start
            antx['corenlp_end'] = corenlp_end

            span_list = self.get_span_list(corenlp_start, corenlp_end)

            antx['start'] = span_list[0]['start']
            antx['end'] = span_list[-1]['end']
            antx['span_list'] = span_list

    # this is destructive
    def adjust_provants_fromto_offsets(self, ant_list: List[ProvisionAnnotation]) -> List[ProvisionAnnotation]:
        result = []
        for antx in ant_list:
            # print("ant start = {}, end = {}".format(antx['start'], antx['end']))
            raw_start, raw_end, label = antx.to_tuple()

            span_list = self.get_span_list(raw_start, raw_end)

            corenlp_start = span_list[0]['start']
            corenlp_end = span_list[-1]['end']
            result.append(ProvisionAnnotation(corenlp_start, corenlp_end, label))
        return result

    def get_se_offsets(self, start:int, end:int) -> Tuple[int, int]:
        result_start = find_offset_to(start, self.frstart_list, self.tostart_list)
        result_end = find_offset_to(end, self.frstart_list, self.tostart_list)

        return result_start, result_end

    def get_fromto_start_lnpos_lists(self) -> Tuple[List[Tuple[int, linepos.LnPos]],
                                                    List[Tuple[int, linepos.LnPos]]]:
        return self.from_start_lnpos_list, self.to_start_lnpos_list



# para_list has the following format
# ([((2206, 2344, 11), (2183, 2321, 11))], '(a)           The definition of "Applicable Committed Loan Margin" in Article 1 is hereby amended and restated to read in full as follows:', [('sechead', '2.', 'Amendments  to  Credit  Agreement.     ', 52)]),
# the type is
# List, a tuple of
#    span_se_list: List[Tuple[linepos.LnPos, linepos.LnPos]]
#    line: str
#    attr_list: List[Tuple]
# this is sorted_by_from
def paras_to_fromto_mapper_sorted_by_from(para_list: List[Tuple[List[Tuple[linepos.LnPos,
                                                                           linepos.LnPos]],
                                                                str,
                                                                List[Tuple[Any]]]]) -> FromToMapper:
    alist = []
    for span_se_list, line, attr_list in para_list:
        # print("  span_se_list: {}".format(span_se_list))
        # for (from_lnpos, to_lnpos) in span_se_list:
        #    print("    from_lnpos = {}, to_lnpos = {}".format(from_lnpos, to_lnpos))

        # at this point from_lnpos is for original text
        # to_lnpos is for nlp text
        for (from_lnpos, to_lnpos) in span_se_list:
            # intentionally not use from_lnpos.end
            # using to_lnpos.end, just in case there is a gap, which migth cause two to_lnpos.start to
            # be the same
            alist.append((from_lnpos.start, from_lnpos.end, to_lnpos.start, from_lnpos, to_lnpos))

    # ordered by to_start, because this is where we will map from,
    # and it need be ordered
    sorted_alist = sorted(alist)

    from_list = [(a, c) for a, a2, b, c, d in sorted_alist]
    to_list = [(b, d) for a, a2, b, c, d in sorted_alist]

    return FromToMapper('an offset mapper', from_list, to_list)
