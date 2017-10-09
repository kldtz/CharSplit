import bisect
from collections import defaultdict
from operator import itemgetter

from kirke.utils import mathutils, strutils

def find_offset_to(fromx: int, from_list, to_list):
    return find_offset_to_linear(fromx, from_list, to_list)

# binary search version
## there is some error in binary_search version, result
## diff from find_offset_to_linear in certain cases???
def find_offset_to_binary(fromx: int, from_list, to_list):

    # find rightmost value less than or equal to fromx
    found_i = bisect.bisect_right(from_list, fromx)
    if found_i:
        if fromx == from_list[found_i-1]:
            return to_list[found_i-1]
        diff = fromx - from_list[found_i-1]
        return to_list[found_i-1] + diff

    return -1


# linear version
def find_offset_to_linear(fromx: int, from_list, to_list):
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


def lnpos2dict(start, end, lnpos):
    adict = {'start': start,
             'end': end,
             'line_num': lnpos.line_num}
    if lnpos.is_gap:
        adict['gap'] = True
    return adict


# startx and endx are related to to_list
# we want the offset in from_list
def find_se_offset_list(startx, endx, from_list, to_list):

    # result is an offset list
    result = []

    start_lnpos = None
    start_diff = 0
    start_idx = 0
    for i, ((fstart, from_lnpos), (tstart, to_lnpos)) in enumerate(zip(from_list, to_list)):
        if fstart == startx:
            start_lnpos = to_lnpos
            start_diff = 0
            start_idx = i
            break
        elif fstart > startx:
            _, start_lnpos = to_list[i-1]
            prev_start, _ = from_list[i-1]
            start_diff = startx - prev_start
            start_idx = i - 1
            break

    end_lnpos = None
    end_diff = 0
    end_idx = 0  # this is inclusive
    for i, ((fstart, from_lnpos), (tstart, to_lnpos)) in enumerate(zip(from_list, to_list)):
        if fstart == endx:
            end_lnpos = to_lnpos
            end_diff = 0
            end_idx = i
            break
        elif fstart > endx:
            _, end_lnpos = to_list[i-1]
            prev_start, _ = from_list[i-1]
            end_diff = endx - prev_start
            end_idx = i - 1
            break

    # they are in the same line
    if start_idx == end_idx:
        result = [{'start': start_lnpos.start + start_diff,
                   'end': start_lnpos.start + end_diff}]
    else:
        # skipping all middle parts for now
        result = [lnpos2dict(start_lnpos.start + start_diff,
                             start_lnpos.end,
                             start_lnpos),
                  lnpos2dict(end_lnpos.start,
                             end_lnpos.start + end_diff,
                             end_lnpos)]
        result = sorted(result, key=itemgetter('start'))
    return result


def read_fromto_json(file_name: str):
    fromto_list = strutils.load_json_list(file_name)

    alist = []
    for fromto in fromto_list:
        from_start = fromto['from_start']
        # from_end = fromto['from_end']
        to_start = fromto['to_start']
        # to_end = fromto['to_end']
        alist.append((from_start, to_start))
        # alist.append((from_end, to_end))

    sorted_alist = sorted(alist)
    
    from_list = [a for a,b in sorted_alist]
    to_list = [b for a,b in sorted_alist]    
    return from_list, to_list



def update_ants_gap_spans(prov_labels_map, gap_span_list, doc_text):

    ant_list = []
    for provision, tmp_ant_list in prov_labels_map.items():
        for antx in tmp_ant_list:
            ant_list.append(antx)

    se_ant_list_map = defaultdict(list)
    for ant in ant_list:
        se = (ant['start'], ant['end'])
        se_ant_list_map[se].append(ant)
    ant_se_list = sorted(se_ant_list_map.keys())
    # print("  ant_se_list: {}".format(ant_se_list))
    # print("gap_span_list: {}".format(gap_span_list))

    min_possible_j, jmax = 0, len(gap_span_list)

    for ant_se in ant_se_list:

        overlap_spans = []
        for j in range(min_possible_j, jmax):
            gap_span = gap_span_list[j]
            if gap_span[1] < ant_se[0]:
                min_possible_j = j+1
            if mathutils.start_end_overlap(ant_se, gap_span):
                overlap_spans.append(gap_span)
                # print('overlap {}, {}'.format(se_ant_list_map[ant_se][0], gap_span))
            # because annotations can overlap,
            # this guarantee is false
            # if gap_span[0] > ant_end:
            #    break

        if overlap_spans:
            # adjusted_spanst_list = []
            endpoint_list = [ant_se[0], ant_se[1]]
            max_end = len(doc_text)
            for gap_span in overlap_spans:
                tmp_start = gap_span[0]
                # find the first space after an non-space to the left of gap
                while tmp_start > 0 and strutils.is_space(doc_text[tmp_start-1]):
                    tmp_start -= 1
                tmp_end = gap_span[1]

                # find the first non-space to the right of the gap
                while tmp_end < max_end and strutils.is_space(doc_text[tmp_end]):
                    tmp_end += 1

                # adjusted_spanst_list.append("{}:{}".format(tmp_start, tmp_end))
                endpoint_list.append(tmp_start)
                endpoint_list.append(tmp_end)
                # print("adjusted span {}:{} -> {}:{}".format(gap_span[0], gap_span[1], tmp_start, tmp_end))
            endpoint_list.sort()

            # add start, end only if not empty
            endpoints_dict_list = []
            for i in range(0, len(endpoint_list), 2):
                aa_start = endpoint_list[i]
                aa_end = endpoint_list[i+1]
                if strutils.remove_space_nl(doc_text[aa_start:aa_end]):
                    endpoints_dict_list.append({ 'start': aa_start,
                                                 'end': aa_end})

            # spans_st = ','.join(adjusted_spanst_list)
            # spans_st = ','.join([str(endpoint) for endpoint in endpoint_list])
            spans_st = endpoints_dict_list
        else:
            spans_st = [{'start': ant_se[0],
                         'end': ant_se[1]}]

        se_ant_list = se_ant_list_map[ant_se]
        for antx in se_ant_list:
            antx['span_list'] = spans_st


### TODO, jshaw, to be removed because upgrade to ebantdoc2
def update_ant_spans(ant_list, gap_span_list, doc_text):

    se_ant_list_map = defaultdict(list)
    for ant in ant_list:
        se = (ant['start'], ant['end'])
        se_ant_list_map[se].append(ant)
    ant_se_list = sorted(se_ant_list_map.keys())
    # print("  ant_se_list: {}".format(ant_se_list))
    # print("gap_span_list: {}".format(gap_span_list))

    min_possible_j, jmax = 0, len(gap_span_list)    
    
    for ant_se in ant_se_list:

        overlap_spans = []
        for j in range(min_possible_j, jmax):
            gap_span = gap_span_list[j]
            if gap_span[1] < ant_se[0]:
                min_possible_j = j+1
            if mathutils.start_end_overlap(ant_se, gap_span):
                overlap_spans.append(gap_span)
                # print('overlap {}, {}'.format(se_ant_list_map[ant_se][0], gap_span))
            # because annotations can overlap,
            # this guarantee is false
            # if gap_span[0] > ant_end:
            #    break

        if overlap_spans:
            # adjusted_spanst_list = []
            endpoint_list = [ant_se[0], ant_se[1]]
            max_end = len(doc_text)
            for gap_span in overlap_spans:
                tmp_start = gap_span[0]
                # find the first space after an non-space to the left of gap
                while tmp_start > 0 and strutils.is_space(doc_text[tmp_start-1]):
                    tmp_start -= 1
                tmp_end = gap_span[1]
                
                # find the first non-space to the right of the gap
                while tmp_end < max_end and strutils.is_space(doc_text[tmp_end]):
                    tmp_end += 1

                # adjusted_spanst_list.append("{}:{}".format(tmp_start, tmp_end))
                endpoint_list.append(tmp_start)
                endpoint_list.append(tmp_end)
                # print("adjusted span {}:{} -> {}:{}".format(gap_span[0], gap_span[1], tmp_start, tmp_end))
            endpoint_list.sort()

            endpoints_st_list = ['{}:{}'.format(endpoint_list[i], endpoint_list[i+1]) for i in range(0, len(endpoint_list), 2)]
            # spans_st = ','.join(adjusted_spanst_list)
            # spans_st = ','.join([str(endpoint) for endpoint in endpoint_list])
            spans_st = ','.join(endpoints_st_list)
        else:
            spans_st = '{}:{}'.format(ant_se[0], ant_se[1])

        se_ant_list = se_ant_list_map[ant_se]
        for antx in se_ant_list:
            antx['start_end_span_list'] = spans_st

""" from htmltxtparser.py
def paras_to_fromto_lists(para_list):
    alist = []
    for span_frto_list, line, attr_list in para_list:
        for (from_start, from_end), (to_start, to_end) in span_frto_list:
            alist.append((from_start, to_start))

    # in HTML files, the 'from' and 'to' are guaranteed to be in order, so
    # this code is slightly different from pdftxtparser.paras_to_fromto_lists.
    # otherwise, should order by 'to_start'
    sorted_alist = sorted(alist)

    from_list = [a for a,b in sorted_alist]
    to_list = [b for a,b in sorted_alist]
    return from_list, to_list
"""

def paras_to_fromto_lists(para_list):
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


def span_frto_list_to_fromto(span_frto_list):
    from_lnpos, to_lnpos = span_frto_list[0]
    from_start, from_end = from_lnpos.start, from_lnpos.end
    to_start, to_end = to_lnpos.start, to_lnpos.end

    for span_fromto in span_frto_list[1:]:
        from_lnpos2, to_lnpos2 = span_fromto
        fe2 = from_lnpos2.end
        te2 = to_lnpos2.end
        if fe2 > from_end:
            from_end = fe2
        if te2 > to_end:
            to_end = te2
    return (from_start, from_end), (to_start, to_end)
