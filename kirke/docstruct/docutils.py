import bisect
from collections import defaultdict

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

        overlap_spans2 = []
        for gap_span in gap_span_list:
            if mathutils.start_end_overlap(ant_se, gap_span):
                overlap_spans2.append(gap_span)
                # print('overlap2 {}, {}'.format(se_ant_list_map[ant_se][0], gap_span))

        #if overlap_spans2 != overlap_spans:
        #    print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", file=sys.stderr)
        #    print("overlap_span = {}".format(overlap_spans), file=sys.stderr)
        #    print("overlap_span2 = {}".format(overlap_spans2), file=sys.stderr)

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

            # endpoints_st_list = ['{}:{}'.format(endpoint_list[i], endpoint_list[i+1]) for i in range(0, len(endpoint_list), 2)]
            endpoints_dict_list = [{ 'start': endpoint_list[i],
                                     'end': endpoint_list[i+1]}
                                   for i in range(0, len(endpoint_list), 2)]
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

        overlap_spans2 = []
        for gap_span in gap_span_list:
            if mathutils.start_end_overlap(ant_se, gap_span):
                overlap_spans2.append(gap_span)
                # print('overlap2 {}, {}'.format(se_ant_list_map[ant_se][0], gap_span))

        #if overlap_spans2 != overlap_spans:
        #    print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", file=sys.stderr)
        #    print("overlap_span = {}".format(overlap_spans), file=sys.stderr)
        #    print("overlap_span2 = {}".format(overlap_spans2), file=sys.stderr)
            
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

