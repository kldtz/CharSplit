# pylint: disable=unused-import
from typing import Any, Dict, FrozenSet, List, Set, Tuple


def start_end_overlap(stend1: Tuple[int, int],
                      stend2: Tuple[int, int]) \
                      -> bool:
    start1, end1 = stend1
    start2, end2 = stend2
    return start1 < end2 and start2 < end1


# we don't assume se_list is sorted
def is_overlap_with_se_list(se_to_check: Tuple[int, int],
                            se_list: List[Tuple[int, int]]) \
                            -> bool:
    for stend in se_list:
        if start_end_overlap(se_to_check, stend):
            return True
    return False


# 1st includes 2nd
# [    stend1    ]
#    [ stend2 ]
def start_end_subsume(stend1, stend2):
    start1, end1 = stend1
    start2, end2 = stend2
    return start1 <= start2 and end1 >= end2


def is_subsumed(alist, elt):
    # print("is_subsumed(): {}".format(alist))
    for anitem in alist:
        if start_end_subsume((anitem[0],
                              anitem[1]),
                             (elt[0],
                              elt[1])):
            return True
    return False

# cannot type this (mypy)
# becuase alist can be
#    List[Tuple[int, int, Any]] or
#    List[Tuple[int, int, str, str]]
def remove_subsumed(alist):
    if not alist:
        return []
    sorted_by_len = []  # type: List[Tuple[int, Any]]
    for elt in alist:
        elt_len = elt[1] - elt[0]
        sorted_by_len.append((elt_len, elt))
    sorted_by_len.sort(reverse=True)

    sorted_by_len_2 = [elt for alen, elt in sorted_by_len]
    result = [sorted_by_len_2[0]]
    for elt in sorted_by_len_2[1:]:
        if not is_subsumed(result, elt):
            result.append(elt)
    return result


def offset_percentage(offsets1, offsets2):
    """A metric for how exact a true positive matches an annotated provision."""
    distance = abs(offsets2[0] - offsets1[0]) + abs(offsets2[1] - offsets1[1])
    total_characters = (offsets1[1] - offsets1[0]) + (offsets2[1] - offsets2[0])
    return 100 * max(1 - distance / total_characters, 0)


def offset_score(answers_offsets, pred_offsets):
    """Returns the best offset_percentage. Returned 0s should be ignored."""
    percentages = [offset_percentage(a, pred_offsets) for a in answers_offsets]
    return max(percentages) if percentages else 0

def find_in_list_of_set(list_of_set, elt):
    for aset in list_of_set:
        if elt in aset:
            return aset
    return set()


# pairs is a list of 2-tuples
def pairs_to_sets(pairs):
    result = []  # list of sets
    # pylint: disable=invalid-name
    for x1, x2 in pairs:
        set1 = find_in_list_of_set(result, x1)
        if set1:
            if x2 in set1:  # they are already in a set, done
                # pair_finished = True
                pass
            else:  # x1 is found, but x2 is not in the same set!?
                set2 = find_in_list_of_set(result, x2)
                if set2:  # x1 and x2 are in different set, merge them
                    set1.update(set2)
                    result.remove(set2)
                else:  # x2 is not in any set
                    set1.add(x2)
        else:  # x1 is not found
            set2 = find_in_list_of_set(result, x2)
            if set2:
                set2.add(x1)
            else: # both x1 and x2 are not found
                result.append({x1, x2})
    return result


def choose_after(xval: int, other_list: List[int]) -> int:
    for yval in other_list:
        if yval > xval:
            return yval
    # if not found, just take first one
    return other_list[0]


def choose_before(xval: int, other_list: List[int]) -> int:
    for yval in other_list:
        if yval < xval:
            return yval
    # if not found, just take first one
    return other_list[0]


def choose_closest(xval: int,
                   other_list: List[int]) -> int:
    diff_list = []  # type: List[Tuple[int, int]]
    for idx, yval in enumerate(other_list):
        diff_list.append((abs(yval - xval), idx))
    sorted_diff_list = sorted(diff_list)
    _, chosen_idx = sorted_diff_list[0]
    return other_list[chosen_idx]


def half_round(aval: float) -> float:
    twice = aval * 2
    chopped = round(twice)
    half = round(chopped / 2, 1)
    return half


def one_fifth_round(aval: float) -> float:
    five_times = aval * 5
    chopped = round(five_times)
    one_fifth = round(chopped / 5, 1)
    return one_fifth


def get_mode_in_list(alist):
    """Return the most frequent element in the list.

    Note: if there is a tie, this is undefined among the most frequent
          entities.
    """
    return max(set(alist), key=alist.count)


def find_overlap_ids(id_se: Tuple[int, int, int],
                     id_se_list: List[Tuple[int, int, int]]) -> List[int]:
    result = []  # type: List[int]
    unused_xid, xstart, xend = id_se
    for yid, ystart, yend in id_se_list:
        if xstart < yend and ystart < xend:
            result.append(yid)
    return result


def find_overlaps_in_id_se_list(id_se_list: List[Tuple[int, int, int]]) -> List[List[int]]:
    overlap_pair_list = []  # type: List[Tuple[int, int]]
    id_overlap_ids_map = {}  # type: Dict[int, Set[int]]
    # pylint: disable=consider-using-enumerate
    for count_i in range(len(id_se_list)):
        i_id_se = id_se_list[count_i]

        is_overlap_id_list = find_overlap_ids(i_id_se, id_se_list[count_i+1:])

        for count_j in is_overlap_id_list:
            overlap_pair_list.append((count_i, count_j))
            # print("overlap_pair: {}".format((count_i, count_j)))
        id_overlap_ids_map[count_i] = set(is_overlap_id_list + [count_i])

    # for tid, overlap_ids in id_overlap_ids_map.items():
    #     print("  id = {}, overlap_ids = {}".format(tid, overlap_ids))
    for tid in id_overlap_ids_map:

        overlap_ids = id_overlap_ids_map[tid]

        for oid in list(overlap_ids):
            if oid != tid:
                oid_overlap_ids = id_overlap_ids_map[oid]

                if oid_overlap_ids != overlap_ids:
                    overlap_ids.update(oid_overlap_ids)
                    # overlap_ids is already updated, so no need for
                    # id_overlap_ids_map[tid] = overlap_ids
                    id_overlap_ids_map[oid] = overlap_ids

        # print("  id = {}, overlap_ids = {}".format(tid, overlap_ids))

    out_group_list = []  # type: List[List[int]]
    seen_set = set([])  # type: Set[FrozenSet[int]]
    for tid in id_overlap_ids_map:
        overlap_set = frozenset(id_overlap_ids_map[tid])
        if len(overlap_set) > 1 and \
           overlap_set not in seen_set:
            out_group_list.append(list(overlap_set))
            seen_set.add(overlap_set)

    # print("overlap_pair_list: {}".format(overlap_pair_list))
    # print("out_group_list: {}".format(out_group_list))
    return out_group_list


def calc_float_list_mode(ilist: List[float], ndigits: int = 2) -> float:
    if ndigits != -1:
        ilist = [round(val, ndigits) for val in ilist]
    # This can be more optimized using a defaultdict.
    # Repeatedly calling ilist.count is not efficient.
    return max(set(ilist), key=ilist.count)


def calc_interval_overlap_percent(top: int,
                                  bot: int,
                                  prev_block_top: int,
                                  prev_block_bot: int) -> float:
    diff = min(bot, prev_block_bot) - max(top, prev_block_top)
    if prev_block_bot - prev_block_top > bot - top:
        return diff / (prev_block_bot - prev_block_top)
    return diff / (bot - top)


def is_interval_overlap(top: int,
                        bot: int,
                        prev_block_top: int,
                        prev_block_bot: int,
                        threshold: float = 0.4) \
                        -> bool:
    is_overlap = top < prev_block_bot and \
                 prev_block_top < bot
    if is_overlap:
        perc_overlap = calc_interval_overlap_percent(top,
                                                     bot,
                                                     prev_block_top,
                                                     prev_block_bot)
        if perc_overlap >= threshold:
            return True
    return False


def is_rect_overlap(bot_left_1: Tuple[int, int],
                    top_right_1: Tuple[int, int],
                    bot_left_2: Tuple[int, int],
                    top_right_2: Tuple[int, int]) \
                    -> bool:
    # if one rectagle is on left side of the other
    if bot_left_1[0] > top_right_2[0] or \
       bot_left_2[0] > top_right_1[0]:
        return False

    # if one rectagle is is above other
    if bot_left_1[1] < top_right_2[1] or \
       bot_left_2[1] < top_right_1[1]:
        return False

    return True


def rect_tblr_to_rect_points(tblr: Tuple[int, int, int, int]) \
    -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # bl, tr
    return (tblr[2], tblr[1]), (tblr[3], tblr[0])


# we don't assume se_list is sorted
def is_overlap_with_rect_list(rect_to_check: Tuple[Tuple[int, int], Tuple[int, int]],
                              rect_list: List[Tuple[Tuple[int, int], Tuple[int, int]]]) \
                              -> bool:
    ck_bl, ck_tr = rect_to_check
    for x_bl, x_tr in rect_list:
        if is_rect_overlap(ck_bl, ck_tr,
                           x_bl, x_tr):
            return True
    return False
