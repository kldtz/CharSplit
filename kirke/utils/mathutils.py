# pylint: disable=unused-import
from typing import List, Tuple

def start_end_overlap(stend1, stend2):
    start1, end1 = stend1
    start2, end2 = stend2
    return start1 < end2 and start2 < end1

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

def remove_subsumed(alist):
    if not alist:
        return []
    sorted_by_len = []
    for elt in alist:
        elt_len = elt[1] - elt[0]
        sorted_by_len.append((elt_len, elt))
    sorted_by_len.sort(reverse=True)

    sorted_by_len = [elt for alen, elt in sorted_by_len]
    result = [sorted_by_len[0]]
    for elt in sorted_by_len[1:]:
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
