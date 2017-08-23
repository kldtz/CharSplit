
def start_end_overlap(stend1, stend2):
    start1, end1 = stend1
    start2, end2 = stend2
    return start1 < end2 and start2 < end1

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
