
def start_end_overlap(stend1, stend2):
    start1, end1 = stend1
    start2, end2 = stend2
    return start1 < end2 and start2 < end1
