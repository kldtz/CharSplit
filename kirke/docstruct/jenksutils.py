#!/usr/bin/env python3

# From https://gist.github.com/drewda/1299198


# code from http://danieljlewis.org/files/2010/06/Jenks.pdf
# described at http://danieljlewis.org/2010/06/07/jenks-natural-breaks-algorithm-in-python/

# pylint: disable=invalid-name, too-many-locals, too-many-branches
def getJenksBreaks(data_list, num_class):
    data_list.sort()
    mat1 = []
    for i in range(0, len(data_list) + 1):
        temp = []
        for j in range(0, num_class + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(0, len(data_list) + 1):
        temp = []
        for j in range(0, num_class + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, num_class + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(data_list) + 1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2, len(data_list) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(data_list[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, num_class + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(data_list)
    kclass = []
    for i in range(0, num_class + 1):
        kclass.append(0)
    kclass[num_class] = float(data_list[len(data_list) - 1])
    countNum = num_class
    while countNum >= 2:  # print "rank = " + str(mat1[k][countNum])
        tid = int((mat1[k][countNum]) - 2)
        # print "val = " + str(data_list[id])
        kclass[countNum - 1] = data_list[tid]
        k = int((mat1[k][countNum] - 1))
        countNum -= 1
    return kclass


def getGVF(data_list, num_class):
    """
    The Goodness of Variance Fit (GVF) is found by taking the
    difference between the squared deviations
    from the array mean (SDAM) and the squared deviations from the
    class means (SDCM), and dividing by the SDAM
    """
    breaks = getJenksBreaks(data_list, num_class)
    data_list.sort()
    listMean = sum(data_list) / len(data_list)
    print(listMean)
    SDAM = 0.0
    for data in data_list:
        sqDev = (data - listMean) ** 2
        SDAM += sqDev
    SDCM = 0.0
    for i in range(0, num_class):
        if breaks[i] == 0:
            classStart = 0
        else:
            classStart = data_list.index(breaks[i])
            classStart += 1
        classEnd = data_list.index(breaks[i + 1])
        classList = data_list[classStart:classEnd + 1]
        classMean = sum(classList) / len(classList)
        print(classMean)
        preSDCM = 0.0
        for aclass in classList:
            sqDev2 = (aclass - classMean) ** 2
            preSDCM += sqDev2
        SDCM += preSDCM
    return (SDAM - SDCM) / SDAM


# written by Drew
# used after running getJenksBreaks()
def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1

NUM_DEFAULT_CLUSTER = 5

# Merge breaks that are really adjacent.  Keep the
# largest one among the adjacent ones.
def collapse_close_bks(break_list):
    prev_break = break_list[0]
    cur_group = [prev_break]
    group_list = [cur_group]
    for bk in break_list[1:]:
        if bk - prev_break <= 2.5:  # 2.5 is empirical
            cur_group.append(bk)
        else:
            cur_group = [bk]
            group_list.append(cur_group)
        prev_break = bk
    # now take the max out of each group
    result = []
    for agroup in group_list:
        result.append(max(agroup))
    return result


# pylint: disable=too-few-public-methods
class Jenks:

    def __init__(self, values):
        self.vals = values
        # print("values: {}".format(values))
        bks = getJenksBreaks(values, NUM_DEFAULT_CLUSTER)
        # print("bks: {}".format(bks))
        # Merge breaks that are really adjacent.  Keep the
        # largest one among the adjacent ones.
        bks = collapse_close_bks(bks)
        # print("collapsed bks: {}".format(bks))

        left_count, center_count, right_count = 1, 1, 1
        aligned_list = ['LF0']  # LF0 will never be used
        for brk_val in bks[1:]:
            if brk_val < 300:
                aligned_list.append("LF{}".format(left_count))
                left_count += 1
            elif brk_val < 480:
                aligned_list.append("CN{}".format(center_count))
                center_count += 1
            else:
                aligned_list.append("RT{}".format(right_count))
                right_count += 1
        self.bks = bks
        self.aligned_list = aligned_list

    def classify(self, value):
        breaks = self.bks
        # print("classify, val = {}, breaks = {}".format(value, self.bks))
        for i in range(1, len(breaks)):
            if value <= breaks[i]:
                return self.aligned_list[i]
            # This is no longer needed because change '<' above to '<='.
            # So far, <= seems to be more appropriate for us.
            # check for value = 72.2, [i-1] = 72.2 and [i] = 72.2
            #if i > 1 and value == breaks[i-1] and value == breaks[i]:
            #    return self.aligned_list[i-1]
        return self.aligned_list[len(breaks) - 1]



# returns num_cluster list of list of values
def cluster1d(values, num_cluster):
    # get the separating values
    # print("cluster1d({},{})".format(values, num_cluster))
    # values = sorted(values)

    bks = getJenksBreaks(values, num_cluster)
    # print("bks:", bks)

    groups = breaks_to_groups(values, bks)
    return bks, groups

def cluster_5align_map(values):
    NUM_CLUSTER = 5
    bks = getJenksBreaks(values, NUM_CLUSTER)
    # print("bks:", bks)

    left_count, center_count, right_count = 1, 1, 1
    aligned_list = ['LF0']  # LF0 will never be used
    for brk_val in bks[1:]:
        if brk_val < 300:
            aligned_list.append("LF{}".format(left_count))
            left_count += 1
        elif brk_val < 480:
            aligned_list.append("CN{}".format(center_count))
            center_count += 1
        else:
            aligned_list.append("RT{}".format(right_count))
            right_count += 1

    xstart_align_map = breaks_to_aligned_map(values, bks, aligned_list)
    return xstart_align_map

def breaks_to_aligned_map(values, breaks, aligned_list):
    topval_index = 1
    topval = breaks[topval_index]
    aligned_label = aligned_list[topval_index]
    aligned_map = {}
    for i in values:
        if i <= topval:
            aligned_map[i] = aligned_label
        else:
            topval_index += 1
            if topval_index < len(breaks):
                topval = breaks[topval_index]
                aligned_label = aligned_list[topval_index]
                aligned_map[i] = aligned_label
            else:
                break
    return aligned_map


def breaks_to_groups(values, breaks):
    # print("breaks: {}".format(breaks))
    topval_index = 1
    topval = breaks[topval_index]
    group_num = 0
    cur_group = []
    result = []
    for i in values:
        if i <= topval:
            cur_group.append(i)
        else:
            result.append(cur_group)
            cur_group = [i]
            group_num += 1
            topval_index += 1
            if topval_index < len(breaks):
                topval = breaks[topval_index]
            else:
                break
    result.append(cur_group)
    return result


def main():
    # pylint: disable=line-too-long
    x = [54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 62, 62, 62, 66, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
         # pylint: disable=line-too-long
         67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67,
         # pylint: disable=line-too-long
         67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 68, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72,
         # pylint: disable=line-too-long
         72, 72, 72, 72, 72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
         # pylint: disable=line-too-long
         73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
         # pylint: disable=line-too-long
         73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
         # pylint: disable=line-too-long
         73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
         # pylint: disable=line-too-long
         73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74,
         # pylint: disable=line-too-long
         74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         # pylint: disable=line-too-long
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         # pylint: disable=line-too-long
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         # pylint: disable=line-too-long
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         # pylint: disable=line-too-long
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75,
         # pylint: disable=line-too-long
         75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76,
         # pylint: disable=line-too-long
         76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76,
         # pylint: disable=line-too-long
         76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 77, 77, 77, 77, 77, 77, 77, 77,
         # pylint: disable=line-too-long
         77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 87, 90, 101, 101, 101, 101, 102, 102, 102, 102,
         # pylint: disable=line-too-long
         102, 102, 105, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108,
         # pylint: disable=line-too-long
         108, 108, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109,
         # pylint: disable=line-too-long
         109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109,
         # pylint: disable=line-too-long
         109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109,
         # pylint: disable=line-too-long
         109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109,
         # pylint: disable=line-too-long
         109, 109, 109, 109, 109, 109, 109, 109, 109, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
         # pylint: disable=line-too-long
         110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
         # pylint: disable=line-too-long
         110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
         # pylint: disable=line-too-long
         110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
         # pylint: disable=line-too-long
         110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110,
         # pylint: disable=line-too-long
         110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 111, 111, 111, 111, 111, 111, 111, 111,
         # pylint: disable=line-too-long
         111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111,
         # pylint: disable=line-too-long
         111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111,
         # pylint: disable=line-too-long
         111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111,
         # pylint: disable=line-too-long
         111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111,
         # pylint: disable=line-too-long
         111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111,
         # pylint: disable=line-too-long
         111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111,
         # pylint: disable=line-too-long
         111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111,
         # pylint: disable=line-too-long
         111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111,
         # pylint: disable=line-too-long
         111, 111, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
         # pylint: disable=line-too-long
         112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112,
         # pylint: disable=line-too-long
         112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 113, 113, 113,
         # pylint: disable=line-too-long
         113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113,
         # pylint: disable=line-too-long
         113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 114, 114, 114,
         # pylint: disable=line-too-long
         114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 122, 126, 130, 137, 144, 145, 145, 145, 145, 146, 146,
         # pylint: disable=line-too-long
         146, 146, 146, 146, 146, 146, 146, 146, 146, 146, 146, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147,
         # pylint: disable=line-too-long
         147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 147, 148, 148,
         # pylint: disable=line-too-long
         148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 148, 149, 149, 149, 149, 149, 149, 149, 150, 150, 150,
         # pylint: disable=line-too-long
         150, 153, 173, 173, 173, 173, 173, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 183, 192, 194, 195,
         # pylint: disable=line-too-long
         200, 208, 210, 210, 213, 221, 226, 229, 235, 243, 253, 255, 256, 257, 269, 271, 276, 276, 276, 277, 277, 278,
         # pylint: disable=line-too-long
         278, 278, 278, 279, 279, 280, 288, 289, 295, 295, 295, 300, 300, 300, 301, 302, 302, 302, 302, 302, 302, 303,
         # pylint: disable=line-too-long
         303, 303, 303, 303, 304, 304, 304, 304, 304, 304, 304, 304, 304, 304, 304, 304, 304, 304, 305, 305, 305, 305,
         # pylint: disable=line-too-long
         305, 305, 305, 305, 306, 306, 306, 306, 306, 306, 307, 307, 307, 308, 308, 314, 331, 333, 351, 366, 366, 367,
         # pylint: disable=line-too-long
         370, 397, 545, 545, 549, 549, 549, 549, 549, 549, 549, 549, 549, 549, 549, 549, 549, 549, 549, 549, 549, 549,
         549, 550, 550, 550, 550, 550, 550, 551, 555, 555, 556, 556]
    bks = getJenksBreaks(x, 5)
    print("bks:", bks)  # print("result:", classify(8, bks))
    groups = breaks_to_groups(x, bks)
    for group_num, cur_group in enumerate(groups, 1):
        print("group[{}], size:{} = {}".format(group_num, len(cur_group), cur_group))

if __name__ == '__main__':
    main()
