#!/usr/bin/env python3

from collections import OrderedDict, defaultdict
import re
import json
import operator


import sys

cat_freq_map = defaultdict(int)
cat_scorelist_map = defaultdict(list)

TAG_LINE_PAT = re.compile(r'(.+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+')

with open('kkk5.scores', 'rt') as fin:
    for line in fin:
        mat = TAG_LINE_PAT.match(line)
        if mat:
            tag = mat.group(1).strip()
            precision = float(mat.group(2))
            recall = float(mat.group(3))
            f1 = float(mat.group(4))
            support = int(mat.group(5))

            cat_scorelist_map[tag].append((precision, recall, f1, support))
            cat_freq_map[tag] += support

    for cat, freq in sorted(cat_freq_map.items(), key=operator.itemgetter(1), reverse=True):
        alist = cat_scorelist_map[cat]
        sum_f1 = 0
        for ascore in alist:
            prec, recall, f1, support = ascore
            sum_f1 += f1
        avg_f1 = sum_f1 / len(alist)

        if avg_f1 > 0.7:
            print("cat ok\t{}\t{}".format(cat, avg_f1))
            for ascore in alist:
                print(('ok', cat, ascore))
        else:
            print("cat NOT ok\t{}\t{}".format(cat, avg_f1))
            for ascore in alist:
                print(('NOT ok', cat, ascore))
    

