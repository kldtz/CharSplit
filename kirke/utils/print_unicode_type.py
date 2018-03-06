#!/usr/bin/env python3

import sys
import unicodedata
from collections import defaultdict

print("max unicode = " + str(sys.maxunicode))


UNICODE_CATEGORY = defaultdict(list)
for c in map(chr, range(sys.maxunicode + 1)):
    UNICODE_CATEGORY[unicodedata.category(c)].append(c)

for i in range(sys.maxunicode + 1):
    c = chr(i)
    # UNICODE_CATEGORY[unicodedata.category(c)].append(c)
    print(str(i) + "\t" + unicodedata.category(c))
