#!/usr/bin/env python

import sys
import unicodedata
from collections import defaultdict

UNICODE_CATEGORY = defaultdict(list)

print("max unicode\t" + str(sys.maxunicode + 1))

for c in map(chr, range(sys.maxunicode + 1)):
    UNICODE_CATEGORY[unicodedata.category(c)].append(c)

for ucat, alist in UNICODE_CATEGORY.items():
    print((ucat, alist[:10]))
