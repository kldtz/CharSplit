#!/usr/bin/env python

import sys
import unicodedata
from collections import defaultdict

print("max unicode = " + str(sys.maxunicode))

unicode_category = defaultdict(list)
for c in map(chr, range(sys.maxunicode + 1)):
    unicode_category[unicodedata.category(c)].append(c)

for i in range(sys.maxunicode + 1):
    c = chr(i)
    # unicode_category[unicodedata.category(c)].append(c)
    print(str(i) + "\t" + unicodedata.category(c))
    
