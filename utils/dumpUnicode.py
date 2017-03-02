#!/usr/bin/env python3

import sys
import unicodedata
from collections import defaultdict

unicode_category = defaultdict(list)

print("max unicode\t" + str(sys.maxunicode + 1))

for c in map(chr, range(sys.maxunicode + 1)):
    unicode_category[unicodedata.category(c)].append(c)

for ucat, alist in unicode_category.items():
    print((ucat,alist[:10]))
    
