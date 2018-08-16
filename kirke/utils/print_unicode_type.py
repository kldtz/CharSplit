#!/usr/bin/env python3

from collections import defaultdict
import sys
import unicodedata
# pylint: disable=unused-import
from typing import DefaultDict, List


print("max unicode = " + str(sys.maxunicode))

UNICODE_CATEGORY = defaultdict(list)  # type: DefaultDict[str, List[str]]

for c in map(chr, range(sys.maxunicode + 1)):
    UNICODE_CATEGORY[unicodedata.category(c)].append(c)

for i in range(sys.maxunicode + 1):
    c = chr(i)
    # UNICODE_CATEGORY[unicodedata.category(c)].append(c)
    print(str(i) + "\t" + unicodedata.category(c))
