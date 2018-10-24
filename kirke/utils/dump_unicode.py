#!/usr/bin/env python3

from collections import defaultdict
import sys
import unicodedata
# pylint: disable=unused-import
from typing import DefaultDict, List

UNICODE_CATEGORY = defaultdict(list)  # type: DefaultDict[str, List[str]]

for c in map(chr, range(sys.maxunicode + 1)):
    UNICODE_CATEGORY[unicodedata.category(c)].append(c)

for ucat, alist in UNICODE_CATEGORY.items():
    print((ucat, alist[:10]))
