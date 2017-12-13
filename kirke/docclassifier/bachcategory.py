
from collections import OrderedDict
import re

# pylint: disable=pointless-string-statement
"""
1,Lease Agreement,648
2,None,514
3,Loan Agreement,475
4,Purchase Agreement,341
5,Services Agreement,329
6,Employment Agreement,291
7,Distribution Agreement,189
8,License Agreement,183
9,Promissory Note,138
10,Conversion Agreement,133
11,Marketing Agreemnt,117
12,Merger Agreement,99
13,Manufacturing Agreement,66
14,Sales Agreement,57
15,Termination Agreement,8
16,Franchise Agreement,6
"""

FNPAT_CATNAME_MAP = OrderedDict([('Lease', 'Lease Agreement'),
                                 ('None', 'None'),
                                 ('Loan', 'Loan Agreement'),
                                 ('Revolving', 'Loan Agreement'),
                                 ('Purchase', 'Purchase Agreement'),
                                 ('Services', 'Services Agreement'),
                                 ('Employment', 'Employment Agreement'),
                                 ('Distribution', 'Distribution Agreement'),
                                 ('License', 'License Agreement'),
                                 ('Promissory', 'Promissory Note'),
                                 ('Conversion', 'Conversion Agreement'),
                                 ('Marketing', 'Marketing Agreemnt'),
                                 ('Merger', 'Merger Agreement'),
                                 ('Manufacturing', 'Manufacturing Agreement'),
                                 ('Sales', 'Sales Agreement'),
                                 ('Termination', 'Termination Agreement'),
                                 ('Franchise', 'Franchise Agreement')])

doc_cat_names = []

catname_catid_map = {}
catid_catname_map = {}

# there is a repeated entry "Loan Agreement"
catid = 0
for _, catname in FNPAT_CATNAME_MAP.items():
    if not catname_catid_map.get(catname):
        catname_catid_map[catname] = catid
        catid_catname_map[catid] = catname
        catid += 1
        doc_cat_names.append(catname)


def tags_to_catnames(tags):
    labels = []
    label_str = ' '.join(tags).lower()
    for word in FNPAT_CATNAME_MAP:
        pat = r"\b" + word.lower() + r"\b"
        if re.search(pat, label_str):
            labels.append(FNPAT_CATNAME_MAP[word])
    return labels


def tags_to_catids(tags):
    catnames = tags_to_catnames(tags)
    return catnames_to_catids(catnames)


def catnames_to_catids(catnames):
    return sorted([catname_catid_map[catname] for catname in catnames])


def catname_to_catid(catname):
    return catname_catid_map.get(catname)


def catid_to_catname(catid):
    return catid_catname_map.get(catid)
