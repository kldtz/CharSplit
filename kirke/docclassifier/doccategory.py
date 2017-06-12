#!/usr/bin/env python

from collections import defaultdict
import json
import operator

def load_normtags(normtag_fname):
    aname_to_normtag_map = {}
    with open(normtag_fname, 'rt') as fin:
        for line in fin:
            aname, norm_name = line.strip().split('\t')
            aname = aname.strip()  # make sure there is no prefix, suffix space
            norm_name = norm_name.strip()

            if aname != norm_name:
                aname_to_normtag_map[aname] = norm_name
            aname_to_normtag_map[norm_name] = norm_name
    # for aname, normtag in aname_to_normtag_map.items():
    #    print("normtags\t[{}]\t[{}]".format(aname, normtag))
    return aname_to_normtag_map

NAME2NORMTAG_MAP = load_normtags('dict/str2normcat66.tsv')


def create_wanted_tags(txt_fn_list_fn):
    tag_count_map = defaultdict(int)
    with open(txt_fn_list_fn, 'rt') as fin:
        for line in fin:
            txt_fn = line.strip()
            ebdata_fn = txt_fn.replace('.txt', '.ebdata')

            with open(ebdata_fn, 'rt') as ebdata_fin:
                tags = json.loads(ebdata_fin.read())['tags']
                # make sure there is no prefix, suffix space
                tags = [tag.strip() for tag in tags]

                for tag in set(tags):
                    tag_count_map[tag] += 1

    wanted_tags = []
    not_found_tags = []
    for tag, freq in sorted(tag_count_map.items(), key=operator.itemgetter(1), reverse=True):
        norm_tag = NAME2NORMTAG_MAP.get(tag)
        if norm_tag:
            if freq >= 20:
                wanted_tags.append(norm_tag)
            print('tag\t{}\t{}\t{}'.format(norm_tag, freq, tag))
        else:
            not_found_tags.append((tag, freq))

    print()
    for tag, freq in not_found_tags:
        print("skip_tag\t{}\t{}".format(tag, freq))

    return wanted_tags


def init_doccats():
    core_tags = create_wanted_tags('sample.filelist')

    with open('dict/coretag_catid.tsv', 'wt') as fout:
        print()
        for catid, tag in enumerate(core_tags):
            print("coretag\t{}\t{}".format(catid, tag))
            print("{}\t{}".format(tag, catid), file=fout)

    return core_tags

# CORE_TAGS = init_doccats()

coretag_catid_map = {}
catid_coretag_map = {}
with open('dict/coretag_catid.tsv', 'rt') as fin:
    for line in fin:
        coretag, catid = line.strip().split('\t')
        coretag_catid_map[coretag] = catid
        catid_coretag_map[catid] = coretag

CORE_TAGS = list(coretag_catid_map.keys())
CORE_TAG_SET = set(CORE_TAGS)

def name_to_coretag(aname):
    aname = aname.strip()
    coretag = NAME2NORMTAG_MAP.get(aname)
    if coretag in CORE_TAG_SET:
        return coretag
    return None

# pylint: disable=invalid-name
coretag_catid_map = {}
catid_coretag_map = {}
# print()
for catid, tag in enumerate(CORE_TAGS):
    # print("coretag\t{}".format(tag))
    coretag_catid_map[tag] = catid
    catid_coretag_map[catid] = tag

def coretag_to_catid(coretag):
    return coretag_catid_map[coretag]

def catid_to_coretag(catid):
    return catid_coretag_map[catid]

def tags_to_coretags(tags):
    result = []
    for tag in tags:
        coretag = name_to_coretag(tag)
        if coretag:
            result.append(coretag)
    return result

def coretags_to_catids(coretags):
    return [coretag_catid_map[coretag] for coretag in coretags]

def tags_to_catids(tags):
    coretags = tags_to_coretags(tags)
    return coretags_to_catids(coretags)

# pylint: disable=invalid-name
doc_cat_names = CORE_TAGS
