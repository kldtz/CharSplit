#!/usr/bin/env python

from collections import OrderedDict, defaultdict
import re
import json
import operator

def load_normtags(normtag_fname):
    aname_to_normtag_map = {}
    with open(normtag_fname, 'rt') as fin:
        for line in fin:
            aname, norm_name = line.strip().split('\t')
            aname = aname.strip()
            norm_name = norm_name.strip()            

            if aname != norm_name:
                aname_to_normtag_map[aname] = norm_name
            aname_to_normtag_map[norm_name] = norm_name
    return aname_to_normtag_map

NAME2NORMTAG_MAP = load_normtags('dict/str2normcat66.tsv')


def create_wanted_tags(txt_fn_list_fn):
    doc_text_list, catids_list = [], []

    tag_count_map = defaultdict(int)
    with open(txt_fn_list_fn, 'rt') as fin:
        for line in fin:
            txt_fn = line.strip()
            ebdata_fn = txt_fn.replace('.txt', '.ebdata')

            with open(ebdata_fn, 'rt') as ebdata_fin:
                tags = json.loads(ebdata_fin.read())['tags']

                for tag in set(tags):
                    tag = tag.strip()
                    tag_count_map[tag] += 1

    wanted_tags = []
    not_found_tags = []    
    for i, (tag, freq) in enumerate(sorted(tag_count_map.items(), key=operator.itemgetter(1), reverse=True)):
        norm_tag = NAME2NORMTAG_MAP.get(tag)
        if norm_tag:
            if freq >= 20:
                wanted_tags.append(norm_tag)
            else:
                # print('xxx less 20 tag\t{}\t{}\t{}\t{}'.format(i, norm_tag, freq, tag))
                print('skip_tag2\t{}\t{}\t{}'.format(norm_tag, freq, tag))                
                
            print('tag\t{}\t{}\t{}\t{}'.format(i, norm_tag, freq, tag))
        else:
            not_found_tags.append((tag, freq))

    print()
    for tag, freq in not_found_tags:
        print("skip_tag\t{}\t{}".format(tag, freq))

    return wanted_tags

CORE_TAGS = create_wanted_tags('sample.filelist')

def name_to_coretag(aname):
    return NAME2NORMTAG_MAP.get(aname)

coretag_catid_map = {}
catid_coretag_map = {}
print()
for catid, tag in enumerate(CORE_TAGS):
    print("coretag\t{}\t{}".format(tag, catid))
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


def create_wanted_tags_file(txt_fn_list_fn):
    with open('doc_category.map.tsv', 'wt') as fout:
        with open(txt_fn_list_fn, 'rt') as fin:
            for line in fin:
                txt_fn = line.strip()
                ebdata_fn = txt_fn.replace('.txt', '.ebdata')

                with open(ebdata_fn, 'rt') as ebdata_fin:
                    tags = json.loads(ebdata_fin.read())['tags']

                coretags = tags_to_coretags(tags)
                print('{}\t{}'.format(txt_fn, ','.join(coretags)), file=fout)

create_wanted_tags_file('sample.filelist')


