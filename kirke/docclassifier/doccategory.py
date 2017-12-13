#!/usr/bin/env python

from collections import defaultdict
import json
import operator

IS_DEBUG = False
IGNORE_TAG = 'IGNORE'

def load_normtags(normtag_fname, ignoretag_fname):
    aname_to_normtag_map = {}
    with open(normtag_fname, 'rt') as fin:
        for line in fin:
            aname, norm_name = line.strip().split('\t')
            aname = aname.strip()  # make sure there is no prefix, suffix space
            norm_name = norm_name.strip()

            if aname != norm_name:
                aname_to_normtag_map[aname] = norm_name
            aname_to_normtag_map[norm_name] = norm_name

    with open(ignoretag_fname, 'rt') as fin:
        for line in fin:
            aname = line.strip()
            aname_to_normtag_map[aname] = IGNORE_TAG

    #for aname, normtag in sorted(aname_to_normtag_map.items()):
    #    print("normtags\t[{}]\t[{}]".format(aname, normtag))
    return aname_to_normtag_map

NAME2NORMTAG_MAP = load_normtags('dict/str2normcat66.tsv', 'dict/catstr2ignore.txt')

def name_to_coretag(aname):
    aname = aname.strip()
    coretag = NAME2NORMTAG_MAP[aname]
    if coretag != IGNORE_TAG:
        return coretag
    return None


def _create_wanted_tags_step1(txt_fn_list_fn):
    tag_count_map = defaultdict(int)
    unwanted_tags = defaultdict(int)

    with open(txt_fn_list_fn, 'rt') as fin:
        for line in fin:
            txt_fn = line.strip()
            ebdata_fn = txt_fn.replace('.txt', '.ebdata')

            with open(ebdata_fn, 'rt') as ebdata_fin:
                tags = json.loads(ebdata_fin.read())['tags']
                # make sure there is no prefix, suffix space
                tags = [tag.strip() for tag in tags]

                for tag in set(tags):
                    coretag = NAME2NORMTAG_MAP[tag]
                    if coretag != IGNORE_TAG:
                        tag_count_map[coretag] += 1
                    else:
                        unwanted_tags[tag] += 1

    wanted_tags = []
    for norm_tag, freq in sorted(tag_count_map.items(), key=operator.itemgetter(1), reverse=True):
        # if tag not found, want throw an error

        if freq >= 20:
            wanted_tags.append(norm_tag)
            print('wanted_tag\t{}\t{}'.format(norm_tag, freq))
        else:
            unwanted_tags[norm_tag] += freq
            print('skip_tag\t{}\t{}'.format(norm_tag, freq))

    #with open('dict/doccat.ignore.tsv', 'wt') as igfout:
    #    for tag, freq in sorted(unwanted_tags.items(), key=operator.itemgetter(1), reverse=True):
    #        print('{}\t{}'.format(tag, freq), file=igfout)

    return wanted_tags


def init_doccats_step1():
    core_tags = _create_wanted_tags_step1('sample.filelist')

    coretag_catid_map = {}
    catid_coretag_map = {}
    with open('dict/coretag_catid_step1.tsv', 'wt') as fout:
        print()
        for catid, coretag in enumerate(core_tags):
            print("coretag\t{}\t{}".format(catid, coretag))
            print("{}\t{}".format(coretag, catid), file=fout)
            coretag_catid_map[coretag] = catid
            catid_coretag_map[catid] = coretag

    return core_tags, coretag_catid_map, catid_coretag_map


def load_doccats_prod():
    coretag_catid_map = {}
    catid_coretag_map = {}
    coretag_list = []
    with open('dict/coretag_wanted_catid.tsv', 'rt') as fin:
        for line in fin:
            coretag, catid = line.strip().split('\t')
            catid = int(catid)
            if IS_DEBUG:
                print("coretag\t{}\t[{}]".format(catid, coretag))
            coretag_list.append(coretag)
            coretag_catid_map[coretag] = catid
            catid_coretag_map[catid] = coretag

    return coretag_list, coretag_catid_map, catid_coretag_map



#def coretag_to_catid(coretag, coretag_catid_map):
#    return coretag_catid_map[coretag]
#
#def catid_to_coretag(catid, catid_coretag_map):
#    return catid_coretag_map[catid]

def tags_to_coretags(tags, catname_catid_map=None):
    result = []
    for tag in tags:
        coretag = name_to_coretag(tag)
        if coretag:
            # no restriction on wanted_coretag_set, then add all
            if not catname_catid_map:
                result.append(coretag)
            else:
                if catname_catid_map.get(coretag) != None:
                    result.append(coretag)
                else:
                    if IS_DEBUG:
                        print('unwanted tag\t[{}]'.format(coretag))
        else:
            if IS_DEBUG:
                print('unwanted tag2\t[{}]'.format(tag))
    return result

def coretags_to_catids(coretags, coretag_catid_map):
    return [coretag_catid_map[coretag] for coretag in coretags]

def tags_to_catids(tags, coretag_catid_map, wanted_coretag_set=None):
    coretags = tags_to_coretags(tags, coretag_catid_map)
    return coretags_to_catids(coretags, coretag_catid_map)

def coretags_to_catids(coretags, coretag_catid_map):
    return [coretag_catid_map[coretag] for coretag in coretags]
