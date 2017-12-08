#!/usr/bin/env python

from collections import defaultdict
import json
import re

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from kirke.utils import txtreader

# based on eval set, 250 seems to work best
# TEXT_SIZE = 1000
TEXT_SIZE = 250


def load_doccat_maps(file_name: str):
    catname_list = []
    catname_catid_map = {}
    # catid_catname_map = {}
    # valid_tags = set([])
    with open(file_name, 'rt') as fin:
        for line in fin:
            tag, freq, catid, is_valid = line.strip().split('\t')
            catid = int(catid)
            # print("tag [{}], freq[{}], catid=[{}]".format(tag, freq, catid))
            catname_list.append(tag)
            catname_catid_map[tag] = catid
            # catid_catname_map[catid] = tag
            # if is_valid == 'valid':
            #     valid_tags.add(tag)
    # double check on the catid and order in catname_list
    #for i, catname in enumerate(catname_list):
    #    tmp_catid = catname_catid_map[catname]
    #    if i != tmp_catid:
    #        print("WRONG tag [{}], catid {}, {}".format(catname, i, tmp_catid))
    # print("YYYYYY")
    return catname_list, catname_catid_map


# Only load files with valid tags, otherwise we will be training on them
def load_data(txt_fn_list_fn, catname_catid_map, valid_tags):

    doc_text_list = []
    catids_list = []

    with open(txt_fn_list_fn, 'rt') as fin:
        for line in fin:
            txt_fn = line.strip()
            ebdata_fn = txt_fn.replace('.txt', '.ebdata')

            with open(ebdata_fn, 'rt') as ebdata_fin:
                parsed = json.load(ebdata_fin)
                tags = parsed.get('tags')

                tag_set = set(tags)
                overlap = valid_tags.intersection(tag_set)
                if not overlap:
                    print("skipping file [{}] because of invalid tags: {}".format(txt_fn, tags))
                    continue
                # we only output valid tagid here because we don't want to train on invalid ones
                catids = [catname_catid_map[tag] for tag in tags if tag in valid_tags]
                catids_list.append(catids)

                doc_text = txtreader.loads(txt_fn)
                doc_text_list.append(doc_text_to_docfeats(doc_text))

    # print('len(catid_list) = {}'.format(len(catids_list)))

    return doc_text_list, catids_list


_STEMMER = SnowballStemmer("english")
_EN_STOPWORD_SET = stopwords.words('english')

def doc_text_to_docfeats(doc_text, wanted_text_len=TEXT_SIZE):
    lc_doc_text = doc_text.lower()
    # Based on the training and testing set, wanted_text_len = 250 is
    # the best (0.88), but our corpus might not reflect real life.
    # Currently, setting it to 1000 instead (0.85).
    # tried 100, 250, 500, 1000, 2000, 4000
    tokens = re.findall(r'\b[A-Za-z]+\b', lc_doc_text[:wanted_text_len])

    return ' '.join([_STEMMER.stem(tok) for tok in tokens if tok not in _EN_STOPWORD_SET])


SCORE_PAT = re.compile(r'avg / total\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)')
# return a tuple of precision, recall, f1
def report_to_eval_scores(lines):
    for line in lines.split('\n'):
        mat = SCORE_PAT.search(line)

        if mat:
            return float(mat.group(1)), float(mat.group(2)), float(mat.group(3))
    return -1, -1, -1


def avg_list(alist):
    sum = 0.0
    for x in alist:
        sum += x
    return sum / len(alist)

# TAG_NUMS_PAT = re.compile(r'^\s+(.+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(\d+)(.*)')
# TAG_NUMS_PAT = re.compile(r'^\s+(.+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+(.+)')
def print_combined_reports(report_list, valid_tags, threshold=None):
    print("combined report for cross validation")
    found_tags = []
    tag_result_map = defaultdict(list)
    for report in report_list:
        for line in report.split('\n'):
            # print("line: [{}]".format(line))
            cols = re.split(r"\s\s+", line)

            # mat = TAG_NUMS_PAT.match(line)
            if len(cols) > 4 and cols[0] in valid_tags:  # confidentiality is really long
                tag = cols[0]
                others = (float(cols[1]),
                          float(cols[2]),
                          float(cols[3]),
                          int(cols[4]))
                if tag not in tag_result_map:
                    found_tags.append(tag)
                tag_result_map[tag].append(others)
                # print("col2\t{}\t{}\t{}\t{}".format(cols[0], cols[1], cols[2], cols[3]))
            elif len(cols) > 5 and cols[1] in valid_tags:
                tag = cols[1]
                others = (float(cols[2]),
                          float(cols[3]),
                          float(cols[4]),
                          int(cols[5]))
                if tag not in tag_result_map:
                    found_tags.append(tag)
                tag_result_map[tag].append(others)

                # print("col2\t{}\t{}\t{}\t{}".format(cols[1], cols[2], cols[3], cols[4]))

    print("{:>36s}{:>11s}{:>10s}{:>10s}{:>10s}".format('', 'precison', 'recall',
                                                       'f1-score', 'support'))

    avg_prec_list, avg_recall_list, avg_f1_list, sum_support_list = [], [], [], []
    for tag in found_tags:
        prec_list = []
        recall_list = []
        f1_list = []
        support_list = []
        for arun in tag_result_map[tag]:
            prec, recall, f1, support = arun

            if f1 != 0.0:
                prec_list.append(prec)
                recall_list.append(recall)
                f1_list.append(f1)
                support_list.append(support)

        tmp_avg_f1 = 0.0
        if f1_list:
            tmp_avg_f1 = avg_list(f1_list)
        if (len(f1_list) == 3 and ((threshold is None) or
                                   (threshold is not None and tmp_avg_f1 >= threshold))):
            avg_prec = avg_list(prec_list)
            avg_recall = avg_list(recall_list)
            avg_f1 = avg_list(f1_list)
            sum_support = sum(support_list)

            avg_prec_list.append(avg_prec)
            avg_recall_list.append(avg_recall)
            avg_f1_list.append(avg_f1)
            sum_support_list.append(sum_support)

            print("{:>36s}{:11.2f}{:10.2f}{:10.2f}{:10d}".format(tag,
                                                                avg_list(prec_list),
                                                                avg_list(recall_list),
                                                                avg_list(f1_list),
                                                                sum(support_list)))
            st_list = [str(prec_list),
                       str(recall_list),
                       str(f1_list),
                       str(support_list)]
            # print("\n{}\t{}".format(tag, "\t".join(st_list)))
        else:
            st_list = [str(arun) for arun in tag_result_map[tag]]
            # print("skip {}\t{}".format(tag, "\t".join(st_list)))
    print()
    print("{:>36s}{:11.2f}{:10.2f}{:10.2f}{:10d}".format('avg / total',
                                                         avg_list(avg_prec_list),
                                                         avg_list(avg_recall_list),
                                                         avg_list(avg_f1_list),
                                                         sum(sum_support_list)))
