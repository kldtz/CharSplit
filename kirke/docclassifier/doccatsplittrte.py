#!/usr/bin/env python3

import argparse
from collections import defaultdict
import itertools
import json
import operator

def load_ebdata_tags_istest(filename):
    result = []
    is_test_set = False
    with open(filename, 'rt') as handle:
        parsed = json.load(handle)
        is_test_set = parsed.get('isTestSet', False)
        tags = parsed.get('tags')
    return tags, is_test_set


def split_doccat_trte(file_name: str):
    doc_tags_list = []
    tag_freq = defaultdict(int)
    tag_cooccur_list = []
    
    test_list = []
    train_list = []
    train_cat_count = defaultdict(int)
    test_cat_count = defaultdict(int)
    all_list = []
    doccat_count = defaultdict(int)
    valid_list = []
    valid_count = defaultdict(int)

    num_files = 0
    num_skipped = 0
    with open(file_name, 'rt') as fin:
        for line in fin:
            line = line.strip()

            txt_fn = line
            ebdata_fn = line.replace(".txt", ".ebdata")
            tags, is_test = load_ebdata_tags_istest(ebdata_fn)

            # if not tags, ignore it
            if tags:
                if is_test:
                    test_list.append("{}\t{}".format(txt_fn, ",".join(tags)))
                    for tag in tags:
                        test_cat_count[tag] += 1
                else:
                    train_list.append("{}\t{}".format(txt_fn, ",".join(tags)))
                    for tag in tags:
                        train_cat_count[tag] += 1
                all_list.append(txt_fn)
                for tag in tags:
                    doccat_count[tag] += 1
                
                # to decide if a tag is really co-occur with others
                doc_tags_list.append(tags)
                for tag in tags:
                    tag_freq[tag] += 1

                if len(tags) > 1:
                    for subset in itertools.combinations(tags, 2):
                        tag_cooccur_list.append(subset)
                
                # print("tags[{}]\t{}".format(tags, is_test))
            else:
                num_skipped += 1
            num_files += 1
    print("read {} files; skipped {} files because they have no tags".format(num_files, num_skipped))

    doccat_all_fn = file_name.replace(".filelist", ".valid.filelist")
    with open(doccat_all_fn, 'wt') as fout:
        for line in all_list:
            print(line, file=fout)
    print("wrote {}, size= {}".format(doccat_all_fn, len(valid_list)))                            

    train_fn = file_name.replace(".filelist", ".train.doccat.filelist")
    with open(train_fn, 'wt') as fout:
        for line in train_list:
            print(line, file=fout)
    print("wrote {}, size= {}".format(train_fn, len(train_list)))            

    test_fn = file_name.replace(".filelist", ".test.doccat.filelist")
    with open(test_fn, 'wt') as fout:
        for line in test_list:
            print(line, file=fout)
    print("wrote {}, size= {}".format(test_fn, len(test_list)))


    tmp_fn = file_name.replace(".filelist", ".doccat.filelist")
    with open(tmp_fn, 'wt') as fout:
        for line in train_list:
            print(line, file=fout)
        for line in test_list:
            print(line, file=fout)
    print("wrote {}, size= {}".format(tmp_fn, len(train_list) + len(test_list)))
    
    tmp_file_name = 'dict/doccat_train.count.tsv'
    with open(tmp_file_name, 'wt') as fout:
        train_view = [(v,k) for k,v in train_cat_count.items()]
        train_view.sort(reverse=True)
        for v, k in train_view:
            print("{}\t{}".format(k, v), file=fout)
        print("wrote {}, size= {}".format(tmp_file_name, len(train_view)))

    tmp_file_name = 'dict/doccat_test.count.tsv'    
    with open(tmp_file_name, 'wt') as fout:
        test_view = [(v,k) for k,v in test_cat_count.items()]
        test_view.sort(reverse=True)
        for v, k in test_view:
            print("{}\t{}".format(k, v), file=fout)
        print("wrote {}, size= {}".format(tmp_file_name, len(test_view)))

    tmp_file_name = 'dict/doccat.co-occur.tsv'            
    with open(tmp_file_name, 'wt') as fout:
        pair_count_map = defaultdict(int)
        for tag_cooccur in tag_cooccur_list:
            pair_count_map[tag_cooccur] += 1
                
        # print("{}\t{}".format(tag, tag_freq[tag]), file=fout)
        for v, k in sorted([(v, k) for k, v in pair_count_map.items()], reverse=True):
            tag = k[0]
            other_tag = k[1]
            tagx = tag_freq[tag]
            other_tagx = tag_freq[other_tag]
            
            # print("   {}\t{}".format(k, v), file=fout)
            print("{}\t{}\t{}\t{}\t{}".format(tag,
                                              tagx,
                                              other_tag,
                                              other_tagx,
                                              v), file=fout)
        print("wrote {}, size= {}".format(tmp_file_name, len(tag_cooccur_list)))

    tmp_file_name = 'dict/doccat.count.tsv'
    with open(tmp_file_name, 'wt') as fout:
        doccat_view = [(v,k) for k,v in doccat_count.items()]
        doccat_view.sort(reverse=True)
        for (v, k) in doccat_view:
            print("{}\t{}\t{}\t{}".format(k,
                                          v,
                                          train_cat_count[k],
                                          test_cat_count[k]), file=fout)
        print("wrote {}, size= {}".format(tmp_file_name, len(doccat_view)))

        
    # Decide if a tag is valid for training and testing
    # Only docs with minimum of 5 traing and 2 testing docs are valid.
    doccat_valid_fn = 'dict/doccat.valid.count.tsv'
    with open(doccat_valid_fn, 'wt') as fout:
        seq = 0  # need to start from 0, used in index of unigramclassifier.py, self.catnames
        for tag, tag_count in sorted(tag_freq.items(), key=operator.itemgetter(1), reverse=True):
            if (train_cat_count[tag] >= 5 and
                test_cat_count[tag] >= 2):
                print('{}\t{}\t{}\tvalid'.format(tag, tag_count, seq), file=fout)
                seq += 1
            else:
                # print('{}\t{}\t{}\tinvalid'.format(tag, tag_count, seq), file=fout)                
                print("  skipped category '%s' because number of train doc (%d < 5) or test doc (%d < 2)" %
                      (tag, train_cat_count[tag], test_cat_count[tag]))
        print("wrote {}, size= {}".format(doccat_valid_fn, seq))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SplitDocument for training and test for DocCat.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    file_name = args.file

    doccat_split_train_test(file_name)
    
                
            
    
            
