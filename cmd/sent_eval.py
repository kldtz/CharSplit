#!/usr/bin/env python

import argparse
from collections import defaultdict
import os
from typing import List

from kirke.utils import ebantdoc4, osutils, evalutils


def read_sent_file(file_name: str) -> List[str]:
    out_list = []  # type: List[str]
    with open(file_name, 'rt') as fin:
        for line in fin:
            # skip empty line and lines that are too short
            if not line.strip():
                continue
            # this is to avoid a lot of noise
            if len(line) < 10:
                continue
            out_list.append(line)
    return out_list


def main():
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt',
    # help='input directory for .txt files')
    parser.add_argument('file1', help='gold sent file')
    parser.add_argument('file2', help='pred sent file')

    args = parser.parse_args()

    gold_sent_list = read_sent_file(args.file1)
    pred_sent_list = read_sent_file(args.file2)

    print('gold_sent_list: {}'.format(len(gold_sent_list)))
    print('pred_sent_list: {}'.format(len(pred_sent_list)))    

    gold_sent_set = set(gold_sent_list)
    pred_sent_set = set(pred_sent_list)
    # pylint: disable=invalid-name
    tp, fn, fp = 0, 0, 0
    tp_list, fn_list, fp_list = [], [], []  # type: Tuple[List[str], List[str], List[str]]
    for gold_sent in gold_sent_list:
        if gold_sent in pred_sent_set:
            tp += 1
            tp_list.append(gold_sent)
            # print("TP: [{}]".format(gold_sent.replace('\n', ' ')))
        else:
            fn += 1
            fn_list.append(gold_sent)
            print("FN: [{}]".format(gold_sent.replace('\n', ' ')))            

    for pred_sent in pred_sent_list:
        if pred_sent not in gold_sent_set:
            fp += 1
            fp_list.append(pred_sent)
            print("FP: [{}]".format(pred_sent.replace('\n', ' ')))                        

    print();
    print('tp= {}, fn= {}, fp= {}'.format(tp, fn, fp))
    prec, recall, f1 = evalutils.calc_precision_recall_f1(0, fp, fn, tp)
    print('senteval resul:\t%s\tprec = %.3f, recall = %.3f, f1 = %.3f'
          % (args.file2, prec, recall, f1))


if __name__ == '__main__':
    main()






