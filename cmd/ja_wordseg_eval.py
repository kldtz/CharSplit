#!/usr/bin/env python

import argparse
import re
from typing import List

from kirke.utils import evalutils

# pylint: disable=line-too-long
# usage: python -m cmd.ja_wordseg_eval rachelv2/1005/1005.sent.tokens.txt ja_human_eval/1005/1005.sent.tokens.txt


def split_by_vline(line: str) -> List[str]:
    out_list = []  # type: List[str]

    for token in line.split('|'):
        token = token.strip().replace(' ', '')

        if token:
            out_list.append(token)
    return out_list


def read_wordseg_file(file_name: str) -> List[List[str]]:
    out_list = []  # type: List[List[str]]
    with open(file_name, 'rt') as fin:
        for line in fin:
            line = line.strip()

            # skip if empty
            if not line:
                continue

            # skip if comment
            if re.search(r'^[A-Z]{1,4}:?', line):
                continue

            words = split_by_vline(line)
            out_list.append(words)
    return out_list


def main():
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt',
    # help='input directory for .txt files')
    parser.add_argument('file1', help='gold wordseg file')
    parser.add_argument('file2', help='pred wordseg file')

    args = parser.parse_args()

    gold_sent_list = read_wordseg_file(args.file1)
    pred_sent_list = read_wordseg_file(args.file2)

    print('len(gold_sent_list) = {}'.format(len(gold_sent_list)))
    # print('pred_sent_list: {}'.format(len(pred_sent_list)))

    # for sent_i, pred_sent in enumerate(pred_sent_list):
    #     print('sent #{}\t{}'.format(sent_i, pred_sent[:5]))

    if len(gold_sent_list) != len(pred_sent_list):
        print('len(gold_sent_list) {} != len(pred_sent_list) {}'.format(
            len(gold_sent_list),
            len(pred_sent_list)))
        return

    for gold_words, pred_words in zip(gold_sent_list, pred_sent_list):
        gold_word_set = set(gold_words)
        pred_word_set = set(pred_words)

        missed = gold_word_set.difference(pred_word_set)
        extra = pred_word_set.difference(gold_word_set)

        if missed or extra:
            print('\nmissed: {}'.format(list(missed)))
            print('extra: {}'.format(list(extra)))

    """
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
    """


if __name__ == '__main__':
    main()
