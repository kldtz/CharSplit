#!/usr/bin/env python3

import json
import os
import unittest
from typing import List, Tuple

import langdetect

from kirke.utils import ebantdoc4, osutils, strutils, evalutils
# from kirke.utils.ebantdoc4 import html_to_ebantdoc, pdf_to_ebantdoc


WORK_DIR = 'dir-work'

osutils.mkpath(WORK_DIR)

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


def calc_prec_recall_f1(gold_sent_list: List[str],
                        pred_sent_list: List[str]) \
                        -> Tuple[float, float, float]:
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
    print('senteval resul:\tprec = %.3f, recall = %.3f, f1 = %.3f'
          % (prec, recall, f1))
    return prec, recall, f1


def gen_sent_list(doc_id: str) -> List[str]:
    txt_fname = 'data/japanese/txt/{}.txt'.format(doc_id)
    base_fname = os.path.basename(txt_fname)

    atext = strutils.loads(txt_fname)
    doc_lang = langdetect.detect(atext)
    ebdoc = ebantdoc4.text_to_ebantdoc(txt_fname, WORK_DIR, doc_lang=doc_lang)

    print('loaded %s' % ebdoc.file_id)
    work_fn = os.path.join(WORK_DIR, base_fname)

    sent_out_fname = work_fn.replace('.txt', '.sent.txt')
    ebantdoc4.save_sent_se_text(ebdoc,
                                sent_out_fname)
    print('wrote %s' % sent_out_fname)

    pred_sent_list = read_sent_file(sent_out_fname)
    # print('pred_sent_list: {}'.format(len(pred_sent_list)))
    return pred_sent_list


    # dir-sent-check/gold/8916.sent.tsv.gold dir-work/8916.sent.txt

class TestSentSegJa(unittest.TestCase):

    def test_doc_1005(self):
        doc_id = '1005'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1006(self):
        doc_id = '1006'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1007(self):
        doc_id = '1007'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1008(self):
        doc_id = '1008'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1009(self):
        doc_id = '1009'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1010(self):
        doc_id = '1010'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1011(self):
        doc_id = '1011'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1012(self):
        doc_id = '1012'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1013(self):
        doc_id = '1013'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1014(self):
        doc_id = '1014'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1015(self):
        doc_id = '1015'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1016(self):
        doc_id = '1016'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)


    def test_doc_1017(self):
        doc_id = '1017'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

