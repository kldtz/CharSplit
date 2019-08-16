#!/usr/bin/env python3

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

    print()
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
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # based on Rachel's annotation.
        # the main source of the discrepancies are
        #    - from itemized list
        #    - sechead head not separated from rest
        #    - there are space-related bad segmentations
        # tp= 122, fn= 61, fp= 50
        # senteval resul: prec = 0.709, recall = 0.667, f1 = 0.687

        # 0.69
        self.assertGreater(f1, 0.64)
        self.assertLess(f1, 0.74)

        # 0.71
        self.assertGreater(prec, 0.66)
        self.assertLess(prec, 0.76)

        # 0.67
        self.assertGreater(recall, 0.62)
        self.assertLess(recall, 0.72)


    def test_doc_1006(self):
        doc_id = '1006'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # tp= 12, fn= 7, fp= 2
        # senteval resul: prec = 0.857, recall = 0.632, f1 = 0.727

        # - itemized list
        # - minor issues in this doc.
        # - bad score mainly due to few sentences.

        # 0.73
        self.assertGreater(f1, 0.68)
        self.assertLess(f1, 0.78)

        # 0.86
        self.assertGreater(prec, 0.81)
        self.assertLess(prec, 0.91)

        # 0.63
        self.assertGreater(recall, 0.58)
        self.assertLess(recall, 0.68)


    # Rachel skipped this doc for some unknown reason
    """
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
    """

    def test_doc_1008(self):
        doc_id = '1008'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # tp= 43, fn= 3, fp= 5
        # senteval resul: prec = 0.896, recall = 0.935, f1 = 0.915

        # mainly due to line spacing between lines in sentences
        # minor issue in this doc.

        # 0.92
        self.assertGreater(f1, 0.87)
        self.assertLess(f1, 0.97)

        # 0.90
        self.assertGreater(prec, 0.85)
        self.assertLess(prec, 0.95)

        # 0.94
        self.assertGreater(recall, 0.89)
        self.assertLess(recall, 0.99)


    def test_doc_1009(self):
        doc_id = '1009'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # tp= 7, fn= 4, fp= 2
        # senteval resul: prec = 0.778, recall = 0.636, f1 = 0.700
        # - minor issues
        # - due to not doing right segmentation inside address section
        # - bad score due to very limited sentences

        # 0.7
        self.assertGreater(f1, 0.65)
        self.assertLess(f1, 0.75)

        # 0.78
        self.assertGreater(prec, 0.73)
        self.assertLess(prec, 0.83)

        # 0.64
        self.assertGreater(recall, 0.59)
        self.assertLess(recall, 0.69)


    def test_doc_1010(self):
        doc_id = '1010'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # tp= 172, fn= 25, fp= 35
        # senteval resul: prec = 0.831, recall = 0.873, f1 = 0.851

        # 0.85
        self.assertGreater(f1, 0.80)
        self.assertLess(f1, 0.90)

        # 0.83
        self.assertGreater(prec, 0.78)
        self.assertLess(prec, 0.88)

        # 0.87
        self.assertGreater(recall, 0.82)
        self.assertLess(recall, 0.92)


    def test_doc_1011(self):
        doc_id = '1011'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # tp= 39, fn= 4, fp= 2
        # senteval resul: prec = 0.951, recall = 0.907, f1 = 0.929

        # 0.93
        self.assertGreater(f1, 0.88)
        self.assertLess(f1, 0.98)

        # 0.95
        self.assertGreater(prec, 0.90)
        self.assertLess(prec, 1.00)

        # 0.91
        self.assertGreater(recall, 0.86)
        self.assertLess(recall, 0.96)


    def test_doc_1012(self):
        doc_id = '1012'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # tp= 24, fn= 32, fp= 11
        # senteval resul: prec = 0.686, recall = 0.429, f1 = 0.527

        # - bad result due to limited number of sentences
        # - most source of error is itemized list or address table

        # 0.53
        self.assertGreater(f1, 0.48)
        self.assertLess(f1, 0.58)

        # 0.69
        self.assertGreater(prec, 0.64)
        self.assertLess(prec, 0.74)

        # 0.43
        self.assertGreater(recall, 0.38)
        self.assertLess(recall, 0.48)


    def test_doc_1013(self):
        doc_id = '1013'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # tp= 57, fn= 4, fp= 7
        # senteval resul: prec = 0.891, recall = 0.934, f1 = 0.912

        # 0.91
        self.assertGreater(f1, 0.86)
        self.assertLess(f1, 0.96)

        # 0.89
        self.assertGreater(prec, 0.84)
        self.assertLess(prec, 0.94)

        # 0.93
        self.assertGreater(recall, 0.88)
        self.assertLess(recall, 0.98)


    def test_doc_1014(self):
        doc_id = '1014'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # tp= 54, fn= 13, fp= 9
        # senteval resul: prec = 0.857, recall = 0.806, f1 = 0.831

        # 0.83
        self.assertGreater(f1, 0.78)
        self.assertLess(f1, 0.88)

        # 0.86
        self.assertGreater(prec, 0.81)
        self.assertLess(prec, 0.91)

        # 0.81
        self.assertGreater(recall, 0.76)
        self.assertLess(recall, 0.86)


    def test_doc_1015(self):
        doc_id = '1015'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # tp= 47, fn= 16, fp= 9
        # senteval resul: prec = 0.839, recall = 0.746, f1 = 0.790

        # 0.79
        self.assertGreater(f1, 0.74)
        self.assertLess(f1, 0.84)

        # 0.84
        self.assertGreater(prec, 0.79)
        self.assertLess(prec, 0.89)

        # 0.75
        self.assertGreater(recall, 0.70)
        self.assertLess(recall, 0.80)


    def test_doc_1016(self):
        doc_id = '1016'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # tp= 32, fn= 11, fp= 8
        # senteval resul: prec = 0.800, recall = 0.744, f1 = 0.771

        # 0.77
        self.assertGreater(f1, 0.72)
        self.assertLess(f1, 0.82)

        # 0.80
        self.assertGreater(prec, 0.75)
        self.assertLess(prec, 0.85)

        # 0.74
        self.assertGreater(recall, 0.69)
        self.assertLess(recall, 0.79)


    def test_doc_1017(self):
        doc_id = '1017'
        gold_sent_list = read_sent_file('data/japanese/gold/{}.sent.txt.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        # pylint: disable=invalid-name
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # tp= 176, fn= 66, fp= 53
        # senteval resul: prec = 0.769, recall = 0.727, f1 = 0.747

        # 0.75
        self.assertGreater(f1, 0.70)
        self.assertLess(f1, 0.80)

        # 0.77
        self.assertGreater(prec, 0.72)
        self.assertLess(prec, 0.82)

        # 0.73
        self.assertGreater(recall, 0.68)
        self.assertLess(recall, 0.78)
