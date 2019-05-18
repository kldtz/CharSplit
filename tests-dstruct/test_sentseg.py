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
    txt_fname = 'dir-sent-check/txt/{}.txt'.format(doc_id)
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
    
class TestSentSeg(unittest.TestCase):
    
    def test_doc_8916(self):
        doc_id = '8916'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)


    def test_doc_8919(self):
        doc_id = '8919'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.93
        self.assertGreater(prec, 0.91)
        self.assertLess(prec, 0.95)

        # 0.97
        self.assertGreater(recall, 0.95)
        self.assertLess(recall, 0.99)

        # 0.95
        self.assertGreater(f1, 0.93)
        self.assertLess(f1, 0.97)


    def test_doc_8933(self):
        doc_id = '8933'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.68
        self.assertGreater(prec, 0.66)
        self.assertLess(prec, 0.7)

        # 0.92
        self.assertGreater(recall, 0.9)
        self.assertLess(recall, 0.94)

        # 0.78
        self.assertGreater(f1, 0.76)
        self.assertLess(f1, 0.8)


    def test_doc_8934(self):
        doc_id = '8934'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)


    def test_doc_8939(self):
        doc_id = '8939'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.41
        self.assertGreater(prec, 0.39)
        self.assertLess(prec, 0.43)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.57
        self.assertGreater(f1, 0.55)
        self.assertLess(f1, 0.59)


    def test_doc_8945(self):
        doc_id = '8945'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.62
        self.assertGreater(prec, 0.6)
        self.assertLess(prec, 0.64)

        # 0.86
        self.assertGreater(recall, 0.84)
        self.assertLess(recall, 0.88)

        # 0.72
        self.assertGreater(f1, 0.7)
        self.assertLess(f1, 0.74)


    def test_doc_8950(self):
        doc_id = '8950'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)


    def test_doc_8953(self):
        doc_id = '8953'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.95
        self.assertGreater(prec, 0.93)
        self.assertLess(prec, 0.97)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.95
        self.assertGreater(f1, 0.93)
        self.assertLess(f1, 0.97)


    def test_doc_8955(self):
        doc_id = '8955'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.91
        self.assertGreater(prec, 0.89)
        self.assertLess(prec, 0.93)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.93
        self.assertGreater(f1, 0.91)
        self.assertLess(f1, 0.95)


    def test_doc_8957(self):
        doc_id = '8957'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.74
        self.assertGreater(prec, 0.72)
        self.assertLess(prec, 0.76)

        # 0.85
        self.assertGreater(recall, 0.83)
        self.assertLess(recall, 0.87)

        # 0.79
        self.assertGreater(f1, 0.77)
        self.assertLess(f1, 0.81)


    def test_doc_8964(self):
        doc_id = '8964'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.98
        self.assertGreater(prec, 0.96)
        self.assertLess(prec, 1.0)

        # 0.98
        self.assertGreater(recall, 0.96)
        self.assertLess(recall, 1.0)

        # 0.98
        self.assertGreater(f1, 0.96)
        self.assertLess(f1, 1.0)


    def test_doc_8969(self):
        doc_id = '8969'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)


    def test_doc_8970(self):
        doc_id = '8970'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.76
        self.assertGreater(prec, 0.74)
        self.assertLess(prec, 0.78)

        # 0.75
        self.assertGreater(recall, 0.73)
        self.assertLess(recall, 0.77)

        # 0.76
        self.assertGreater(f1, 0.74)
        self.assertLess(f1, 0.78)


    def test_doc_8971(self):
        doc_id = '8971'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.99
        self.assertGreater(prec, 0.97)
        self.assertLess(prec, 1.01)

        # 0.98
        self.assertGreater(recall, 0.96)
        self.assertLess(recall, 1.0)

        # 0.98
        self.assertGreater(f1, 0.96)
        self.assertLess(f1, 1.0)


    def test_doc_8973(self):
        doc_id = '8973'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.9
        self.assertGreater(prec, 0.88)
        self.assertLess(prec, 0.92)

        # 0.94
        self.assertGreater(recall, 0.92)
        self.assertLess(recall, 0.96)

        # 0.92
        self.assertGreater(f1, 0.9)
        self.assertLess(f1, 0.94)


    def test_doc_8976(self):
        doc_id = '8976'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.99
        self.assertGreater(prec, 0.97)
        self.assertLess(prec, 1.01)

        # 0.98
        self.assertGreater(recall, 0.96)
        self.assertLess(recall, 1.0)

        # 0.98
        self.assertGreater(f1, 0.96)
        self.assertLess(f1, 1.0)


    def test_doc_8977(self):
        doc_id = '8977'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)


    def test_doc_8978(self):
        doc_id = '8978'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.95
        self.assertGreater(prec, 0.93)
        self.assertLess(prec, 0.97)

        # 0.94
        self.assertGreater(recall, 0.92)
        self.assertLess(recall, 0.96)

        # 0.95
        self.assertGreater(f1, 0.93)
        self.assertLess(f1, 0.97)


    def test_doc_8979(self):
        doc_id = '8979'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 0.92
        self.assertGreater(recall, 0.9)
        self.assertLess(recall, 0.94)

        # 0.96
        self.assertGreater(f1, 0.94)
        self.assertLess(f1, 0.98)


    def test_doc_8980(self):
        doc_id = '8980'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.93
        self.assertGreater(prec, 0.91)
        self.assertLess(prec, 0.95)

        # 0.92
        self.assertGreater(recall, 0.9)
        self.assertLess(recall, 0.94)

        # 0.92
        self.assertGreater(f1, 0.9)
        self.assertLess(f1, 0.94)


    def test_doc_8982(self):
        doc_id = '8982'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.88
        self.assertGreater(prec, 0.86)
        self.assertLess(prec, 0.9)

        # 0.9
        self.assertGreater(recall, 0.88)
        self.assertLess(recall, 0.92)

        # 0.89
        self.assertGreater(f1, 0.87)
        self.assertLess(f1, 0.91)


    def test_doc_8983(self):
        doc_id = '8983'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)


    def test_doc_8985(self):
        doc_id = '8985'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.95
        self.assertGreater(prec, 0.93)
        self.assertLess(prec, 0.97)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.95
        self.assertGreater(f1, 0.93)
        self.assertLess(f1, 0.97)


    def test_doc_8986(self):
        doc_id = '8986'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.98
        self.assertGreater(prec, 0.96)
        self.assertLess(prec, 1.0)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.97
        self.assertGreater(f1, 0.95)
        self.assertLess(f1, 0.99)


    def test_doc_8987(self):
        doc_id = '8987'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 0.97
        self.assertGreater(recall, 0.95)
        self.assertLess(recall, 0.99)

        # 0.99
        self.assertGreater(f1, 0.97)
        self.assertLess(f1, 1.01)


    def test_doc_8990(self):
        doc_id = '8990'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)


    def test_doc_8991(self):
        doc_id = '8991'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.77
        self.assertGreater(prec, 0.75)
        self.assertLess(prec, 0.79)

        # 0.92
        self.assertGreater(recall, 0.9)
        self.assertLess(recall, 0.94)

        # 0.84
        self.assertGreater(f1, 0.82)
        self.assertLess(f1, 0.86)


    def test_doc_8993(self):
        doc_id = '8993'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.64
        self.assertGreater(prec, 0.62)
        self.assertLess(prec, 0.66)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.77
        self.assertGreater(f1, 0.75)
        self.assertLess(f1, 0.79)


    def test_doc_8994(self):
        doc_id = '8994'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)


    def test_doc_8995(self):
        doc_id = '8995'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)


    def test_doc_8996(self):
        doc_id = '8996'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.99
        self.assertGreater(prec, 0.97)
        self.assertLess(prec, 1.01)

        # 0.99
        self.assertGreater(recall, 0.97)
        self.assertLess(recall, 1.01)

        # 0.99
        self.assertGreater(f1, 0.97)
        self.assertLess(f1, 1.01)


    def test_doc_9001(self):
        doc_id = '9001'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.94
        self.assertGreater(prec, 0.92)
        self.assertLess(prec, 0.96)

        # 0.97
        self.assertGreater(recall, 0.95)
        self.assertLess(recall, 0.99)

        # 0.96
        self.assertGreater(f1, 0.94)
        self.assertLess(f1, 0.98)


    def test_doc_9003(self):
        doc_id = '9003'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)


    def test_doc_9012(self):
        doc_id = '9012'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.8
        self.assertGreater(prec, 0.78)
        self.assertLess(prec, 0.82)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 0.89
        self.assertGreater(f1, 0.87)
        self.assertLess(f1, 0.91)


    def test_doc_9015(self):
        doc_id = '9015'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.92
        self.assertGreater(prec, 0.9)
        self.assertLess(prec, 0.94)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.94
        self.assertGreater(f1, 0.92)
        self.assertLess(f1, 0.96)


    def test_doc_9016(self):
        doc_id = '9016'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.93
        self.assertGreater(prec, 0.91)
        self.assertLess(prec, 0.95)

        # 0.94
        self.assertGreater(recall, 0.92)
        self.assertLess(recall, 0.96)

        # 0.94
        self.assertGreater(f1, 0.92)
        self.assertLess(f1, 0.96)


    def test_doc_9042(self):
        doc_id = '9042'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.81
        self.assertGreater(prec, 0.79)
        self.assertLess(prec, 0.83)

        # 0.88
        self.assertGreater(recall, 0.86)
        self.assertLess(recall, 0.9)

        # 0.85
        self.assertGreater(f1, 0.83)
        self.assertLess(f1, 0.87)


    def test_doc_9045(self):
        doc_id = '9045'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.93
        self.assertGreater(prec, 0.91)
        self.assertLess(prec, 0.95)

        # 0.96
        self.assertGreater(recall, 0.94)
        self.assertLess(recall, 0.98)

        # 0.94
        self.assertGreater(f1, 0.92)
        self.assertLess(f1, 0.96)


    def test_doc_9325(self):
        doc_id = '9325'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)


    def test_doc_9326(self):
        doc_id = '9326'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 1.0
        self.assertGreater(prec, 0.98)
        self.assertLess(prec, 1.02)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 1.0
        self.assertGreater(f1, 0.98)
        self.assertLess(f1, 1.02)
