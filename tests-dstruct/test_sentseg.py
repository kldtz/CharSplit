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

        # we are not removing page number correctly

        # 0.94
        self.assertGreater(prec, 0.92)
        self.assertLess(prec, 0.96)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 0.97
        self.assertGreater(f1, 0.95)
        self.assertLess(f1, 0.99)


    def test_doc_8919(self):
        doc_id = '8919'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # there is also an incorrect 'party line' sentence break
        # otherwise, we are doing ok.

        # 0.92
        self.assertGreater(prec, 0.90)
        self.assertLess(prec, 0.94)

        # 0.97
        self.assertGreater(recall, 0.95)
        self.assertLess(recall, 0.99)

        # 0.94
        self.assertGreater(f1, 0.92)
        self.assertLess(f1, 0.96)


    def test_doc_8933(self):
        doc_id = '8933'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # footer, header, paragraph across pages are fixed.
        # They are put together at the end of a page to preserve
        # all linex in the text.
        # Same applies to table of content lines too.  we don't
        # delete them anymore.
        # Because we do NOT delete any page number, footer, header.
        # the score is penalized.  Otherwise, we do ok in tests.

        # This particular document has header split into two, with
        # 'Contents' as first line in a page, 'Clause Page' as the last
        # line in the page.  Including such heading into nlp.txt causes
        # the whole page to be included if heading are output as a block.
        # as a result, had to insert an empty line between them to
        # avoid such nasty "block".

        # 0.65
        self.assertGreater(prec, 0.63)
        self.assertLess(prec, 0.67)

        # 0.92
        self.assertGreater(recall, 0.90)
        self.assertLess(recall, 0.94)

        # 0.76
        self.assertGreater(f1, 0.74)
        self.assertLess(f1, 0.78)


    # There is grease in the doc, result not reliable.
    # ignore.
    # def test_doc_8934(self):
    #    doc_id = '8934'
    #    gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
    #    pred_sent_list = gen_sent_list(doc_id)
    #    prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # the new code seems to be a better job of sentence seg
        # will probably switch the gold
        # 1.0
    #    self.assertGreater(prec, 0.98)
    #    self.assertLess(prec, 1.02)

        # 1.0
    #    self.assertGreater(recall, 0.98)
    #    self.assertLess(recall, 1.02)

        # 1.0
    #    self.assertGreater(f1, 0.98)
    #    self.assertLess(f1, 1.02)


    def test_doc_8939(self):
        doc_id = '8939'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # we are doing better than before
        # 0.62
        self.assertGreater(prec, 0.60)
        self.assertLess(prec, 0.64)

        # 0.89
        self.assertGreater(recall, 0.87)
        self.assertLess(recall, 0.91)

        # 0.73
        self.assertGreater(f1, 0.71)
        self.assertLess(f1, 0.75)


    # senteval resul: dir-work/8945.sent.txt  prec = 0.546, recall = 0.929, f1 = 0.688
    def test_doc_8945(self):
        doc_id = '8945'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # comparable to before

        # 0.55
        self.assertGreater(prec, 0.53)
        self.assertLess(prec, 0.57)

        # 0.93
        self.assertGreater(recall, 0.91)
        self.assertLess(recall, 0.95)

        # 0.69
        self.assertGreater(f1, 0.67)
        self.assertLess(f1, 0.71)


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

        # there were some double-byte character issues, but resolved it by
        # always use the original text intead of the normalized single-byte
        # characters.

        # doing ok
        # 0.95 in double-bytes
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

        # we are doing worse than before.  Many bad sentseg than before
        # senteval resul: dir-work/8955.sent.txt  prec = 0.875, recall = 0.930, f1 = 0.902

        # 0.88
        self.assertGreater(prec, 0.86)
        self.assertLess(prec, 0.90)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.90
        self.assertGreater(f1, 0.88)
        self.assertLess(f1, 0.91)


    def test_doc_8957(self):
        doc_id = '8957'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # slightly worse than before.
        # should be an interesting case to study 2 bad sentseg.  One for party line.
        # might have something to do with font size.  We might want to use ystart, yend
        # instead of just yend.
        
        # 0.66
        self.assertGreater(prec, 0.64)
        self.assertLess(prec, 0.68)

        # 0.88
        self.assertGreater(recall, 0.86)
        self.assertLess(recall, 0.90)

        # 0.75
        self.assertGreater(f1, 0.73)
        self.assertLess(f1, 0.77)


    def test_doc_8964(self):
        doc_id = '8964'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # slightly worse than before.
        
        # 0.90
        self.assertGreater(prec, 0.88)
        self.assertLess(prec, 0.92)

        # 0.97
        self.assertGreater(recall, 0.95)
        self.assertLess(recall, 0.99)

        # 0.94
        self.assertGreater(f1, 0.92)
        self.assertLess(f1, 0.96)


    def test_doc_8969(self):
        doc_id = '8969'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # not an issue
        
        # 0.94
        self.assertGreater(prec, 0.92)
        self.assertLess(prec, 0.96)

        # 0.94
        self.assertGreater(recall, 0.92)
        self.assertLess(recall, 0.96)

        # 0.94
        self.assertGreater(f1, 0.92)
        self.assertLess(f1, 0.96)

    def test_doc_8970(self):
        doc_id = '8970'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # worse than before, a lot bad sentseg issue
        
        # 0.70
        self.assertGreater(prec, 0.68)
        self.assertLess(prec, 0.72)

        # 0.75
        self.assertGreater(recall, 0.73)
        self.assertLess(recall, 0.77)

        # 0.72
        self.assertGreater(f1, 0.70)
        self.assertLess(f1, 0.74)

    def test_doc_8971(self):
        doc_id = '8971'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # senteval resul: dir-work/8971.sent.txt  prec = 0.960, recall = 0.967, f1 = 0.964
        # we are doing ok, across page can still be improved

        # 0.96
        self.assertGreater(prec, 0.94)
        self.assertLess(prec, 0.98)

        # 0.97
        self.assertGreater(recall, 0.95)
        self.assertLess(recall, 0.99)

        # 0.96
        self.assertGreater(f1, 0.94)
        self.assertLess(f1, 0.98)

    def test_doc_8973(self):
        doc_id = '8973'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # ok
        
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

        # ok
        
        # 0.90
        self.assertGreater(prec, 0.88)
        self.assertLess(prec, 0.92)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.92
        self.assertGreater(f1, 0.90)
        self.assertLess(f1, 0.94)


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

        # ok, list differences
        
        # 0.92
        self.assertGreater(prec, 0.90)
        self.assertLess(prec, 0.94)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.93
        self.assertGreater(f1, 0.91)
        self.assertLess(f1, 0.95)

    def test_doc_8979(self):
        doc_id = '8979'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # 0.93
        self.assertGreater(prec, 0.91)
        self.assertLess(prec, 0.95)

        # 1.0
        self.assertGreater(recall, 0.98)
        self.assertLess(recall, 1.02)

        # 0.96
        self.assertGreater(f1, 0.94)
        self.assertLess(f1, 0.98)

    def test_doc_8980(self):
        doc_id = '8980'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # seemed ok, nothing alarming

        # 0.83
        self.assertGreater(prec, 0.81)
        self.assertLess(prec, 0.85)

        # 0.90
        self.assertGreater(recall, 0.88)
        self.assertLess(recall, 0.92)

        # 0.87
        self.assertGreater(f1, 0.85)
        self.assertLess(f1, 0.89)


    def test_doc_8982(self):
        doc_id = '8982'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # seemed ok, just one error
        # senteval resul: prec = 0.769, recall = 0.917, f1 = 0.837

        # 0.77
        self.assertGreater(prec, 0.75)
        self.assertLess(prec, 0.79)

        # 0.9
        self.assertGreater(recall, 0.88)
        self.assertLess(recall, 0.92)

        # 0.84
        self.assertGreater(f1, 0.82)
        self.assertLess(f1, 0.86)


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

        # seemed ok
        
        # 0.90
        self.assertGreater(prec, 0.88)
        self.assertLess(prec, 0.92)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.92
        self.assertGreater(f1, 0.90)
        self.assertLess(f1, 0.94)

    def test_doc_8986(self):
        doc_id = '8986'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # senteval resul: dir-work/8986.sent.txt  prec = 0.935, recall = 0.977, f1 = 0.956
        # seems ok
        
        # 0.94
        self.assertGreater(prec, 0.92)
        self.assertLess(prec, 0.96)

        # 0.98
        self.assertGreater(recall, 0.95)
        self.assertLess(recall, 1.00)

        # 0.96
        self.assertGreater(f1, 0.94)
        self.assertLess(f1, 0.98)

    def test_doc_8987(self):
        doc_id = '8987'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # ok
        # suffering due to no removal of lines

        # senteval resul: prec = 0.895, recall = 0.944, f1 = 0.919
        # 0.90
        self.assertGreater(prec, 0.88)
        self.assertLess(prec, 0.92)

        # 0.94
        self.assertGreater(recall, 0.92)
        self.assertLess(recall, 0.96)

        # 0.92
        self.assertGreater(f1, 0.90)
        self.assertLess(f1, 0.94)

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

        # should be better.  split on a party sentence.
        # but otherwise wok.
        
        # 0.72
        self.assertGreater(prec, 0.70)
        self.assertLess(prec, 0.74)

        # 0.90
        self.assertGreater(recall, 0.88)
        self.assertLess(recall, 0.92)

        # 0.80
        self.assertGreater(f1, 0.78)
        self.assertLess(f1, 0.82)

    def test_doc_8993(self):
        doc_id = '8993'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # so so,  Issue with not detecting header and footer as a block.
        # Should fix in future header, footer block refactoring, instead
        # of header, footer line.
        
        # 0.59
        self.assertGreater(prec, 0.57)
        self.assertLess(prec, 0.61)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.72
        self.assertGreater(f1, 0.70)
        self.assertLess(f1, 0.74)


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

        # ok
        
        # 0.85
        self.assertGreater(prec, 0.83)
        self.assertLess(prec, 0.87)

        # 0.89
        self.assertGreater(recall, 0.87)
        self.assertLess(recall, 0.91)

        # 0.87
        self.assertGreater(f1, 0.85)
        self.assertLess(f1, 0.89)

    def test_doc_8996(self):
        doc_id = '8996'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # ok
        # somewhat difficult case with joining paragraph across
        # pages.  All letters are capitalized
        
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

        # ok, except one annoying sentseg issue
        
        # 0.87
        self.assertGreater(prec, 0.85)
        self.assertLess(prec, 0.89)

        # 0.95
        self.assertGreater(recall, 0.93)
        self.assertLess(recall, 0.97)

        # 0.90
        self.assertGreater(f1, 0.88)
        self.assertLess(f1, 0.92)


    def test_doc_9003(self):
        doc_id = '9003'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # ok
        
        # 0.89
        self.assertGreater(prec, 0.87)
        self.assertLess(prec, 0.91)

        # 0.96
        self.assertGreater(recall, 0.94)
        self.assertLess(recall, 0.98)

        # 0.93
        self.assertGreater(f1, 0.91)
        self.assertLess(f1, 0.95)


    def test_doc_9012(self):
        doc_id = '9012'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # worse, a few not great sentseg issues

        # 0.73
        self.assertGreater(prec, 0.71)
        self.assertLess(prec, 0.75)

        # 0.98
        self.assertGreater(recall, 0.96)
        self.assertLess(recall, 1.00)

        # 0.83
        self.assertGreater(f1, 0.81)
        self.assertLess(f1, 0.85)


    def test_doc_9015(self):
        doc_id = '9015'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # ok
        # header and footer block might help
        
        # 0.86
        self.assertGreater(prec, 0.84)
        self.assertLess(prec, 0.88)

        # 0.96
        self.assertGreater(recall, 0.95)
        self.assertLess(recall, 0.98)

        # 0.90
        self.assertGreater(f1, 0.88)
        self.assertLess(f1, 0.92)


    def test_doc_9016(self):
        doc_id = '9016'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # ok, should be improvable
        
        # 0.90
        self.assertGreater(prec, 0.88)
        self.assertLess(prec, 0.92)

        # 0.94
        self.assertGreater(recall, 0.92)
        self.assertLess(recall, 0.96)

        # 0.92
        self.assertGreater(f1, 0.90)
        self.assertLess(f1, 0.94)


    def test_doc_9042(self):
        doc_id = '9042'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # something is going on, se have more sentseg issues
        
        # 0.82
        self.assertGreater(prec, 0.80)
        self.assertLess(prec, 0.82)

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

        # seemed ok, mainly because our evaluation function
        # doesn't handle the removal of output of page number well

        # there are some issue with paragraph merging across pages
        
        # 0.88
        self.assertGreater(prec, 0.86)
        self.assertLess(prec, 0.90)

        # 0.97
        self.assertGreater(recall, 0.95)
        self.assertLess(recall, 0.99)

        # 0.92
        self.assertGreater(f1, 0.90)
        self.assertLess(f1, 0.94)

    # todo, working
    def test_doc_9325(self):
        doc_id = '9325'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # ok, seems to be mainly related to footer and header not being removed.
        # which is the correct behavior
        # 0.82
        self.assertGreater(prec, 0.80)
        self.assertLess(prec, 0.84)

        # 0.92
        self.assertGreater(recall, 0.90)
        self.assertLess(recall, 0.94)

        # 0.87
        self.assertGreater(f1, 0.85)
        self.assertLess(f1, 0.89)


    def test_doc_9326(self):
        doc_id = '9326'
        gold_sent_list = read_sent_file('dir-sent-check/gold/{}.sent.tsv.gold'.format(doc_id))
        pred_sent_list = gen_sent_list(doc_id)
        prec, recall, f1 = calc_prec_recall_f1(gold_sent_list, pred_sent_list)

        # a lot of issue with header and footer removal in the gold data set
        # probably ok
        
        # 0.92
        self.assertGreater(prec, 0.90)
        self.assertLess(prec, 0.94)

        # 0.96
        self.assertGreater(recall, 0.94)
        self.assertLess(recall, 0.98)

        # 0.94
        self.assertGreater(f1, 0.92)
        self.assertLess(f1, 0.96)

