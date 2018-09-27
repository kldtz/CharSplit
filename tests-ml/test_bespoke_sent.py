#!/usr/bin/env python3

import configparser
import json
import unittest

from kirke.client import postfileutils


# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'

class TestBespokeSent(unittest.TestCase):

    def test_bespoke_12345(self):

        custid = '12345'
        custid_data_dir = 'cust_' + custid
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           custid_data_dir,
                                           candidate_types='SENTENCE',
                                           nbest=-1)
        ajson = json.loads(result_text)
        ant_result = ajson['en']
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # [[0, 1], [9, 139]]
        # {'fn': 9, 'fp': 1, 'tn': 0, 'tp': 139})
        # 0, 1, 0, 148
        # 7, 1, 0, 141
        # 9, 0, 0, 139
        # 3, 1, 0, 145

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 1, delta=2)
        self.assertAlmostEqual(fn, 6, delta=6)
        self.assertAlmostEqual(tp, 143, delta=7)

        # round(ant_result['f1'], 2),
        # 1.0
        # 0.97
        # 0.97
        # 0.99
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.96)
        self.assertLessEqual(f1, 1.0)

        # round(ant_result['prec'], 2),
        # 0.99
        # 0.99
        # 1.00
        # 0.99
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.98)
        self.assertLessEqual(precision, 1.0)

        recall = round(ant_result['recall'], 2)
        # 1.0
        # 0.95
        # 0.93
        # 0.98
        self.assertGreaterEqual(recall, 0.92)
        self.assertLessEqual(recall, 1.0)

        # self.assertEqual(round(ant_result['threshold'], 2),
        #                  0.24)

    def test_bespoke_9_pdf(self):
        """We can train using just regular provision name also.

        This particular test also focus on PDF files.
        """
        provision = 'term'
        custid_data_dir = 'cust_9_pdf'
        result_text = \
            postfileutils.upload_train_dir(provision,
                                           custid_data_dir,
                                           candidate_types='SENTENCE',
                                           nbest=-1)
        ajson = json.loads(result_text)
        ant_result = ajson['en']
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # [[0, 9], [6, 19]]
        # {'fn': 6, 'fp': 9, 'tn': 0, 'tp': 19})

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 9, delta=2)
        self.assertAlmostEqual(fn, 6, delta=6)
        self.assertAlmostEqual(tp, 19, delta=7)

        # round(ant_result['f1'], 2),
        # 0.72
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.70)
        self.assertLessEqual(f1, 0.74)

        # round(ant_result['prec'], 2),
        # 0.68
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.66)
        self.assertLessEqual(precision, 0.70)

        recall = round(ant_result['recall'], 2)
        # 0.76
        self.assertGreaterEqual(recall, 0.74)
        self.assertLessEqual(recall, 0.78)

if __name__ == "__main__":
    unittest.main()
