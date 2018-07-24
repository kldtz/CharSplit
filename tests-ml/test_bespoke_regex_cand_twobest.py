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

    def test_bespoke_currency(self):

        custid = '9'
        custid_data_dir = 'cust_' + custid
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           custid_data_dir,
                                           candidate_types='CURRENCY',
                                           nbest=2)
        ajson = json.loads(result_text)
        ant_result = ajson['en']
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # {'fn': 3, 'fp': 13, 'tn': 0, 'tp': 100})
        # [[0, 13], [3, 100]])

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 13, delta=2)
        self.assertAlmostEqual(fn, 3, delta=2)
        self.assertAlmostEqual(tp, 100, delta=2)

        # round(ant_result['f1'], 2)
        # 0.92
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.90)
        self.assertLessEqual(f1, 0.94)

        # round(ant_result['prec'], 2)
        # .88
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.86)
        self.assertLessEqual(precision, 0.90)

        recall = round(ant_result['recall'], 2)
        # 0.97
        self.assertGreaterEqual(recall, 0.95)
        self.assertLessEqual(recall, 0.99)



if __name__ == "__main__":
    unittest.main()
