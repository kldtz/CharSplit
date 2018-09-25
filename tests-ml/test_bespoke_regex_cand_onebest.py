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

class TestBespokeRegex(unittest.TestCase):

    def test_bespoke_currency(self):

        custid = '9'
        custid_data_dir = 'cust_' + custid
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           custid_data_dir,
                                           candidate_types='CURRENCY',
                                           nbest=1)
        ajson = json.loads(result_text)
        ant_result = ajson['en']
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']

        # {'fn': 10, 'fp': 2, 'tn': 0, 'tp': 90})
        # [[0, 2], [10, 90]])

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 2, delta=2)
        self.assertAlmostEqual(fn, 10, delta=2)
        self.assertAlmostEqual(tp, 90, delta=2)

        # round(ant_result['f1'], 2)
        # 0.94
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.92)
        self.assertLessEqual(f1, 0.96)

        # round(ant_result['prec'], 2)
        # .98
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.96)
        self.assertLessEqual(precision, 1.00)

        recall = round(ant_result['recall'], 2)
        # 0.90
        self.assertGreaterEqual(recall, 0.88)
        self.assertLessEqual(recall, 0.92)


if __name__ == "__main__":
    unittest.main()
