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
                                           upload_dir=custid_data_dir,
                                           candidate_types='CURRENCY',
                                           nbest=-1).text
        ajson = json.loads(result_text)
        ant_result = ajson['en']
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # {'fn': 3, 'fp': 24, 'tn': 0, 'tp': 101})
        # [[0, 24], [3, 101]]

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(24, fp, delta=2)
        self.assertAlmostEqual(5, fn, delta=2)
        self.assertAlmostEqual(99, tp, delta=2)

        # 0.88
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.85)
        self.assertLessEqual(f1, 0.89)

        # 0.81
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.79)
        self.assertLessEqual(precision, 0.83)

        # 0.97
        recall = round(ant_result['recall'], 2)
        self.assertGreaterEqual(recall, 0.92)
        self.assertLessEqual(recall, 0.96)


if __name__ == "__main__":
    unittest.main()
