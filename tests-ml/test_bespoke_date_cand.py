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

class TestBespokeDate(unittest.TestCase):

    def test_bespoke_date(self):

        custid = '10'
        custid_data_dir = 'cust_' + custid
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           custid_data_dir,
                                           candidate_types='DATE',
                                           nbest=-1)
        ajson = json.loads(result_text)
        ant_result = ajson['en']

        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # {'fn': 15, 'fp': 7, 'tn': 0, 'tp': 42}
        # [[0, 7], [15, 42]]

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 7, delta=2)
        self.assertAlmostEqual(fn, 15, delta=2)
        self.assertAlmostEqual(tp, 42, delta=2)

        # round(ant_result['f1'], 2)
        # 0.79
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.77)
        self.assertLessEqual(f1, 0.81)

        # round(ant_result['prec'], 2)
        # .86
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.84)
        self.assertLessEqual(precision, 0.88)

        # 0.74
        recall = round(ant_result['recall'], 2)
        self.assertGreaterEqual(recall, 0.72)
        self.assertLessEqual(recall, 0.76)

if __name__ == "__main__":
    unittest.main()