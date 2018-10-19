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
                                           nbest=2)
        ajson = json.loads(result_text)
        ant_result = ajson['en']
        
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # [[0, 3], [14, 42]]

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 3, delta=2)
        self.assertAlmostEqual(fn, 14, delta=2)
        self.assertAlmostEqual(tp, 42, delta=2)

        # round(ant_result['f1'], 2)
        # 0.83
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.81)
        self.assertLessEqual(f1, 0.85)

        # round(ant_result['prec'], 2)
        # 0.93
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.91)
        self.assertLessEqual(precision, 0.95)

        # 0.75
        recall = round(ant_result['recall'], 2)
        self.assertGreaterEqual(recall, 0.73)
        self.assertLessEqual(recall, 0.77)

if __name__ == "__main__":
    unittest.main()
