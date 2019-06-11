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

class TestBespokeValues(unittest.TestCase):

    def test_bespoke_values(self):

        custid = '21'
        custid_data_dir = 'cust_' + custid
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           custid_data_dir,
                                           candidate_types='NUMBER,PERCENT,CURRENCY',
                                           nbest=-1)
        ajson = json.loads(result_text)
        print(ajson)
        ant_result = ajson['en']

        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # {'fn': 21, 'fp': 11, 'tn': 0, 'tp': 21}
        # [[0, 11], [21, 21]]

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 14, delta=2)
        self.assertAlmostEqual(fn, 20, delta=2)
        self.assertAlmostEqual(tp, 22, delta=2)

        # 0.57
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.54)
        self.assertLessEqual(f1, 0.58)

        # .66
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.59)
        self.assertLessEqual(precision, 0.63)

        # 0.50
        recall = round(ant_result['recall'], 2)
        self.assertGreaterEqual(recall, 0.50)
        self.assertLessEqual(recall, 0.54)

if __name__ == "__main__":
    unittest.main()
