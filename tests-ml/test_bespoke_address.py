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

class TestBespokeAddr(unittest.TestCase):

    def test_bespoke_addr(self):

        custid = '42'
        custid_data_dir = 'cust_' + custid
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           custid_data_dir,
                                           candidate_types='ADDRESS',
                                           nbest=-1)
        ajson = json.loads(result_text)
        ant_result = ajson['en']
        
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # {'fn': 12, 'fp': 6, 'tn': 0, 'tp': 18}
        # [[0, 6], [12, 18]]

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 6, delta=2)
        self.assertAlmostEqual(fn, 12, delta=2)
        self.assertAlmostEqual(tp, 18, delta=2)

        # round(ant_result['f1'], 2)
        # 0.66
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.64)
        self.assertLessEqual(f1, 0.68)

        # round(ant_result['prec'], 2)
        # .75
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.73)
        self.assertLessEqual(precision, 0.77)

        recall = round(ant_result['recall'], 2)
        # 0.6
        self.assertGreaterEqual(recall, 0.58)
        self.assertLessEqual(recall, 0.63)

if __name__ == "__main__":
    unittest.main()
