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

        custid = 'cust_555'
        custid_data_dir = 'cust_555'
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

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 25, delta=6)
        self.assertAlmostEqual(fn, 20, delta=4)
        self.assertAlmostEqual(tp, 73, delta=4)

        # round(ant_result['f1'], 2),
        # 0.76
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.70)
        self.assertLessEqual(f1, 0.82)

        # round(ant_result['prec'], 2),
        # 0.74
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.68)
        self.assertLessEqual(precision, 0.80)

        recall = round(ant_result['recall'], 2)
        # 0.78
        self.assertGreaterEqual(recall, 0.72)
        self.assertLessEqual(recall, 0.84)

if __name__ == "__main__":
    unittest.main()
