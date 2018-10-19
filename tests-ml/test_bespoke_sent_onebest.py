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
                                           nbest=1)
        ajson = json.loads(result_text)
        ant_result = ajson['en']
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # [[0, 4], [7, 141]] 

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 4, delta=2)
        self.assertAlmostEqual(fn, 7, delta=6)
        self.assertAlmostEqual(tp, 141, delta=7)

        # round(ant_result['f1'], 2),
        # 0.96
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.94)
        self.assertLessEqual(f1, 0.98)

        # round(ant_result['prec'], 2),
        # 0.97
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.95)
        self.assertLessEqual(precision, 0.99)

        recall = round(ant_result['recall'], 2)
        # 0.95
        self.assertGreaterEqual(recall, 0.93)
        self.assertLessEqual(recall, 0.97)

        # self.assertEqual(round(ant_result['threshold'], 2),
        #                  0.24)


if __name__ == "__main__":
    unittest.main()
