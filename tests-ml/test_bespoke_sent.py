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

    def test_bespoke_555(self):

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

        # {'confusion_matrix': [[0, 24], [33, 60]], 'fscore': 0.6779661016949152, 'model_number': 1001, 'precision': 0.7142857142857143, 'provision': 'cust_555', 'recall': 0.6451612903225806}

        # old: {'confusion_matrix': [[0, 19], [30, 63]], 'fscore': 0.7199999999999999, 'model_number': 1020,
        #  'precision': 0.7682926829268293, 'provision': 'cust_555', 'recall': 0.6774193548387096}

        conf_matrix = ant_result['confusion_matrix']

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 24, delta=5)
        self.assertAlmostEqual(fn, 33, delta=5)
        self.assertAlmostEqual(tp, 60, delta=5)

        # round(ant_result['f1'], 2),
        # 0.68
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.63)
        self.assertLessEqual(f1, 0.73)

        # round(ant_result['prec'], 2),
        # 0.71
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.66)
        self.assertLessEqual(precision, 0.76)

        recall = round(ant_result['recall'], 2)
        # 0.65
        self.assertGreaterEqual(recall, 0.60)
        self.assertLessEqual(recall, 0.70)


if __name__ == "__main__":
    unittest.main()
