#!/usr/bin/env python3

import os
import pprint
import configparser
import json
import unittest

from typing import Any, Dict

from kirke.client import postfileutils


# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'


class TestBespokeSentKorean(unittest.TestCase):

    # pylint: disable=too-many-locals
    def test_bespoke_sent(self):

        custid = 'cust_4'
        custid_data_dir = 'dir-korean/text'
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           upload_dir=custid_data_dir,
                                           candidate_types='SENTENCE',
                                           nbest=-1).text
        ajson = json.loads(result_text)
        ant_result = ajson['ko']

        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 0, delta=2)
        self.assertAlmostEqual(fn, 0, delta=2)
        self.assertAlmostEqual(tp, 7, delta=2)

        # 1.0
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.98)
        self.assertLessEqual(f1, 1.01)

        # 1.0        
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.98)
        self.assertLessEqual(precision, 1.01)

        # 1.0
        recall = round(ant_result['recall'], 2)
        self.assertGreaterEqual(recall, 0.98)
        self.assertLessEqual(recall, 1.01)


if __name__ == "__main__":
    unittest.main()
