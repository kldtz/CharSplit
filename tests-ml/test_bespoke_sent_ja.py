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

        # jurisdiction
        custid = 'cust_13263'
        fname_list_fname = 'data/japanese/jurisdiction_train_doclist.txt'
        result_text = \
            postfileutils.upload_train_files(custid,
                                             fname_list_fname=fname_list_fname,
                                             candidate_types='SENTENCE',
                                             nbest=-1).text
        ajson = json.loads(result_text)
        ant_result = ajson['ja']

        print("ant_result:")
        print(ant_result)

        """
        {'confusion_matrix': [[0, 1], [1, 6]],
         'fscore': 0.857,
         'model_number': 1134,
         'precision': 0.857,
         'provision': 'cust_13263',
         'recall': 0.857}
        """

        conf_matrix = ant_result['confusion_matrix']
        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 1, delta=2)
        self.assertAlmostEqual(fn, 1, delta=2)
        self.assertAlmostEqual(tp, 6, delta=2)

        # 0.86
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.81)
        self.assertLessEqual(f1, 0.91)

        # 0.86
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.81)
        self.assertLessEqual(precision, 0.91)

        # 0.86
        recall = round(ant_result['recall'], 2)
        self.assertGreaterEqual(recall, 0.81)
        self.assertLessEqual(recall, 0.91)


if __name__ == "__main__":
    unittest.main()
