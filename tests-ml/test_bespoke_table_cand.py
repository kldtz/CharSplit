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

def upload_annotate_doc(file_name: str, provision: str) -> Dict[str, Any]:

    text = postfileutils.post_annotate_document(file_name,
                                                [provision],
                                                is_detect_lang=False,
                                                is_classify_doc=False)

    ajson = json.loads(text)

    return ajson

class TestBespokeTable(unittest.TestCase):

    # pylint: disable=too-many-locals
    def test_bespoke_table(self):

        custid = 'rate_table'
        custid_data_dir = 'data-rate-table'
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           custid_data_dir,
                                           candidate_types='TABLE',
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

        # {'confusion_matrix': [[0, 7], [4, 5]], 'fscore': 0.4761904761904762, 'model_number': 1016, 'precision': 0.4166666666666667, 'provision': 'rate_table', 'recall': 0.5555555555555556}
        # {'confusion_matrix': [[0, 7], [3, 6]], 'fscore': 0.5454545454545455, 'model_number': 1181, 'precision': 0.46153846153846156, 'provision': 'rate_table', 'recall': 0.6666666666666666}
        # previous: {'confusion_matrix': [[0, 5], [3, 6]], 'fscore': 0.6, 'model_number': 1016, 'precision': 0.5454545454545454, 'provision': 'rate_table', 'recall': 0.6666666666666666}
        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 7, delta=2)
        self.assertAlmostEqual(fn, 4, delta=2)
        self.assertAlmostEqual(tp, 5, delta=2)

        # round(ant_result['f1'], 2)
        # 0.55, was 0.6, was 0.57, was 0.55
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.42)
        self.assertLessEqual(f1, 0.60)

        # round(ant_result['prec'], 2)
        # 0.46, was .55
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.41)
        self.assertLessEqual(precision, 0.51)

        # 0.66
        recall = round(ant_result['recall'], 2)
        self.assertGreaterEqual(recall, 0.50)
        self.assertLessEqual(recall, 0.71)
        txt_fnames = []
        for file in os.listdir(custid_data_dir):
            fname = '{}/{}'.format(custid_data_dir, file)
            if file.endswith(".txt"):
                txt_fnames.append(fname)


if __name__ == "__main__":
    unittest.main()
