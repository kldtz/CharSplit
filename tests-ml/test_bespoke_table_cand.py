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

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 7, delta=2)
        self.assertAlmostEqual(fn, 3, delta=2)
        self.assertAlmostEqual(tp, 6, delta=2)

        # round(ant_result['f1'], 2)
        # 0.55
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.50)
        self.assertLessEqual(f1, 0.60)

        # round(ant_result['prec'], 2)
        # .46
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.42)
        self.assertLessEqual(precision, 0.50)

        # 0.66
        recall = round(ant_result['recall'], 2)
        self.assertGreaterEqual(recall, 0.62)
        self.assertLessEqual(recall, 0.70)
        txt_fnames = []
        for file in os.listdir(custid_data_dir):
            fname = '{}/{}'.format(custid_data_dir, file)
            if file.endswith(".txt"):
                txt_fnames.append(fname)


if __name__ == "__main__":
    unittest.main()
