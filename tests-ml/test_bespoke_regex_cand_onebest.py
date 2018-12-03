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

class TestBespokeRegex(unittest.TestCase):

    # pylint: disable=too-many-locals
    def test_bespoke_currency(self):

        custid = '9'
        custid_data_dir = 'cust_' + custid
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           custid_data_dir,
                                           candidate_types='CURRENCY',
                                           nbest=1)
        ajson = json.loads(result_text)
        ant_result = ajson['en']
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # {'fn': 3, 'fp': 4, 'tn': 0, 'tp': 98})
        # [[0, 4], [3, 98]])

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 4, delta=2)
        self.assertAlmostEqual(fn, 3, delta=2)
        self.assertAlmostEqual(tp, 98, delta=2)

        # round(ant_result['f1'], 2)
        # 0.96
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.94)
        self.assertLessEqual(f1, 0.98)

        # round(ant_result['prec'], 2)
        # .96
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.94)
        self.assertLessEqual(precision, 0.98)

        recall = round(ant_result['recall'], 2)
        # 0.97
        self.assertGreaterEqual(recall, 0.95)
        self.assertLessEqual(recall, 0.99)

        txt_fnames = []
        for file in os.listdir(custid_data_dir):
            fname = '{}/{}'.format(custid_data_dir, file)
            if file.endswith(".txt"):
                txt_fnames.append(fname)

        provision = '{}.{}'.format(custid_data_dir, ant_result['model_number'])
        return_lens = []
        for fname in sorted(txt_fnames)[:5]:
            prov_labels_map = upload_annotate_doc(fname, provision)
            print('prov_labels_map')
            print(prov_labels_map)
            date_list = prov_labels_map['ebannotations'].get(custid_data_dir, [])
            print('date_list:')
            pprint.pprint(date_list)
            return_lens.append(len(date_list))
        self.assertEqual(return_lens, [1, 0, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
