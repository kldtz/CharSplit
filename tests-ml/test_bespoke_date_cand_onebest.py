#!/usr/bin/env python3

import os
import configparser
import json
import pprint
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


class TestBespokeDate(unittest.TestCase):

    # pylint: disable=too-many-locals
    def test_bespoke_date(self):

        custid = '10'
        custid_data_dir = 'cust_' + custid
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           upload_dir=custid_data_dir,
                                           candidate_types='DATE',
                                           nbest=1).text
        ajson = json.loads(result_text)
        ant_result = ajson['en']

        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # [[0, 1], [11, 42]]

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        # {'confusion_matrix': [[0, 2], [11, 42]], 'fscore': 0.865979381443299, 'model_number': 1029, 'precision': 0.9545454545454546, 'provision': 'cust_10', 'recall': 0.7924528301886793}

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 2, delta=2)
        self.assertAlmostEqual(fn, 11, delta=2)
        self.assertAlmostEqual(tp, 42, delta=2)

        # round(ant_result['f1'], 2)
        # 0.87, was 86
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.85)
        self.assertLessEqual(f1, 0.89)

        # round(ant_result['prec'], 2)
        # 0.96, was 0.97
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.94)
        self.assertLessEqual(f1, 0.98)

        # 0.79
        recall = round(ant_result['recall'], 2)
        self.assertGreaterEqual(recall, 0.77)
        self.assertLessEqual(recall, 0.81)

        txt_fnames = []
        for file in os.listdir(custid_data_dir):
            fname = '{}/{}'.format(custid_data_dir, file)
            if file.endswith(".txt"):
                txt_fnames.append(fname)

        provision = '{}.{}'.format(custid_data_dir, ant_result['model_number'])
        return_lens = []
        for fname in sorted(txt_fnames)[10:20]:
            print(fname)
            prov_labels_map = upload_annotate_doc(fname, provision)
            print('prov_labels_map')
            print(prov_labels_map)
            date_list = prov_labels_map['ebannotations'].get(custid_data_dir, [])
            print('date_list:')
            pprint.pprint(date_list)
            return_lens.append(len(date_list))
        self.assertEqual(return_lens, [1, 1, 1, 0, 1, 1, 0, 1, 1, 1])


    # The following data set will cause div by zero error
    # in 7.0-maintenance branch.
    # Resolved in 8.0-maintenance branch.
    def test_bespoke_date_div_zero_7m(self):

        custid = '10'
        custid_data_dir = 'cust_10-1best-div-zero'
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           upload_dir=custid_data_dir,
                                           candidate_types='DATE',
                                           nbest=1).text
        ajson = json.loads(result_text)
        ant_result = ajson['en']

        print("ant_result:")
        print(ant_result)

        # We don't care about numerical performance.
        # We just need it to not fail or div-by-zero

        # round(ant_result['f1'], 2)
        # 0.22
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.1)

if __name__ == "__main__":
    unittest.main()
