#!/usr/bin/env python3

import configparser
import json
import pprint
import unittest

from typing import Any, Dict, List, Tuple

from kirke.client import postfileutils


# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'

def upload_annotate_doc(file_name: str) -> Dict[str, Any]:

    text = postfileutils.post_annotate_document(file_name,
                                                ['cust_10'],
                                                is_detect_lang=False,
                                                is_classify_doc=False)

    ajson = json.loads(text)

    return ajson


class TestBespokeDate(unittest.TestCase):

    def test_bespoke_date(self):

        custid = '10'
        custid_data_dir = 'cust_' + custid
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           custid_data_dir,
                                           candidate_types='DATE',
                                           nbest=1)
        ajson = json.loads(result_text)
        ant_result = ajson['en']

        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        # [[0, 0], [16, 37]]

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 0, delta=2)
        self.assertAlmostEqual(fn, 16, delta=2)
        self.assertAlmostEqual(tp, 37, delta=2)

        # round(ant_result['f1'], 2)
        # 0.82
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.80)
        self.assertLessEqual(f1, 0.84)

        # round(ant_result['prec'], 2)
        # 1.0
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.98)

        # 0.70
        recall = round(ant_result['recall'], 2)
        self.assertGreaterEqual(recall, 0.68)
        self.assertLessEqual(recall, 0.72)

        prov_labels_map = upload_annotate_doc('cust_10/695.txt')
        print('prov_labels_map')
        print(prov_labels_map)
        # prov_labels_map = upload_annotate_doc('cust_10/736.txt')
        date_list = prov_labels_map['ebannotations'].get('cust_10', [])
        print('date_list:')
        pprint.pprint(date_list)
        self.assertEqual(len(date_list), 1)

if __name__ == "__main__":
    unittest.main()
