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

class TestBespokeSent(unittest.TestCase):

    # pylint: disable=too-many-locals
    def test_bespoke_12345(self):

        custid = 'cust_555'
        custid_data_dir = 'cust_555'
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
        # [[0, 19], [14, 32]]

        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 11, delta=6)
        self.assertAlmostEqual(fn, 11, delta=4)
        self.assertAlmostEqual(tp, 35, delta=4)

        # round(ant_result['f1'], 2),
        # 0.72
        f1 = round(ant_result['fscore'], 2)
        self.assertGreaterEqual(f1, 0.66)
        self.assertLessEqual(f1, 0.77)

        # round(ant_result['prec'], 2),
        # 0.71
        precision = round(ant_result['precision'], 2)
        self.assertGreaterEqual(precision, 0.65)
        self.assertLessEqual(precision, 0.76)

        recall = round(ant_result['recall'], 2)
        # 0.74
        self.assertGreaterEqual(recall, 0.70)
        self.assertLessEqual(recall, 0.78)

        txt_fnames = []
        for file in os.listdir(custid_data_dir):
            fname = '{}/{}'.format(custid_data_dir, file)
            if file.endswith(".txt"):
                txt_fnames.append(fname)

        provision = '{}.{}'.format(custid, ant_result['model_number'])
        return_lens = []
        for fname in sorted(txt_fnames)[20:30]:
            prov_labels_map = upload_annotate_doc(fname, provision)
            print('prov_labels_map')
            print(prov_labels_map)
            pred_list = prov_labels_map['ebannotations'].get(custid, [])
            print('pred list:')
            pprint.pprint(pred_list)
            return_lens.append(len(pred_list))
        self.assertEqual(return_lens, [0, 1, 1, 1, 1, 0, 0, 0, 1, 0])

if __name__ == "__main__":
    unittest.main()
