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
                                           nbest=-1)
        ajson = json.loads(result_text)
        ant_result = ajson['en']
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']

        self.assertEqual(conf_matrix,
                         # {'fn': 9, 'fp': 1, 'tn': 0, 'tp': 139})
                         [[0, 1], [9, 139]])
        
        # round(ant_result['f1'], 2),
        self.assertEqual(round(ant_result['fscore'], 2),
                         0.97)
        # round(ant_result['prec'], 2),
        self.assertEqual(round(ant_result['precision'], 2),
                         .99)
        self.assertEqual(round(ant_result['recall'], 2),
                         0.94)
        # self.assertEqual(round(ant_result['threshold'], 2),
        #                  0.24)


if __name__ == "__main__":
    unittest.main()
