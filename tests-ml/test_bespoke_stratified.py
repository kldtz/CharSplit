#!/usr/bin/env python3

import configparser
import json
import pprint
import unittest

from kirke.client import postfileutils


# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'

class TestBespokeStratifiedGroupKFold(unittest.TestCase):

    def test_bespoke_myparty(self):

        custid = 'my_party'
        custid_data_dir = 'data-myparty'
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           upload_dir=custid_data_dir,
                                           candidate_types='SENTENCE',
                                           nbest=-1).text
        ajson = json.loads(result_text)

        # print("ajson:")
        # print(ajson)
        ant_result = ajson['fr']
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']
        tn = conf_matrix[0][0]
        fp = conf_matrix[0][1]
        fn = conf_matrix[1][0]
        tp = conf_matrix[1][1]

        # We don't care about the numerical result.
        # We just want to make sure that the stratifiedgroupkfold works
        # and the training is successful.
        # If StratifiedGroupKFold is not used, we will get
        #    grid_search.fit()...
        #    Multiprocessing exception:
        #    ValueError: Class label 1 not present.
        # This error is due to without stratification, 2 fold might
        # get no positive training documents during GridSearchCV.
        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 0, delta=2)
        self.assertAlmostEqual(fn, 0, delta=2)
        self.assertAlmostEqual(tp, 6, delta=2)

    def test_bespoke_myparty_fail(self):

        custid = 'my_party'
        custid_data_dir = 'data-myparty-fail'
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           upload_dir=custid_data_dir,
                                           candidate_types='SENTENCE',
                                           nbest=-1).text
        ajson = json.loads(result_text)
        print('ajson')
        pprint.pprint(ajson)
        user_msg = ajson['fr']['user_message']
        self.assertTrue(user_msg, "< 6")


if __name__ == "__main__":
    unittest.main()
