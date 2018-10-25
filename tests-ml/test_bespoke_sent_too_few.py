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

        custid = 'cust_555'
        custid_data_dir = 'cust_555-too-few'

        custid = 'cust_555'
        custid_data_dir = 'cust_555-too-few'

        result_resp = \
            postfileutils.upload_train_dir_resp(custid,
                                                custid_data_dir,
                                                candidate_types='SENTENCE',
                                                nbest=-1)
        self.assertEqual(result_resp.status_code, 500)
        self.assertTrue('INSUFFICIENT_EXAMPLES' in result_resp.text)
            

if __name__ == "__main__":
    unittest.main()
