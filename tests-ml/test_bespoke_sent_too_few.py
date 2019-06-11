#!/usr/bin/env python3

import configparser
import json
import re
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

    def test_bespoke_555(self):

        custid = 'cust_555'
        custid_data_dir = 'cust_555-too-few'

        custid = 'cust_555'
        custid_data_dir = 'cust_555-too-few'

        result_resp = \
            postfileutils.upload_train_dir_resp(custid,
                                                custid_data_dir,
                                                candidate_types='SENTENCE',
                                                nbest=-1)
        print('result_resp.text')
        print(result_resp.text)
        print('result_resp')
        print(result_resp)

        self.assertEqual(result_resp.status_code, 200)
        self.assertTrue(re.search(r'only.*positive training documents',
                                  result_resp.text, re.I))


if __name__ == "__main__":
    unittest.main()
