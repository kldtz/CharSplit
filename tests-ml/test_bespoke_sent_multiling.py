#!/usr/bin/env python3

import configparser
import json
import unittest
from typing import Any, Dict, List, Set, Tuple

from kirke.client import postfileutils


# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

def upload_annotate_doc(file_name: str, provision: str) -> Dict[str, Any]:
    text = postfileutils.post_unittest_annotate_document(file_name, provision)
    ajson = json.loads(text)

    return ajson

class TestBespokeSent(unittest.TestCase):

    def test_bespoke_sent(self):
        '''
        Tagged Romanian and Dutch europarl data. Of the tagged data from other
        testing, these two langs get F1 > 0.0 with only 5 docs, so we use these
        two to keep unit testing as quick as possible.
        '''

        '''
        # train the model with data from both langs
        custid = '39'
        custid_data_dir = 'cust_' + custid
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           custid_data_dir,
                                           candidate_types='SENTENCE',
                                           nbest=-1)
        '''

        # verify expected predictions on a training doc for each
        docid_pred_ajson_map = {}
        ro_doc_test = 'cust_39/2732.txt'
        docid_pred_ajson_map['ro'] = upload_annotate_doc(ro_doc_test, 'cust_39')
        print(docid_pred_ajson_map['ro']['ebannotations']['cust_39_ro'])

        nl_doc_test = 'cust_39/2783.txt'
        docid_pred_ajson_map['nl'] = upload_annotate_doc(nl_doc_test, 'cust_39')
        print(docid_pred_ajson_map['nl']['ebannotations']['cust_39_nl'])
        

if __name__ == "__main__":
    unittest.main()
