#!/usr/bin/env python3

import json
import unittest
from typing import Any, Dict, List, Set, Tuple

from kirke.client import postfileutils
from kirke.utils import antdocutils


MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'

UNIT_TEST_PROVS = ['change_control',
                   'choiceoflaw',
                   'date',
                   'effectivedate',
                   'force_majeure',
                   'limliability',
                   'noncompete',
                   'party',
                   'remedy',
                   'renewal',
                   'termination',
                   'term',
                   'title',
                   'warranty',
                   'cust_9']

def upload_annotate_doc(file_name: str) -> Dict[str, Any]:

    text = postfileutils.post_unittest_annotate_document(file_name)

    ajson = json.loads(text)

    return ajson


def upload_get_antdoc_prov_list(file_name: str,
                                provision: str) -> List[Dict]:
    ajson = upload_annotate_doc(file_name)

    prov_list = antdocutils.get_ant_out_json_prov_list(ajson,
                                                       provision)
    return prov_list


def get_antdoc_validate_prov_list(file_name: str,
                                 provision: str) -> List[Dict]:
    prov_list = antdocutils.get_ant_out_file_prov_list(file_name, provision)
    return prov_list
    

class TestAntDocCat(unittest.TestCase):

    def test_antdoc_doccat(self):
        docid_pred_ajson_map = {}  # type: Dict[int, Any]
        docid_valid_ajson_map = {}  # type: Dict[int, Any]
        for docid in range(8285, 8301):
            txt_doc_fn = 'demo-txt/{}.txt'.format(docid)
            valid_doc_fn = 'demo-validate/{}.log'.format(docid)
            docid_pred_ajson_map[docid] = upload_annotate_doc(txt_doc_fn)
            docid_valid_ajson_map[docid] = antdocutils.get_ant_out_json(valid_doc_fn)
                                 
        for provision in UNIT_TEST_PROVS:

            for docid in range(8285, 8301):
                pred_ajson = docid_pred_ajson_map[docid]
                valid_ajson = docid_pred_ajson_map[docid]
                
                pred_prov_list = antdocutils.get_ant_out_json_prov_list(pred_ajson,
                                                                        provision)
            
                valid_prov_list = antdocutils.get_ant_out_json_prov_list(valid_ajson,
                                                                         provision)
                self.assertEqual(pred_prov_list,
                                 valid_prov_list)


if __name__ == "__main__":
    unittest.main()
