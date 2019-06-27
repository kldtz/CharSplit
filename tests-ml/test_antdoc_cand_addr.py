#!/usr/bin/env python3

import copy
import json
import pprint
import unittest
from typing import Any, Dict, List, Tuple

from kirke.client import postfileutils
from kirke.utils import antdocutils
from kirke.utils import modelfileutils


MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'eb_files_test/pymodel'

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
                   'cust_9.1005',
                   'cust_42.1001']

UNIT_TEST_PROVS_V2 = ['change_control',
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
                      'cust_9.1005',
                      'cust_42.1179']


def upload_annotate_doc(file_name: str) -> Dict[str, Any]:
    text = postfileutils.upload_unittest_annotate_doc(file_name,
                                                      prov_list=UNIT_TEST_PROVS)
    ajson = json.loads(text)
    return ajson

def upload_annotate_doc_v2(file_name: str) -> Dict[str, Any]:
    text = postfileutils.upload_unittest_annotate_doc(file_name,
                                                      prov_list=UNIT_TEST_PROVS_V2)
    ajson = json.loads(text)
    return ajson


def annotate_doc_with_provision(docid: str, adir: str, provision: str) \
    -> List[Tuple[str, str, int, int, int]]:
    txt_doc_fn = '{}/{}.txt'.format(adir, docid)
    pred_ajson = upload_annotate_doc(txt_doc_fn)

    print("checking provision '{}' in {}".format(provision, txt_doc_fn))

    pred_prov_list = antdocutils.get_ant_out_json_prov_list(pred_ajson,
                                                            provision)
    return pred_prov_list

def annotate_doc_with_provision_v2(docid: str, adir: str, provision: str) \
    -> List[Tuple[str, str, int, int, int]]:
    txt_doc_fn = '{}/{}.txt'.format(adir, docid)
    pred_ajson = upload_annotate_doc_v2(txt_doc_fn)

    print("checking provision '{}' in {}".format(provision, txt_doc_fn))

    pred_prov_list = antdocutils.get_ant_out_json_prov_list(pred_ajson,
                                                            provision)
    return pred_prov_list



class TestAntCandAddr(unittest.TestCase):

    def test_antdoc_cand_addr_1968(self):
        self.maxDiff = None
        docid = '1968'
        prov_result_list = annotate_doc_with_provision(docid, 'cust_42', 'cust_42')

        # print("prov_result_list:")
        # pprint.pprint(prov_result_list)

        got_result = []
        for aresult in prov_result_list:
            got_result.append((aresult['start'],
                               aresult['end'],
                               aresult['text'].replace('\n', ' ')))
        print('got_result')
        print(got_result)

        expected_result = [(11189, 11276,
                            '4021   Stirrup Creek Drive           Suite 100           Durham,   North Carolina 27703'),
                           (11423, 11483,
                            '2880   Slater Road, Suite 200        Morrisville,   NC 27560')]

        self.assertEqual(expected_result, got_result)


    def test_antdoc_cand_addr_866(self):
        self.maxDiff = None
        docid = '866'
        prov_result_list = annotate_doc_with_provision(docid, 'cust_42', 'cust_42')

        # print("prov_result_list:")
        # pprint.pprint(prov_result_list)

        got_result = []
        for aresult in prov_result_list:
            got_result.append((aresult['start'],
                               aresult['end'],
                               aresult['text'].replace('\n', ' ')))
        print('got_result')
        print(got_result)

        expected_result = [(20205, 20252, '11501 Domain Drive, Suite 200  Austin, TX 78758')]

        self.assertEqual(expected_result, got_result)

    def test_antdoc_cand_addr_2008(self):
        self.maxDiff = None
        docid = '2008'
        prov_result_list = annotate_doc_with_provision(docid, 'cust_42', 'cust_42')

        # print("prov_result_list:")
        # pprint.pprint(prov_result_list)

        got_result = []
        for aresult in prov_result_list:
            got_result.append((aresult['start'],
                               aresult['end'],
                               aresult['text'].replace('\n', ' ')))
        print('got_result')
        print(got_result)

        expected_result = [(6754, 6823, "85 Mechanic Street, Rivermill Complex, Suite 400, Lebanon, NH, 03766.")]

        self.assertEqual(expected_result, got_result)


class TestAntCandAddrV2(unittest.TestCase):

    def test_antdoc_cand_addr_1968(self):
        self.maxDiff = None
        docid = '1968'
        prov_result_list = annotate_doc_with_provision_v2(docid, 'cust_42', 'cust_42')

        # print("prov_result_list:")
        # pprint.pprint(prov_result_list)

        got_result = []
        for aresult in prov_result_list:
            got_result.append((aresult['start'],
                               aresult['end'],
                               aresult['text'].replace('\n', ' ')))
        print('got_result')
        print(got_result)

        expected_result = [(11189, 11276,
                            '4021   Stirrup Creek Drive           Suite 100           Durham,   North Carolina 27703'),
                           (11423, 11483,
                            '2880   Slater Road, Suite 200        Morrisville,   NC 27560')]

        self.assertEqual(expected_result, got_result)


    def test_antdoc_cand_addr_866(self):
        self.maxDiff = None
        docid = '866'
        prov_result_list = annotate_doc_with_provision_v2(docid, 'cust_42', 'cust_42')

        # print("prov_result_list:")
        # pprint.pprint(prov_result_list)

        got_result = []
        for aresult in prov_result_list:
            got_result.append((aresult['start'],
                               aresult['end'],
                               aresult['text'].replace('\n', ' ')))
        print('got_result')
        print(got_result)

        expected_result = [(20205, 20252, '11501 Domain Drive, Suite 200  Austin, TX 78758')]

        self.assertEqual(expected_result, got_result)

    # this is different from v1
    def test_antdoc_cand_addr_2008(self):
        self.maxDiff = None
        docid = '2008'
        prov_result_list = annotate_doc_with_provision_v2(docid, 'cust_42', 'cust_42')

        # print("prov_result_list:")
        # pprint.pprint(prov_result_list)

        got_result = []
        for aresult in prov_result_list:
            got_result.append((aresult['start'],
                               aresult['end'],
                               aresult['text'].replace('\n', ' ')))
        print('got_result')
        print(got_result)

        expected_result = [(6754, 6822, "85 Mechanic Street, Rivermill Complex, Suite 400, Lebanon, NH, 03766")]

        self.assertEqual(expected_result, got_result)
        

if __name__ == "__main__":
    unittest.main()
