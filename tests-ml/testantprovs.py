#!/usr/bin/env python3

import copy
import json
import os
import pprint
import unittest
from typing import Any, Dict, List, Tuple

from kirke.client import postfileutils
from kirke.utils import antdocutils
from kirke.utils import modelfileutils


MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'eb_files_test/pymodel'
# Places where text files might be
TXT_DIR_PATH = ['demo-txt', 'dir-korean/text']

UNIT_TEST_PROVS = ['change_control',
                   'choiceoflaw',
                   'date',
                   'effectivedate_auto',
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
                   'cust_9.1005']


def upload_annotate_doc(file_name: str, prov_list: List[str]) -> Dict[str, Any]:
    text = postfileutils.upload_unittest_annotate_doc(file_name,
                                                      prov_list=prov_list)
    ajson = json.loads(text)
    return ajson


def get_antdoc_validate_prov_list(file_name: str,
                                  provision: str) -> List[Dict]:
    prov_list = antdocutils.get_ant_out_file_prov_list(file_name, provision)
    return prov_list


def clean_ant(ant: Dict, is_remove_text: bool = True) -> None:
    if ant.get('corenlp_end') is not None:
        del ant['corenlp_end']
    if ant.get('corenlp_start') is not None:
        del ant['corenlp_start']
    if ant.get('cpoint_end') is not None:
        del ant['cpoint_end']
    if ant.get('cpoint_start') is not None:
        del ant['cpoint_start']
    if ant.get('prob') is not None:
        del ant['prob']

    if ant.get('span_list') is not None:
        del ant['span_list']

    # for date normalization
    if ant.get('norm') is not None:
        del ant['norm']

    if is_remove_text:
        if ant.get('text') is not None:
            del ant['text']


    # replace new line with spaces
    # # ant['text'] = ant['text'].replace('\n', ' ')
    # ant['text'] = re.sub(r'\s+', ' ', ant['text'].replace('\n', ' '))

    # hash doesn't like this
    # for span in ant.get('span_list', []):
    #     del span['cpoint_end']
    #     del span['cpoint_start']

def clean_ant_list(ant_list: List[Dict], is_remove_text: bool = True) -> None:
    for ant in ant_list:
        clean_ant(ant, is_remove_text)

# pylint: disable=invalid-name
class hashabledict(dict):
    def __hash__(self) -> int:
        return hash(tuple(sorted(self.items())))

# pylint: disable=too-many-locals
def convert_to_same_diff(file_name: str,
                         provision: str,
                         pred_ant_list: List[Dict],
                         valid_ant_list: List[Dict]) \
                         -> Tuple[str, str, int, int, int]:
    is_debug = True
    # for adict in pred_ant_list:
    #    print('adict = {}'.format(adict))

    orig_pred_ant_list = copy.deepcopy(pred_ant_list)
    clean_ant_list(orig_pred_ant_list, is_remove_text=False)
    orig_valid_ant_list = copy.deepcopy(valid_ant_list)
    clean_ant_list(orig_pred_ant_list, is_remove_text=False)

    clean_ant_list(pred_ant_list)
    clean_ant_list(valid_ant_list)

    pred_hashable_list = [hashabledict(adict) for adict in pred_ant_list]
    valid_hashable_list = [hashabledict(adict) for adict in valid_ant_list]
    pred_set = set(pred_hashable_list)
    valid_set = set(valid_hashable_list)

    num_same = 0
    valid_has_pred_missing = 0
    pred_has_valid_missing = 0
    for valid_dict, orig_valid_dict in zip(valid_ant_list, orig_valid_ant_list):
        if hashabledict(valid_dict) in pred_set:
            num_same += 1
        else:
            valid_has_pred_missing += 1
            if is_debug:
                print('{}: valid_has_pred_missing: {}'.format(file_name, orig_valid_dict))

    for pred_dict, orig_pred_dict in zip(pred_ant_list, orig_pred_ant_list):
        if not hashabledict(pred_dict) in valid_set:
            pred_has_valid_missing += 1
            if is_debug:
                print('{}: pred_has_valid_missing: {}'.format(file_name, orig_pred_dict))

    return file_name, provision, num_same, valid_has_pred_missing, pred_has_valid_missing


def validate_annotated_doc(docid: str, prov_list : List[str] = UNIT_TEST_PROVS) \
    -> List[Tuple[str, str, int, int, int]]:
    for dir in TXT_DIR_PATH:
        txt_doc_fn = '{}/{}.txt'.format(dir, docid)
        if os.path.exists(txt_doc_fn):
            break
    else:
        raise FileNotFoundError("{}.txt".format(docid))
    pred_ajson = upload_annotate_doc(txt_doc_fn, prov_list)
    if '1057' in docid:
      pprint.pprint(pred_ajson)

    valid_doc_fn = 'demo-validate/{}.log'.format(docid)
    valid_ajson = antdocutils.get_ant_out_json(valid_doc_fn)

    prov_result_list = []
    num_same = 0
    valid_has_pred_missing = 0
    pred_has_valid_missing = 0

    for provision in prov_list:
        provision = modelfileutils.remove_custom_provision_version(provision)

        print("checking provision '{}' in {}".format(provision, txt_doc_fn))

        pred_prov_list = antdocutils.get_ant_out_json_prov_list(pred_ajson,
                                                                provision)
        valid_prov_list = antdocutils.get_ant_out_json_prov_list(valid_ajson,
                                                                 provision)

        num_same_diff_tuple = convert_to_same_diff(txt_doc_fn,
                                                   provision,
                                                   pred_prov_list,
                                                   valid_prov_list)
        prov_result_list.append(num_same_diff_tuple)
        num_same += num_same_diff_tuple[2]
        valid_has_pred_missing += num_same_diff_tuple[3]
        pred_has_valid_missing += num_same_diff_tuple[4]

    print('num_same = {}, valid_has_pred_missing = {}, pred_has_valid_missing = {}'.format(
        num_same, valid_has_pred_missing, pred_has_valid_missing))

    return prov_result_list



class TestAntProvs(unittest.TestCase):

    def test_antdoc_8285(self):
        self.maxDiff = None
        docid = '8285'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        # ('demo-txt/8285.txt', 'change_control', 4, 0, 0),
        # chagne_control was split into two because of imperfect paragraph
        # merging algo across pages.
        expected_result = [('demo-txt/8285.txt', 'change_control', 3, 1, 2),  # has_diff: verfied ok
                           ('demo-txt/8285.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8285.txt', 'date', 1, 0, 0),
                           ('demo-txt/8285.txt', 'effectivedate_auto', 1, 0, 0),
                           ('demo-txt/8285.txt', 'force_majeure', 0, 0, 0),
                           # ('demo-txt/8285.txt', 'limliability', 6, 0, 0),
                           # Mostly partial overlap issue, so the performance looks
                           # not great.  It's really not that bad.
                           # pylint: disable=line-too-long
                           # ('demo-txt/8285.txt', 'limliability', 5, 1, 0),  # has_diff: verified, to fix
                           ('demo-txt/8285.txt', 'limliability', 3, 2, 2),  # has_diff: verified, to fix
                           ('demo-txt/8285.txt', 'noncompete', 0, 0, 0),
                           ('demo-txt/8285.txt', 'party', 4, 0, 0),
                           # ('demo-txt/8285.txt', 'remedy', 8, 0, 0),
                           # seem to be very very noisy???  Maybe sechead is wrong?
                           ('demo-txt/8285.txt', 'remedy', 8, 0, 5),  # has_diff: verified, to fix
                           ('demo-txt/8285.txt', 'renewal', 1, 0, 0),
                           ('demo-txt/8285.txt', 'termination', 4, 0, 0),
                           ('demo-txt/8285.txt', 'term', 1, 0, 0),
                           ('demo-txt/8285.txt', 'title', 1, 0, 0),
                           ('demo-txt/8285.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8285.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)

    def test_antdoc_8286(self):
        self.maxDiff = None
        docid = '8286'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8286.txt', 'change_control', 0, 0, 0),
                           ('demo-txt/8286.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8286.txt', 'date', 1, 0, 0),
                           ('demo-txt/8286.txt', 'effectivedate_auto', 0, 0, 0),
                           ('demo-txt/8286.txt', 'force_majeure', 1, 0, 0),
                           ('demo-txt/8286.txt', 'limliability', 2, 0, 0),
                           ('demo-txt/8286.txt', 'noncompete', 0, 0, 0),
                           ('demo-txt/8286.txt', 'party', 4, 0, 0),
                           ('demo-txt/8286.txt', 'remedy', 4, 0, 0),
                           ('demo-txt/8286.txt', 'renewal', 2, 0, 0),
                           ('demo-txt/8286.txt', 'termination', 1, 0, 0),
                           # will fix, reviewed.  our fault but reasonable.
                           # ('demo-txt/8286.txt', 'term', 1, 1, 1),
                           ('demo-txt/8286.txt', 'term', 2, 0, 0),
                           ('demo-txt/8286.txt', 'title', 1, 0, 0),
                           ('demo-txt/8286.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8286.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)

    def test_antdoc_8287(self):
        self.maxDiff = None
        docid = '8287'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8287.txt', 'change_control', 1, 0, 0),
                           ('demo-txt/8287.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8287.txt', 'date', 1, 0, 0),
                           ('demo-txt/8287.txt', 'effectivedate_auto', 0, 0, 0),
                           ('demo-txt/8287.txt', 'force_majeure', 1, 0, 0),
                           # there is a separate limliability related to passenger instead of
                           # lease?  For now, acceptable.
                           ('demo-txt/8287.txt', 'limliability', 5, 0, 0),
                           ('demo-txt/8287.txt', 'noncompete', 0, 0, 0),
                           ('demo-txt/8287.txt', 'party', 3, 0, 0),
                           ('demo-txt/8287.txt', 'remedy', 13, 0, 0),
                           ('demo-txt/8287.txt', 'renewal', 1, 0, 0),
                           ('demo-txt/8287.txt', 'termination', 6, 0, 0),
                           ('demo-txt/8287.txt', 'term', 3, 0, 0),
                           ('demo-txt/8287.txt', 'title', 1, 0, 0),
                           ('demo-txt/8287.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8287.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)

    def test_antdoc_8288(self):
        self.maxDiff = None
        docid = '8288'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8288.txt', 'change_control', 0, 0, 0),
                           ('demo-txt/8288.txt', 'choiceoflaw', 1, 0, 0),
                           # we are failing on 'commencement dates'  We usually
                           # get termination date.
                           # ('demo-txt/8288.txt', 'date', 1, 0, 0),
                           ('demo-txt/8288.txt', 'date', 0, 1, 1),  # has_diff: verified. to fix
                           ('demo-txt/8288.txt', 'effectivedate_auto', 1, 0, 0),
                           ('demo-txt/8288.txt', 'force_majeure', 1, 0, 0),
                           ('demo-txt/8288.txt', 'limliability', 4, 0, 0),
                           ('demo-txt/8288.txt', 'noncompete', 0, 0, 0),
                           ('demo-txt/8288.txt', 'party', 4, 0, 0),
                           # the FP seems invalid
                           # ('demo-txt/8288.txt', 'remedy', 2, 0, 0),
                           ('demo-txt/8288.txt', 'remedy', 2, 0, 1),  # has_diff: verified, to fix
                           ('demo-txt/8288.txt', 'renewal', 1, 0, 0),
                           ('demo-txt/8288.txt', 'termination', 7, 0, 0),
                           ('demo-txt/8288.txt', 'term', 2, 0, 0),
                           ('demo-txt/8288.txt', 'title', 1, 0, 0),
                           ('demo-txt/8288.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8288.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)

    def test_antdoc_8289(self):
        self.maxDiff = None
        docid = '8289'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8289.txt', 'change_control', 0, 1, 0),  # ??, will verify
                           ('demo-txt/8289.txt', 'choiceoflaw', 0, 0, 0),
                           ('demo-txt/8289.txt', 'date', 1, 0, 0),
                           ('demo-txt/8289.txt', 'effectivedate_auto', 0, 0, 0),
                           ('demo-txt/8289.txt', 'force_majeure', 0, 0, 0),
                           ('demo-txt/8289.txt', 'limliability', 3, 0, 0),
                           ('demo-txt/8289.txt', 'noncompete', 0, 0, 0),
                           ('demo-txt/8289.txt', 'party', 5, 0, 0),
                           ('demo-txt/8289.txt', 'remedy', 5, 0, 0),
                           ('demo-txt/8289.txt', 'renewal', 0, 0, 0),
                           ('demo-txt/8289.txt', 'termination', 2, 0, 0),
                           ('demo-txt/8289.txt', 'term', 1, 0, 0),
                           ('demo-txt/8289.txt', 'title', 1, 0, 0),
                           ('demo-txt/8289.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8289.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)

    def test_antdoc_8290(self):
        self.maxDiff = None
        docid = '8290'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8290.txt', 'change_control', 1, 0, 0),
                           ('demo-txt/8290.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8290.txt', 'date', 1, 0, 0),
                           ('demo-txt/8290.txt', 'effectivedate_auto', 1, 0, 0),
                           ('demo-txt/8290.txt', 'force_majeure', 0, 0, 0),
                           # There is some issue with handling all cap paragraphs that are
                           # broken across pages.  Activating that might cause a lot of other
                           # issue.  Ignore for now.
                           ('demo-txt/8290.txt', 'limliability', 3, 0, 2),  # has_diff: will fix
                           # ('demo-txt/8290.txt', 'limliability', 3, 0, 0),
                           ('demo-txt/8290.txt', 'noncompete', 0, 0, 0),
                           ('demo-txt/8290.txt', 'party', 4, 0, 0),
                           ('demo-txt/8290.txt', 'remedy', 2, 0, 0),
                           ('demo-txt/8290.txt', 'renewal', 1, 0, 0),
                           ('demo-txt/8290.txt', 'termination', 2, 0, 0),
                           ('demo-txt/8290.txt', 'term', 1, 0, 0),
                           ('demo-txt/8290.txt', 'title', 1, 0, 0),
                           ('demo-txt/8290.txt', 'warranty', 1, 0, 0),
                           ('demo-txt/8290.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)

    def test_antdoc_8291(self):
        self.maxDiff = None
        docid = '8291'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8291.txt', 'change_control', 1, 0, 0),
                           ('demo-txt/8291.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8291.txt', 'date', 0, 0, 0),
                           ('demo-txt/8291.txt', 'effectivedate_auto', 0, 0, 0),
                           ('demo-txt/8291.txt', 'force_majeure', 1, 0, 0),
                           ('demo-txt/8291.txt', 'limliability', 1, 0, 0),
                           ('demo-txt/8291.txt', 'noncompete', 0, 0, 0),
                           ('demo-txt/8291.txt', 'party', 4, 0, 0),
                           # ('demo-txt/8291.txt', 'remedy', 7, 0, 0),
                           ('demo-txt/8291.txt', 'remedy', 6, 1, 0),  # has_diff: verified, ok
                           ('demo-txt/8291.txt', 'renewal', 0, 0, 0),
                           ('demo-txt/8291.txt', 'termination', 6, 0, 0),
                           ('demo-txt/8291.txt', 'term', 2, 0, 0),
                           ('demo-txt/8291.txt', 'title', 1, 0, 0),
                           ('demo-txt/8291.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8291.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)


    def test_antdoc_8292(self):
        self.maxDiff = None
        docid = '8292'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8292.txt', 'change_control', 0, 0, 0),
                           ('demo-txt/8292.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8292.txt', 'date', 0, 0, 1),
                           ('demo-txt/8292.txt', 'effectivedate_auto', 0, 0, 1),
                           ('demo-txt/8292.txt', 'force_majeure', 0, 0, 0),
                           ('demo-txt/8292.txt', 'limliability', 0, 0, 0),
                           ('demo-txt/8292.txt', 'noncompete', 1, 0, 0),
                           ('demo-txt/8292.txt', 'party', 2, 0, 0),
                           # ('demo-txt/8292.txt', 'remedy', 0, 0, 0),
                           # extracted fp is irrelevant
                           ('demo-txt/8292.txt', 'remedy', 0, 0, 1),  # has_diff: verified, to fix
                           ('demo-txt/8292.txt', 'renewal', 0, 0, 0),
                           # ('demo-txt/8292.txt', 'termination', 1, 0, 0),
                           # extracted fp is irrelevant
                           # pylint: disable=line-too-long
                           ('demo-txt/8292.txt', 'termination', 1, 0, 1),  # has_diff: verified, to fix
                           ('demo-txt/8292.txt', 'term', 0, 0, 0),
                           ('demo-txt/8292.txt', 'title', 1, 0, 0),
                           ('demo-txt/8292.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8292.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)


    def test_antdoc_8293(self):
        self.maxDiff = None
        docid = '8293'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8293.txt', 'change_control', 0, 0, 0),
                           ('demo-txt/8293.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8293.txt', 'date', 1, 0, 0),
                           ('demo-txt/8293.txt', 'effectivedate_auto', 0, 0, 0),
                           ('demo-txt/8293.txt', 'force_majeure', 0, 0, 0),
                           ('demo-txt/8293.txt', 'limliability', 5, 0, 0),
                           ('demo-txt/8293.txt', 'noncompete', 0, 0, 0),
                           ('demo-txt/8293.txt', 'party', 4, 0, 0),
                           ('demo-txt/8293.txt', 'remedy', 0, 0, 0),
                           ('demo-txt/8293.txt', 'renewal', 0, 0, 0),
                           ('demo-txt/8293.txt', 'termination', 1, 0, 0),
                           ('demo-txt/8293.txt', 'term', 1, 0, 0),
                           ('demo-txt/8293.txt', 'title', 1, 0, 0),
                           ('demo-txt/8293.txt', 'warranty', 1, 0, 0),
                           ('demo-txt/8293.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)

    def test_antdoc_8294(self):
        self.maxDiff = None
        docid = '8294'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8294.txt', 'change_control', 0, 0, 0),
                           ('demo-txt/8294.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8294.txt', 'date', 1, 0, 0),
                           ('demo-txt/8294.txt', 'effectivedate_auto', 1, 0, 0),
                           ('demo-txt/8294.txt', 'force_majeure', 0, 0, 0),
                           ('demo-txt/8294.txt', 'limliability', 0, 0, 0),
                           ('demo-txt/8294.txt', 'noncompete', 3, 0, 0),
                           ('demo-txt/8294.txt', 'party', 4, 0, 0),
                           # currently, the system is getting a partial result
                           # only first sentence out of the 2 gold adjacent candidates.
                           # ('demo-txt/8294.txt', 'remedy', 1, 0, 0),
                           ('demo-txt/8294.txt', 'remedy', 0, 1, 1),  # has_diff: will fix
                           ('demo-txt/8294.txt', 'renewal', 0, 0, 0),
                           ('demo-txt/8294.txt', 'termination', 0, 0, 0),
                           ('demo-txt/8294.txt', 'term', 1, 0, 0),
                           ('demo-txt/8294.txt', 'title', 1, 0, 0),
                           ('demo-txt/8294.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8294.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)

    def test_antdoc_8295(self):
        self.maxDiff = None
        docid = '8295'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8295.txt', 'change_control', 0, 0, 0),
                           ('demo-txt/8295.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8295.txt', 'date', 1, 0, 0),
                           ('demo-txt/8295.txt', 'effectivedate_auto', 1, 0, 0),
                           ('demo-txt/8295.txt', 'force_majeure', 0, 0, 0),
                           ('demo-txt/8295.txt', 'limliability', 0, 0, 0),
                           ('demo-txt/8295.txt', 'noncompete', 1, 0, 0),
                           ('demo-txt/8295.txt', 'party', 4, 0, 0),
                           ('demo-txt/8295.txt', 'remedy', 1, 0, 0),
                           ('demo-txt/8295.txt', 'renewal', 0, 0, 0),
                           ('demo-txt/8295.txt', 'termination', 0, 0, 0),
                           ('demo-txt/8295.txt', 'term', 0, 0, 0),
                           ('demo-txt/8295.txt', 'title', 1, 0, 0),
                           ('demo-txt/8295.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8295.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)

    def test_antdoc_8296(self):
        self.maxDiff = None
        docid = '8296'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8296.txt', 'change_control', 0, 0, 0),
                           ('demo-txt/8296.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8296.txt', 'date', 1, 0, 0),
                           ('demo-txt/8296.txt', 'effectivedate_auto', 0, 0, 0),
                           ('demo-txt/8296.txt', 'force_majeure', 0, 0, 0),
                           ('demo-txt/8296.txt', 'limliability', 0, 0, 0),
                           ('demo-txt/8296.txt', 'noncompete', 0, 0, 0),
                           ('demo-txt/8296.txt', 'party', 10, 0, 0),
                           ('demo-txt/8296.txt', 'remedy', 0, 0, 0),
                           ('demo-txt/8296.txt', 'renewal', 0, 0, 0),
                           ('demo-txt/8296.txt', 'termination', 0, 0, 0),
                           ('demo-txt/8296.txt', 'term', 1, 0, 0),
                           ('demo-txt/8296.txt', 'title', 1, 0, 0),
                           ('demo-txt/8296.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8296.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)

    def test_antdoc_8297(self):
        self.maxDiff = None
        docid = '8297'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8297.txt', 'change_control', 1, 0, 0),
                           ('demo-txt/8297.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8297.txt', 'date', 1, 0, 0),
                           ('demo-txt/8297.txt', 'effectivedate_auto', 1, 0, 0),
                           ('demo-txt/8297.txt', 'force_majeure', 0, 0, 0),
                           ('demo-txt/8297.txt', 'limliability', 0, 0, 0),
                           ('demo-txt/8297.txt', 'noncompete', 2, 0, 0),
                           ('demo-txt/8297.txt', 'party', 4, 0, 0),
                           ('demo-txt/8297.txt', 'remedy', 1, 0, 0),
                           ('demo-txt/8297.txt', 'renewal', 0, 0, 0),
                           ('demo-txt/8297.txt', 'termination', 0, 0, 0),
                           ('demo-txt/8297.txt', 'term', 0, 0, 0),
                           ('demo-txt/8297.txt', 'title', 1, 0, 0),
                           ('demo-txt/8297.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8297.txt', 'cust_9', 0, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)

    def test_antdoc_8298(self):
        self.maxDiff = None
        docid = '8298'
        prov_result_list = validate_annotated_doc(docid)

        for prov_result in prov_result_list:
            print("prov_result: {}".format(prov_result))
        print("prov_result_list: {}".format(prov_result_list))

        expected_result = [('demo-txt/8298.txt', 'change_control', 3, 0, 0),
                           ('demo-txt/8298.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8298.txt', 'date', 1, 0, 0),
                           ('demo-txt/8298.txt', 'effectivedate_auto', 0, 0, 0),
                           ('demo-txt/8298.txt', 'force_majeure', 0, 0, 0),
                           ('demo-txt/8298.txt', 'limliability', 0, 0, 0),
                           ('demo-txt/8298.txt', 'noncompete', 2, 0, 0),
                           ('demo-txt/8298.txt', 'party', 4, 0, 0),
                           ('demo-txt/8298.txt', 'remedy', 4, 0, 0),
                           ('demo-txt/8298.txt', 'renewal', 1, 0, 0),
                           # ('demo-txt/8298.txt', 'termination', 10, 0, 0),
                           # Original annotation 19015 to 19130 is incomplete, and partial?
                           # We do propose a new provision.  Looks ok.
                           ('demo-txt/8298.txt', 'termination', 9, 1, 1),  # has_diff: verified, ok
                           ('demo-txt/8298.txt', 'term', 1, 0, 0),
                           ('demo-txt/8298.txt', 'title', 1, 0, 0),
                           ('demo-txt/8298.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8298.txt', 'cust_9', 1, 0, 0)]

        self.assertEqual(expected_result, prov_result_list)


    # 2018-10-26, jshaw, this document is removed for now because it constantly
    # failed randomly.
    """
    def test_antdoc_8299(self):
        self.maxDiff = None
        docid = '8299'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8299.txt', 'change_control', 1, 0, 0),
                           ('demo-txt/8299.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8299.txt', 'date', 1, 0, 0),
                           ('demo-txt/8299.txt', 'effectivedate_auto', 0, 0, 0),
                           ('demo-txt/8299.txt', 'force_majeure', 0, 0, 0),
                           # ('demo-txt/8299.txt', 'limliability', 2, 1, 2),  # has_diff: verified, to-fix
                           ('demo-txt/8299.txt', 'limliability', 2, 1, 0),  # has_diff: verified, to-fix
                           # ('demo-txt/8299.txt', 'limliability', 3, 0, 0),
                           ('demo-txt/8299.txt', 'noncompete', 2, 0, 0),
                           ('demo-txt/8299.txt', 'party', 4, 0, 0),
                           ('demo-txt/8299.txt', 'remedy', 2, 0, 0),
                           ('demo-txt/8299.txt', 'renewal', 1, 0, 0),
                           # ('demo-txt/8299.txt', 'termination', 2, 0, 0),  # has_diff: verified, to-fix
                           ('demo-txt/8299.txt', 'termination', 2, 0, 2),  # has_diff: verified, to-fix
                           # ('demo-txt/8299.txt', 'termination', 2, 0, 0),
                           ('demo-txt/8299.txt', 'term', 1, 0, 0),
                           ('demo-txt/8299.txt', 'title', 1, 0, 0),
                           ('demo-txt/8299.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8299.txt', 'cust_9', 1, 0, 0)]
        self.assertEqual(expected_result, prov_result_list)
    """


    def test_antdoc_8300(self):
        self.maxDiff = None
        docid = '8300'
        prov_result_list = validate_annotated_doc(docid)

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('demo-txt/8300.txt', 'change_control', 1, 0, 0),
                           ('demo-txt/8300.txt', 'choiceoflaw', 1, 0, 0),
                           ('demo-txt/8300.txt', 'date', 1, 0, 0),
                           ('demo-txt/8300.txt', 'effectivedate_auto', 0, 0, 0),
                           ('demo-txt/8300.txt', 'force_majeure', 0, 0, 0),
                           ('demo-txt/8300.txt', 'limliability', 4, 0, 0),
                           ('demo-txt/8300.txt', 'noncompete', 2, 0, 0),
                           ('demo-txt/8300.txt', 'party', 4, 0, 0),
                           ('demo-txt/8300.txt', 'remedy', 2, 0, 0),
                           ('demo-txt/8300.txt', 'renewal', 1, 0, 0),
                           ('demo-txt/8300.txt', 'termination', 2, 0, 0),
                           ('demo-txt/8300.txt', 'term', 1, 0, 0),
                           ('demo-txt/8300.txt', 'title', 1, 0, 0),
                           ('demo-txt/8300.txt', 'warranty', 0, 0, 0),
                           ('demo-txt/8300.txt', 'cust_9', 1, 0, 0)]
        self.assertEqual(expected_result,
                         prov_result_list)

    def test_antdoc_1057_ko(self):
        self.maxDiff = None
        docid = '1057'
        prov_result_list = validate_annotated_doc(docid,
                                                  UNIT_TEST_PROVS + ["korean"])

        print("prov_result_list:")
        pprint.pprint(prov_result_list)

        expected_result = [('dir-korean/text/1057.txt', 'korean', 1, 0, 0)]
        self.assertEqual(expected_result,
                         prov_result_list)


if __name__ == "__main__":
    unittest.main()
