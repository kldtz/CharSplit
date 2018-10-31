#!/usr/bin/env python3

import copy
import json
import pprint
import unittest
from typing import Any, Dict, List, Tuple

from kirke.eblearn import ebrunner
from kirke.client import postfileutils


MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'

EB_RUNNER = ebrunner.EbRunner(MODEL_DIR,
                              WORK_DIR,
                              CUSTOM_MODEL_DIR)


def annotate_doc(file_name: str) -> Dict[str, Any]:
    doc_lang = 'en'
    provision_set = set([])  # type: Set[str]

    # provision_set = set(['choiceoflaw','change_control', 'indemnify', 'jurisdiction',
    #                      'party', 'warranty', 'termination', 'term']))
    prov_labels_map, _ = EB_RUNNER.annotate_document(file_name,
                                                     provision_set=provision_set,
                                                     work_dir=WORK_DIR,
                                                     doc_lang=doc_lang)

    # because special case of 'effectivdate_auto'
    if prov_labels_map.get('effectivedate'):
        effectivedate_annotations = copy.deepcopy(prov_labels_map.get('effectivedate', []))
        for eff_ant in effectivedate_annotations:
            eff_ant['label'] = 'effectivedate_auto'
            prov_labels_map['effectivedate_auto'] = effectivedate_annotations
            del prov_labels_map['effectivedate']

    pprint.pprint(prov_labels_map)
    return prov_labels_map


def upload_annotate_doc(file_name: str) -> Dict[str, Any]:

    text = postfileutils.post_unittest_annotate_document(file_name)

    ajson = json.loads(text)

    return ajson


def get_party_list(prov_labels_map: Dict) -> List[Tuple[int, int, str]]:
    party_ant_list = prov_labels_map.get('party', [])

    return [(ant['start'],
             ant['end'],
             ant['text']) for ant in party_ant_list]


class TestParties3(unittest.TestCase):

    def test_party(self):
        # pylint: disable=invalid-name
        self.maxDiff = None


    def test_export_train_party_40980(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # this is a place holder
        # This test is already performed in testantparties2.py
        prov_labels_map = annotate_doc('export-train/40980.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(207, 236, 'Universal Display Corporation'),
                          (238, 251, 'the “Company”'),
                          (257, 272, 'Mauro Premutico'),
                          (274, 287, 'the “Grantee”')])


if __name__ == "__main__":
    unittest.main()
