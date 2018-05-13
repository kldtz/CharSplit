#!/usr/bin/env python3

import unittest
import pprint
import copy

from typing import Any, Dict, List, Tuple

from kirke.eblearn import ebrunner
from kirke.ebrules import parties
from kirke.docstruct import partyutils

from kirke.utils import strutils

MODEL_DIR = 'dir-scut-mode'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'

EB_RUNNER = ebrunner.EbRunner(MODEL_DIR,
                              WORK_DIR,
                              CUSTOM_MODEL_DIR)

# TODO, NOT YET HANDLED
# 37320.txt, tabled
# 35457.txt??
# 37352.txt, warrant, no party found
# 44085.txt, lease, but in table format, no party found

def annotate_doc(file_name: str) -> Dict[str, Any]:
    doc_lang = 'en'
    provision_set = set([])
    is_doc_structure = True

    # provision_set = set(['choiceoflaw','change_control', 'indemnify', 'jurisdiction',
    #                      'party', 'warranty', 'termination', 'term']))
    prov_labels_map, _ = EB_RUNNER.annotate_document(file_name,
                                                     provision_set=provision_set,
                                                     work_dir=WORK_DIR,
                                                     doc_lang=doc_lang,
                                                     is_doc_structure=is_doc_structure)

    # because special case of 'effectivdate_auto'
    if prov_labels_map.get('effectivedate'):
        effectivedate_annotations = copy.deepcopy(prov_labels_map.get('effectivedate', []))
        for eff_ant in effectivedate_annotations:
            eff_ant['label'] = 'effectivedate_auto'
        prov_labels_map['effectivedate_auto'] = effectivedate_annotations
        del prov_labels_map['effectivedate']

    pprint.pprint(prov_labels_map)
    return prov_labels_map

def get_party_list(prov_labels_map: Dict) -> List[Tuple[int, int, str]]:
    party_ant_list = prov_labels_map.get('party', [])

    return [(ant['start'],
             ant['end'],
             ant['text']) for ant in party_ant_list]


class TestParties2(unittest.TestCase):

    def test_party(self):

        self.maxDiff = None


    def test_export_train_party(self):

        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/35642.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(641, 654, 'VITAMINSPICE.'),
                           (679, 712, 'hereinafter called the “Borrower”'),
                           (754, 787, 'Integrated Capital Partners, Inc.'),
                           (811, 823, 'the “Holder”')])


        # The result not manually verified
        prov_labels_map = annotate_doc('export-train/37028.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(3761, 3778, 'Big Cedar, L.L.C.'),
                           (3818, 3829, '“Big Cedar”'),
                           (3832, 3846, 'Bass Pro, Inc.'),
                           (3872, 3882, '“Bass Pro”'),
                           (3885, 3915, 'Bass Pro Outdoor World, L.L.C.'),
                           (3955, 3961, '“BPOW”'),
                           (3964, 3996, 'Bass Pro Outdoors Online, L.L.C.'),
                           (4036, 4047, '“BP Online”'),
                           (4050, 4067, 'BPS Catalog, L.P.'),
                           (4101, 4114, '“BPS Catalog”'),
                           (4117, 4144, 'Bass Pro Trademarks, L.L.C.'),
                           (4184, 4199, '“BP Trademarks”'),
                           (4202, 4228, 'World Wide Sportsman, Inc.'),
                           (4260, 4274, '“WW Sportsman”'),
                           (4277, 4304, 'Bass Pro Shops Canada, Inc.'),
                           (4330, 4342, '“BPS Canada”'),
                           (4345, 4382, 'Bass Pro Shops Canada (Calgary), Inc.'),
                           (4406, 4426, '“BPS Canada Calgary”'),
                           (4429, 4438, 'BPIP, LLC'),
                           (4479, 4485, '“BPIP”'),
                           (4488, 4510, 'Tracker Marine, L.L.C.'),
                           (4550, 4566, '“Tracker Marine”'),
                           (4569, 4604, 'Bluegreen Vacations Unlimited, Inc.'),
                           (4629, 4640, '“Bluegreen”'),
                           (4647, 4681, 'Bluegreen/Big Cedar Vacations, LLC'),
                           (4721, 4734, 'the “Company”')])

        prov_labels_map = annotate_doc('export-train/35753.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(184, 206, 'Game Face Gaming, Inc.'),
                            (231, 244, 'the "Company"'),
                            (314, 344, 'collectively, the "Purchasers"')])

        prov_labels_map = annotate_doc('export-train/39871.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(6279, 6302, 'AGL Capital Corporation'),
                           (6326, 6339, 'the “Company”'),
                           (6346, 6364, 'AGL Resources Inc.'),
                           (6550, 6605, 'each, a “Purchaser” and, collectively, the “Purchasers”')]
                          # TODO
                          # missing, but a little weird
                          # each of the purchasers whose names appear at the end hereof (each, a “Purchaser” and, collectively, the “Purchasers”)
        )

        # promissory note, 2nd sentence
        prov_labels_map = annotate_doc('export-train/35754.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(351, 383, 'Great Essential Investment, Ltd.'),
                            (429, 446, '“Great Essential”'),
                            (449, 487, 'Carlyle Asia Growth Partners III, L.P.'),
                            (559, 565, '“CAGP”'),
                            (571, 599, 'CAGP III Co-Investment, L.P.'),
                            (671, 720, '“CAGP III,” and together with CAGP, the “Holders”'),
                            (726, 760, 'China Recycling Energy Corporation'),
                            (784, 797, 'the “Company”')])

        # TODO, parties are in 'Definitions'
        prov_labels_map = annotate_doc('export-train/41296.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          # missing ARS
                          # the United States Department of Agriculture, Agricultural       Research  Service.
                          # Viridax Corporation
                          # ("Viridax")
                          [])

        # TODO, the input is in a lined format.  The input is probably
        # a .txt document
        prov_labels_map = annotate_doc('export-train/37103.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          # missing all parties
                          [])

        prov_labels_map = annotate_doc('export-train/35667.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(183, 222, 'Sanomedics International Holdings, Inc.'),
                           (248, 259, 'the “Maker”'),
                           (266, 280, 'Keith Houlihan'),
                           (315, 323, '"Holder"')])

        prov_labels_map = annotate_doc('export-train/41207.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(196, 213, 'Chuy’s Opco, Inc.'),
                            (239, 249, '“Licensor”'),
                            (256, 276, 'MY/ZP IP Group, Ltd.'),
                            (307, 317, '“Licensee”')])
                          
        prov_labels_map = annotate_doc('export-train/40228.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,                          
                          [(207, 227, 'SpectraScience, Inc.'),
                           (333, 349, '“SpectraScience”'),
                           (364, 382, 'PENTAX Europe GmbH'),
                           (489, 502, '“DISTRIBUTOR”')])

        # TODO, not studied
        """
        prov_labels_map = annotate_doc('export-train/35670.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [])
        """
        
        

if __name__ == "__main__":
    unittest.main()
