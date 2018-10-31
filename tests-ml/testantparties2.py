#!/usr/bin/env python3

import unittest
import pprint
import copy

from typing import Any, Dict, List, Tuple

from kirke.eblearn import ebrunner

MODEL_DIR = 'dir-scut-model'
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
    provision_set = set([])  # type: Set[str]

    # provision_set = set(['choiceoflaw','change_control', 'indemnify', 'jurisdiction',
    #                      'party', 'warranty', 'termination', 'term']))
    prov_labels_map, _ = EB_RUNNER.annotate_document(file_name,
                                                     provision_set=provision_set,
                                                     work_dir=WORK_DIR,
                                                     doc_lang=doc_lang,
                                                     is_dev_mode=True)

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

    def test_export_train_party_35642(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/35642.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(641, 654, 'VITAMINSPICE.'),
                          (679, 712, 'hereinafter called the “Borrower”'),
                          (754, 787, 'Integrated Capital Partners, Inc.'),
                          (811, 823, 'the “Holder”')])

    def test_export_train_party_37028(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # The result not manually verified
        prov_labels_map = annotate_doc('export-train/37028.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
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

    def test_party_35753(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/35753.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(184, 206, 'Game Face Gaming, Inc.'),
                          (231, 244, 'the "Company"'),
                          (314, 344, 'collectively, the "Purchasers"')])

    def test_party_39871(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/39871.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(6279, 6302, 'AGL Capital Corporation'),
                          (6326, 6339, 'the “Company”'),
                          (6346, 6364, 'AGL Resources Inc.'),
                          # TODO, 06/18/2018
                          # missing, but a little weird
                          # each of the purchasers whose names appear at the end
                          # hereof (each, a “Purchaser” and, collectively, the “Purchasers”)
                          # (6550, 6605, 'each, a “Purchaser” and, collectively, the “Purchasers”')])
                          (6389, 6445, '“Holdings” and together with the Company, the “Obligors”')])

    def test_party_35754(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # promissory note, 2nd sentence
        prov_labels_map = annotate_doc('export-train/35754.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(351, 383, 'Great Essential Investment, Ltd.'),
                          (429, 446, '“Great Essential”'),
                          (449, 487, 'Carlyle Asia Growth Partners III, L.P.'),
                          (559, 565, '“CAGP”'),
                          (571, 599, 'CAGP III Co-Investment, L.P.'),
                          (671, 720, '“CAGP III,” and together with CAGP, the “Holders”'),
                          (726, 760, 'China Recycling Energy Corporation'),
                          (784, 797, 'the “Company”')])

    def test_party_41296(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # TODO, parties are in 'Definitions'
        prov_labels_map = annotate_doc('export-train/41296.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         # missing ARS
                         # pylint: disable=line-too-long
                         # the United States Department of Agriculture, Agricultural       Research  Service.
                         # Viridax Corporation
                         # ("Viridax")
                         [])

        # TODO, the input is in a lined format.  The input is probably
        # a .txt document
        # TODO, 06/18/2018
        # skipping this for now
        # prov_labels_map = annotate_doc('export-train/37103.txt')
        # party_list = get_party_list(prov_labels_map)
        # self.assertEqual(party_list,
        #                 [(137, 152, 'XL VISION, INC.'),
        #                  (179, 190, '"XL Vision"'),
        #                  (196, 226, 'ENHANCED\r\nVISION SYSTEMS, INC.'),
        #                  (253, 258, '"EVS"')])


    def test_party_35667(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/35667.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(183, 222, 'Sanomedics International Holdings, Inc.'),
                          (248, 259, 'the “Maker”'),
                          (266, 280, 'Keith Houlihan'),
                          (315, 323, '"Holder"')])

    def test_party_41207(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/41207.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(196, 213, 'Chuy’s Opco, Inc.'),
                          (239, 249, '“Licensor”'),
                          (256, 276, 'MY/ZP IP Group, Ltd.'),
                          (307, 317, '“Licensee”')])

    def test_party_40228(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/40228.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(207, 227, 'SpectraScience, Inc.'),
                          (333, 349, '“SpectraScience”'),
                          (364, 382, 'PENTAX Europe GmbH'),
                          (489, 502, '“DISTRIBUTOR”')])

        # TODO, not studied
        # pylint: disable=pointless-string-statement
        """
        prov_labels_map = annotate_doc('export-train/35670.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                          [])
        """

    def test_party_39206(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # TODO, somewhat difficult
        # required full parse to get it
        prov_labels_map = annotate_doc('export-train/39206.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(6545, 6567, 'CMS ENERGY CORPORATION'),
                          (6593, 6606, 'the “Company”'),
                          # There is also
                          # 'the financial institutions listed on the signature pages'...
                          # but we don't extract references
                          # TODO, 06/18/2018, missing
                          # (6674, 6740,
                          #  'together with their respective successors and assigns, the “Banks”'),
                          (6746, 6763, 'BARCLAYS BANK PLC'),
                          (6768, 6774, 'Agent.')])

    def test_party_39838(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # Note:
        # party line with itemized parties
        # The system is now using special rules to parse title page.  Result is ok.
        # The normal mechanism to get party line seems ok, but not triggered.
        # It is triggered in ebpostproc, but not used because lineannotator took over.
        # As a result, the evaluation script scores this as 0, but it is OK.
        prov_labels_map = annotate_doc('export-train/39838.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(101, 122, 'DELTA AIR LINES, INC.'),
                          (126, 162, 'U.S. BANK TRUST NATIONAL ASSOCIATION'),
                          (167, 195, 'Class A Pass Through Trustee'),
                          (198, 234, 'U.S. BANK TRUST NATIONAL ASSOCIATION'),
                          (239, 258, 'Subordination Agent'),
                          (261, 291, 'U.S. BANK NATIONAL ASSOCIATION'),
                          (296, 308, 'Escrow Agent'),
                          (317, 353, 'U.S. BANK TRUST NATIONAL ASSOCIATION'),
                          (358, 370, 'Paying Agent')])

    def test_party_39829(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/39829.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(1702, 1726, 'Patrick Industries, Inc.'),
                          (1752, 1765, 'the “Company”'),
                          (1768, 1813, 'Tontine Capital Overseas Master Fund II, L.P.'),
                          (1853, 1862, '“Tontine”'),
                          (1869, 1902, 'Northcreek Mezzanine Fund I, L.P.'),
                          (1951, 2054,
                           '“Northcreek”, and each of Tontine and Northcreek individually, a “Buyer” and collectively, the “Buyers”'),
                          (2097, 2115, '“Collateral Agent”')])

        # TODO, this is hard
        # Backround\nA. B. C.
        # pylint: disable=pointless-string-statement
        """
        prov_labels_map = annotate_doc('export-train/35304.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                          [])
        """

        # TODO, hard
        # "Company" and "Consultant", then "part1", "party2"
        # prov_labels_map = annotate_doc('export-train/35373.txt')

        # TODO, hard
        # prov_labels_map = annotate_doc('export-train/44126.txt')
        # lease between 'landlord and tenant',   landlord: party1 ..adress..,
        # tenant: party2, guarantor:, address

        # TODO, Hard
        # '.  And'  Parties in 2 sentences.
        # pylint: disable=line-too-long
        # 'Forboss  Solar  (ShenZhen)   Co.  Ltd.  And  Shenzhen   Fuwaysun  Technology Company  Limited),'
        # prov_labels_map = annotate_doc('export-train/40266.txt')

        # TODO, later
        # HTML format issues, between\n(1)\nparty1\n(2)\nparty2
        # prov_labels_map = annotate_doc('export-train/38461.txt')

    def test_party_38461(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/38461.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(222, 247, 'IMAGE ENTERTAINMENT, INC.'),
                          (273, 280, '“Image”'),
                          (283, 319, 'IMAGE/MADACY HOME ENTERTAINMENT, LLC'),
                          # pylint: disable=line-too-long
                          (370, 433, 'Image and IMHE, each a “Borrower”, and collectively “Borrowers”'),
                          (469, 499, 'PNC BANK, NATIONAL ASSOCIATION'),
                          (501, 506, '“PNC”')])
                          # TODO, 06/18/2018, seems to be missing
                          # (621, 655, 'PNC, in such capacity, the “Agent”')])

        # TODO, we don't handle 'Warrant' well at all
        # prov_labels_map = annotate_doc('export-train/37310.txt')

    def test_party_40980(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/40980.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(207, 236, 'Universal Display Corporation'),
                          (238, 251, 'the “Company”'),
                          (257, 272, 'Mauro Premutico'),
                          (274, 287, 'the “Grantee”')])


if __name__ == "__main__":
    unittest.main()
