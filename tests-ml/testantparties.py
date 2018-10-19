#!/usr/bin/env python3

import unittest
# import pprint
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

    # pprint.pprint(prov_labels_map)
    return prov_labels_map

def get_party_list(prov_labels_map: Dict) -> List[Tuple[int, int, str]]:
    party_ant_list = prov_labels_map.get('party', [])

    return [(ant['start'],
             ant['end'],
             ant['text']) for ant in party_ant_list]


class TestParties(unittest.TestCase):

    # pylint: disable=too-many-statements
    def test_party_1(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc1.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(6617, 6636, 'LSH Auto UK Limited'),
                          (6736, 6750, 'the “Borrower"'),
                          (6763, 6776, 'HSBC BANK PLC'),
                          (6851, 6863, 'the “Lender"')])

    def test_party_2(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc2.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(1884, 1904, 'Volkswagen Bank GmbH'),
                          (2053, 2067, 'the "Borrower"'),
                          (2077, 2090, 'HSBC Bank pic'),
                          (2216, 2228, 'the "Lender"')])

    def test_party_3(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc3.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(7967, 7981, 'Tetra Pak Ltda'),
                          (8273, 8285, 'the Borrower'),
                          (8298, 8311, 'HSBC Bank pic'),
                          (8413, 8423, 'the Lender'),
                          # not sure which one is more correct
                          # (8436, 8473, 'HSBC Bank Brasil S.A - Banco Multiple'),
                          (8436, 8456, 'HSBC Bank Brasil S.A'),
                          (8683, 8694, 'HSBC Brazil')])

    def test_party_4(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc4.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(59, 92, 'STAGECOACH TRANSPORT HOLDINGS PLC'),
                          # TODO, this would be better
                          # (162, 182, 'the initial Borrower'),
                          (162, 187, 'the initial Borrower; and'),
                          (194, 207, 'HSBC BANK PLC'),
                          (247, 251, 'Bank')])

    def test_party_5(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc5.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(8683, 8693, 'AEGON N.V.'),
                          (8707, 8721, 'the "Borrower"'),
                          (8731, 8765, 'BANC OF AMERICA SECURITIES LIMITED'),
                          (8770, 8802, 'CITIGROUP GLOBAL MARKETS LIMITED'),
                          (8821, 8839, 'the "Coordinators"'),
                          (8849, 8867, 'ABN AMRO BANK N.V.'),
                          (8869, 8903, 'BANC OF AMERICA SECURITIES LIMITED'),
                          (8905, 8921, 'BARCLAYS CAPITAL'),
                          (8924, 8956, 'CITIGROUP GLOBAL MARKETS LIMITED'),
                          (8958, 9011, 'COÖPERATIEVE CENTRALE RAIFFEISEN- BOERENLEENBANK B.A.'),
                          (9013, 9029, 'DEUTSCHE BANK AG'),
                          # (9031, 9044, 'LONDON BRANCH'),  # maybe remove in future
                          (9046, 9059, 'HSBC BANK PLC'),
                          (9061, 9075, 'ING  BANK N.V.'),
                          (9077, 9101, 'JPMORGAN CHASE BANK N.A.'),
                          (9103, 9145, 'MORGAN STANLEY BANK INTERNATIONAL  LIMITED'),
                          (9147, 9167, 'ROYAL BANK OF CANADA'),
                          (9172, 9202, 'THE ROYAL BANK OF SCOTLAND PLC'),
                          (9232, 9261, 'the "Mandated Lead Arrangers"'),
                          (9271, 9282, 'BNP PARIBAS'),
                          (9284, 9306, 'GOLDMAN SACHS BANK USA'),
                          (9311, 9332, 'SOCIÉTÉ GÉNÉRALE S.A.'),
                          (9390, 9410, 'the “Lead Arrangers”'),
                          (9421, 9439, 'ABN AMRO BANK N.V.'),
                          (9441, 9475, 'BANC OF AMERICA SECURITIES LIMITED'),
                          (9477, 9493, 'BARCLAYS CAPITAL'),
                          (9496, 9528, 'CITIGROUP GLOBAL MARKETS LIMITED'),
                          (9530, 9546, 'DEUTSCHE BANK AG'),
                          (9548, 9561, 'ING BANK N.V.'),
                          (9564, 9588, 'JPMORGAN CHASE BANK N.A.'),
                          (9590, 9604, 'MORGAN STANLEY'),
                          (9609, 9629, 'ROYAL BANK OF CANADA'),
                          (9647, 9664, 'the "Bookrunners"'),
                          (9674, 9700, 'THE FINANCIAL INSTITUTIONS'),
                          (9909, 9941, 'together, the "Original Lenders"'),
                          (9951, 9977, 'THE FINANCIAL INSTITUTIONS'),
                          (10028, 10057, 'the "Original  Issuing Banks"'),
                          (10067, 10101, 'BANC OF AMERICA SECURITIES LIMITED'),
                          (10141, 10153, 'the  "Agent"'),
                          (10164, 10185, 'BANK OF AMERICA, N.A.'),
                          (10216, 10247, 'the "US Dollar Swingline Agent"'),
                          (10259, 10293, 'BANC OF AMERICA SECURITIES LIMITED'),
                          (10319, 10346, 'the "Euro Swingline  Agent"'),
                          (10361, 10382, 'BANK OF AMERICA, N.A.'),
                          (10426, 10454, 'the "Fronting Issuing  Bank"')])

    def test_party_6(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc6.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(8427, 8438, 'AMEY UK PLC'),
                          (8475, 8488, 'the ’Company"'),
                          (8497, 8513, 'THE SUBSIDIARIES'),
                          (8611, 8636, 'the "Original Guarantors’'),
                          (8649, 8662, 'HSBC BANK PLC'),
                          (8674, 8686, 'the "Lender”')])

    def test_party_7(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc7.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(2727, 2736, 'Aviva pic'),
                          (2859, 2866, '"Aviva"'),
                          (2874, 2911, 'Aviva International Insurance Limited'),
                          (3035, 3059, 'Aviva Insurance  Limited'),
                          (3163, 3187, 'together “the Guarantor”'),
                          (3195, 3208, 'HSBC Bank pic'),
                          (3357, 3367, '"the Bank”')])

    def test_party_8(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc8.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(8380, 8405, 'LIBERTY INTERNATIONAL PLC'),
                          (8452, 8466, 'the "Borrower"'),
                          (8479, 8492, 'HSBC BANK PLC'),
                          (8494, 8506, 'the "Lender"')])

    def test_party_9(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc9.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(1897, 1914, 'HATTA ONE LIMITED'),
                          (2142, 2152, 'the Lessor'),
                          (2161, 2174, 'HSBC BANK PLC'),
                          (2291,
                           2364,
                           # pylint: disable=line-too-long
                           'together with its permitted assignees and  transferees, called the Lender'),
                          (2373, 2386, 'HSBC BANK PLC'),
                          (2529, 2534, 'Agent'),
                          (2547, 2560, 'HSBC BANK PLC'),
                          (2713, 2727, 'Security Agent')])

    def test_party_10(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc10.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(89, 131, 'Volkswagen Financial Services (UK) Limited'),
                          (135, 145, '“Borrower”'),
                          (155, 168, 'HSBC Bank pic'),
                          (172, 181, '“Lender”.')])


    def test_party_11(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc11.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(7462, 7474, 'DELOITTE LLP'),
                          (7576, 7590, 'the "Borrower"'),
                          (7599, 7619, 'DELOITTE MCS LIMITED'),
                          (7699, 7723, 'the "Original Guarantor"'),
                          (7736, 7749, 'HSBC BANK PLC'),
                          (7751, 7763, 'the "Lender"')])


    def test_party_12(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc12.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(2237, 2288, 'Johnson & Johnson Medikal Sanayi Ve Ticaret Limited'),
                          (2359, 2373, 'the “Borrower”'),
                          (2382, 2395, 'HSBC BANK PLC'),
                          (2434, 2444, 'the “Bank”')])


    def test_party_13(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc13.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(529, 556, 'Gulf Agency Company Limited'),
                          (608, 622, 'the “Borrower”'),
                          (632, 645, 'HSBC BANK PLC'),
                          (683, 693, 'the “Bank”')])

    def test_party_14(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc14.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(85, 98, 'UNILEVER N.V.'),
                          (134, 146, 'UNILEVER PLC'),
                          (148, 184, 'UNILEVER  FINANCE INTERNATIONAL B.V.'),
                          (224, 253, 'UNILEVER  CAPITAL CORPORATION'),
                          (268, 315, 'each a “Borrower” and together the  “Borrowers”'),
                          (325, 338, 'UNILEVER N.V.'),
                          (340, 352, 'UNILEVER PLC'),
                          (357, 385, 'UNILEVER UNITED STATES, INC.'),
                          (402, 450, 'each a “Guarantor” and together the “Guarantors”'),
                          (460, 473, 'HSBC BANK PLC'),
                          (485, 497, 'the “Lender”'),
                          (511, 524, 'HSBC BANK PLC'),
                          (558, 593, 'the “U.S. Dollar Swingline  Lender”')])

    def test_party_15(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc15.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(1997, 2036, 'CHINA MERCHANTS SECURITIES (UK) LIMITED'),
                          (2146, 2162, 'the  ‘ Borrower"'),
                          (2175, 2188, 'HSBC BANK PLC'),
                          (2225, 2235, 'the “Bank”')])


    def test_party_16(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc16.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(74, 93, 'GLAXOSMITHKLINE pic'),
                          (95, 109, 'the "Borrower"'),
                          (124, 137, 'HSBC BANK PLC'),
                          (139, 149, 'the "Bank"')])

    def test_party_100(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # box
        prov_labels_map = annotate_doc('dir-test-doc/doc100.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [# (213, 216, 'Box'),
                             (224, 238, 'Documents Inc.'),
                             (266, 275, '”Partner”'),
                             (367, 382, 'Box and Partner'),
                             (419, 459, 'a “Party” and together as the “Parties”.')])

    def test_party_101(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc101.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(158, 186, 'CUSTOMER THREE HOLDINGS B.V.'),
                          (361, 365, '"C3"'),
                          (483, 499, 'Box.com (UK) Ltd'),
                          (601, 607, '"Box "')])


    def test_party_102(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc102.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(306, 320, 'Partner 4, LLC'),
                          (480, 498, 'collectively, “P4”'),
                          (505, 514, 'Box, Inc.'),
                          (677, 701, 'collectively, “Supplier”')])

    def test_party_103(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc103.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(221, 242, 'ContentX Technologies'),
                          (352, 362, '“ContentX”'),
                          (368, 405, 'Cybeitnesh International  Corporation'),
                          # pylint: disable=line-too-long
                          (508, 575, 'Cybermesh and together with ContentX, the “Members" each a “Member”')])

    def test_party_104(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc104.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(118, 172, 'Hadasit Medical Research  Services and Development Ltd'),
                          (229, 238, '“Hadasit”'),
                          (244, 273, 'Cell  Cure Neurosctcnccs Ltd.'),
                          (329, 342, 'the “Company”')])

    def test_party_105(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # uses dict/parties/location.list to remove 'Wales'
        # 'Wales' is originally an issue due to and_org_index,
        # which captured 'Wales' as a separate party
        prov_labels_map = annotate_doc('dir-test-doc/doc105.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(296, 312, 'Box.com (UK) Ltd')])

    def test_party_106(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc106.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(185, 211, 'LipimetiX Development, LLC'),
                          (252, 265, 'the "Company"'),
                          (271, 298, 'Capstone Therapeutics Corp.'),
                          (325, 336, '"Capstone11')])

    def test_party_107(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # missing 'Customer One, LLC___..., TX 99223               (jointly' at end of
        # page.  OCR issue.
        prov_labels_map = annotate_doc('dir-test-doc/doc107.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(279, 288, 'Box, Inc.'),
                          (380, 393, '“Participant”')])

    def test_party_108(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc108.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(3055, 3076, 'TriLinc Advisors, LLC'),
                          (3121, 3152, 'TriLinc Global Impact Fund, LLC')])

    def test_party_109(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc109.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(157, 176, 'BIOPURE CORPORATION'),
                          (276, 284, '“Seller”'),
                          (291, 308, 'SPEAR REALTY, LLC'),
                          (435, 446, '“Purchaser”')])

    #def test_party_110(self):
    #    # pylint: disable=invalid-name
    #    self.maxDiff = None
    #
        # TODO, 06/18/2018
        # this is not a contract, a letter
        # prov_labels_map = annotate_doc('dir-test-doc/doc110.txt')
        # party_list = get_party_list(prov_labels_map)
        # self.assertEqual(party_list,
        #                  [(3569, 3606, 'Prudential Investment Management, Inc'),
        #                   (3930, 3950, 'LTC Properties, Inc.'),
        #                   (3976, 3989, 'the “Company”')])

    def test_party_111(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc111.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(126, 145, 'Arrayit Corporation'),
                          (170, 179, '“Arrayit”'),
                          (182, 209, 'Ovarian Cancer Testing, LLP'),
                          (260, 277, 'the “Partnership”'),
                          (283, 302, 'Arrayit Diagnostics'),
                          (328, 341, 'the “Company”'),
                          (398, 465,
                           'individually, a  “Royalty holder” or collectively “Royalty holders”')])

    # def test_party_112(self):
    #     # pylint: disable=invalid-name
    #    self.maxDiff = None
    #
        # TODO
        # This is problematic due to each line is a separate paragraph
        # Need to fix paragraph understanding part in pdftxtparser
        # pylint: disable=pointless-string-statement
        """
        prov_labels_map = annotate_doc('dir-test-doc/doc112.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                          [])
        """

    def test_party_113(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc113.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(244, 253, 'Box, Inc.'),
                          (274, 279, '“Box”'),
                          (375, 388, '“Participant”')])

    def test_party_114(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc114.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(197, 214, 'ADVANTENNIS CORP.'),
                          (241, 254, '“ADVANTENNIS”'),
                          (260, 292, 'WORLD TEAMTENNIS FRANCHISE, INC.'),
                          (320, 326, '“WTTF”')])

    def test_party_115(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc115.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(117, 134, 'Johanna Templeton'),
                          (136, 144, '"Tenant"'),
                          (151, 165, 'Ravneet Uberoi'),
                          (167, 178, '"Subtenant"')])

    def test_party_116(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc116.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(879, 928, 'Oriental Intra-Asia Entertainment (China) Limited'),
                          (931, 941, '“Oriental”'),
                          (944, 986, 'China TransInfo Technology Group Co., Ltd.'),
                          (988, 1003, '“Group Company”'),
                          (1006, 1055, 'Beijing PKU Chinafront  High Technology Co., Ltd.'),
                          (1057, 1062, '“PKU”'),
                          # pylint: disable=line-too-long
                          (1065, 1123, 'Beijing Tian Hao Ding Xin Science and Technology Co., Ltd.'),
                          (1126, 1143, '“Bejing Tian Hao”'),
                          (1146, 1192, 'Beijing Zhangcheng Culture and Media Co., Ltd.'),
                          (1194, 1214, '“Zhangcheng Culture”'),
                          (1217, 1268, 'Bejing  Zhangcheng Science and Technology Co., Ltd.'),
                          (1270, 1290, '“Zhangcheng Science”'),
                          (1293, 1344, 'China TranWiseway Information  Technology Co., Ltd.'),
                          (1367, 1419, 'Xinjiang Zhangcheng Science and Technology Co., Ltd.'),
                          (1422, 1443, '“Xinjiang Zhangcheng”'),
                          (1446, 1497, 'Dalian Dajian Zhitong Information Service Co., Ltd.'),
                          (1499, 1514, '“Dalian Dajian”'),
                          (1521, 1568, 'Shanghai Yootu Information Technology Co., Ltd.'),
                          (1570,
                           1761,
                           # pylint: disable=line-too-long
                           '“Shanghai Yootu” and together with Group Company,  PKU, Beijing Tian Hao, '
                           'Zhangcheng Culture, Zhangcheng Science, China TranWiseway, Xinjiang  '
                           'Zhangcheng and Dalian Dajian, the “VIE Entities”')])

    def test_party_117(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc117.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(250, 280, 'Apollo Global Management,  LLC'),
                          (320, 333, 'the “Company"'),
                          (340, 353, 'Leon D. Black'),
                          (355, 366, '“Executive"')])

    def test_party_118(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc118.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(102, 122, 'SHBV (Hong Kong) Ltd'),
                          (125, 131, '“SHBV”'),
                          (266, 284, 'WASTE2ENERGY GROUP'),
                          (454, 461, '“W2EGH”'),
                          (464, 496, 'WASTE2ENERGY ENGINEERING LIMITED'),
                          (659, 665, '“W2EE”'),
                          (671, 719, 'WASTE2ENERGY  TECHNOLOGIES INTERNATIONAL LIMITED'),
                          (882, 889, '“W2ETI”')])

    def test_party_119(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # This is not a contract.
        # The full list of party name is not verified.
        prov_labels_map = annotate_doc('dir-test-doc/doc119.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(796, 805, 'UDR, Inc.'),
                          (875, 906, 'Banc of  America Securities LLC'),
                          (908, 937, 'Citigroup Global Markets Inc.'),
                          (939, 968, 'Deutsche Bank Securities Inc.'),
                          (970, 998, 'J.P. Morgan  Securities Inc.'),
                          (1000, 1013, 'Merrill Lynch'),
                          (1015, 1050, 'Pierce, Fenner & Smith Incorporated'),
                          (1052, 1086, 'Morgan Stanley & Co.  Incorporated'),
                          (831, 844, 'the “Company”'),
                          (1091, 1118, 'Wells Fargo Securities, LLC'),
                          (1120, 1164, 'each, an “Agent,” and together, the “Agents”')])

    def test_party_120(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc120.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(183, 205, 'Lincolnway Energy, LLC'),
                          (207, 217, '"Producer"'),
                          (260, 288, 'Green Plains Trade Group LLC'),
                          (328, 334, '"GPTG"')])

    def test_party_121(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc121.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(140, 161, 'NEW YORK AIRCAM CORF.'),
                          (258, 266, '“Lessor”'),
                          (273, 292, 'CSC TRANSPORT, INC.'),
                          (399, 407, '“Lessee”')])

    def test_party_122(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # very similar to doc113.txt, but slightly different
        prov_labels_map = annotate_doc('dir-test-doc/doc122.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(217, 226, 'Box, Inc.'),
                          (247, 252, '“Box”'),
                          (348, 361, '“Participant”')])

    def test_party_123(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # TODO, missing
        prov_labels_map = annotate_doc('dir-test-doc/doc123.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(159, 183, 'Fidelity Funding Company'),
                          (207, 217, '“Landlord”'),
                          (225, 244, 'Extend Health, Inc.'),
                          (270, 278, '“Tenant”'),
                          # missing "Landlord and Tenant"
                          (341, 386, 'the “Parties” and individually, as a “Party.”')])

    def test_party_130(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # these are test documents
        # TODO, missing
        prov_labels_map = annotate_doc('dir-test-doc/doc130.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(167, 187, 'THE BOARD OF REGENTS'),
                          # pylint: disable=line-too-long
                          # THE BOARD OF REGENTS ("BOARD") of THE UNIVERSITY OF TEXAS  SYSTEM ("SYSTEM")
                          # missing ("BOARD")
                          (201, 232, 'THE UNIVERSITY OF TEXAS  SYSTEM'),
                          (351, 403, 'THE UNIVERSITY OF TEXAS M. D. ANDERSON CANCER CENTER'),
                          (234, 242, '"SYSTEM"'),
                          # TODO, 06/18/2018, missing?
                          # (406, 415, '"UTMDACC"'),
                          (457, 487, 'SIGNPATH PHARMACEUTICALS, INC.'),
                          (612, 622, '"LICENSEE"')])


    def test_party_131(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc131.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(244, 253, 'Box, Inc.'),
                          (274, 279, '“Box”'),
                          (375, 388, '“Participant”')])

    def test_party_133(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # doc132 is the same as doc131.txt

        prov_labels_map = annotate_doc('dir-test-doc/doc133.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(158, 177, 'Comverse Technology'),
                          (251, 264, 'the “Company”'),
                          (270, 280, 'Joel Legon'),
                          (283, 293, '“Employee”')])

    #def test_party_134(self):
    #    # pylint: disable=invalid-name
    #    self.maxDiff = None
    #
        # TODO, failed
        # pylint: disable=pointless-string-statement
        """
        prov_labels_map = annotate_doc('dir-test-doc/doc134.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                          [])
        """

    def test_party_135(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc135.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(291, 300, 'Box, Inc.'),
                          (321, 326, '"Box"'),
                          (390, 409, 'Customer  Six Corp.'),
                          (411, 424, '"Participant"')])

    def test_party_136(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc136.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(166, 188, 'Trio Resources AG Inc.'),
                          (243, 300, 'with its affiliated entities, collectively, the “Company”'),
                          (306, 331, 'Seagel Investment  Corp..'),
                          (381, 397, 'the “Consultant”')])


    def test_party_137(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # TODO, "hereinafter defined' is wrong
        prov_labels_map = annotate_doc('dir-test-doc/doc137.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(13343, 13351, 'DSW INC.'),
                          (13374, 13379, '“DSW”'),
                          (13382, 13407, 'DSW SHOE  WAREHOUSE, INC.'),
                          (13433, 13562,
                           '“DSW Shoe”, and together with DSW, individually, a  “Borrower”, and '
                           'collectively, the “Borrowers”, as hereinafter further defined'),
                          (13657, 13688, 'PNC BANK,  NATIONAL ASSOCIATION'),
                          (13813, 13839, 'the “Administrative Agent”')])


    def test_party_138(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('dir-test-doc/doc138.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(188, 202, 'CRG Finance AG'),
                          (205, 210, '“CRG”'),
                          (217, 238, 'Ardent  Mines Limited'),
                          (240, 253, 'the “Company”')])


    def test_party_kodak1(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # unknown source
        prov_labels_map = annotate_doc('dir-test-doc/kodak1.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(104, 131, 'Kodak (Australasia) Pty Ltd'),
                          (209, 214, 'Kodak'),
                          (225, 249, 'Printcraft (QLD) Pty Ltd'),
                          (333, 341, 'Customer')])


    def test_export_train_party_39811(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/39811.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(235, 261, 'The Princeton Review, Inc.'),
                          (263, 275, 'the “Issuer”'),
                          (282, 302, 'the Collateral Agent'),
                          (307, 327, 'the Purchasers party'),
                          (343, 363, 'the Guarantors party')])


    def test_export_train_party_44090(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/44090.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(80, 100, 'LORRAINE SALCICCIOLI'),
                          (180, 194, 'the “Landlord”'),
                          (206, 230, 'PRINTING COMPONENTS INC.'),
                          (332, 344, 'the “Tenant”')])

    def test_export_train_party_39074(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/39074.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(6316, 6328, 'HASBRO, INC.'),
                          (6358, 6371, 'the “Company”'),
                          (6374, 6383, 'HASBRO SA'),
                          (6635, 6691, 'collectively, the “Lenders” and individually, a “Lender”'),
                          (6698, 6719, 'BANK OF AMERICA, N.A.'),
                          (6724, 6779, 'Administrative Agent, Swing Line Lender and L/C Issuer.')])


    def test_export_train_party_40331(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/40331.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(4330, 4341, 'ante4, Inc.'),
                          (4367, 4375, '“Parent”'),
                          (4382, 4393, 'ante5, Inc.'),
                          (4476, 4488, '“Subsidiary”')])

    def test_export_train_party_41305(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/41305.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(204, 242, 'Holly Refining & Marketing — Tulsa LLC'),
                          (269, 285, '“Tulsa Refining”'),
                          (292, 305, 'HEP Tulsa LLC'),
                          (345, 356, '“HEP Tulsa”')])

        # disabled 28.1
        # pylint: disable=pointless-string-statement
        """
        prov_labels_map = annotate_doc('export-train/44102.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                          [(120, 141, 'SCI LA DEFENSE ASTORG'),
                            (944, 953, '“Lessor”;'),
                            (999, 1021, 'SEQUANS COMMUNICATIONS'),
                            (1323, 1332, '“Lessee”;')])
        """

    def test_export_train_party_39749(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        prov_labels_map = annotate_doc('export-train/39749.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(104, 120, 'US AIRWAYS, INC.'),
                          (124, 148, 'WILMINGTON TRUST COMPANY'),
                          (224, 248, 'WILMINGTON TRUST COMPANY'),
                          (253, 272, 'Subordination Agent'),
                          (275, 323, 'WELLS FARGO BANK NORTHWEST, NATIONAL ASSOCIATION'),
                          (328, 340, 'Escrow Agent'),
                          (349, 373, 'WILMINGTON TRUST COMPANY'),
                          (378, 390, 'Paying Agent')])

    def test_export_train_party_35814(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # This one has 'Corporation.,'
        # Not a real contract.
        prov_labels_map = annotate_doc('export-train/35814.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(115, 134, 'Kranem Corporation.'),
                          (162, 175, 'the “Company”'),
                          (197, 205, 'Investco'),
                          (244, 256, 'the “Holder”')])


    def test_export_train_party_36039(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        # 'of' org is ignored
        prov_labels_map = annotate_doc('export-train/36039.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEqual(party_list,
                         [(173, 204, 'DUKE REALTY LIMITED PARTNERSHIP'),
                          # TODO,
                          # it would be more correct "Duke Reality of Indiana Limited Partnership"
                          (288, 315, 'Indiana Limited Partnership'),
                          (317, 327, '“Landlord”'),
                          (334, 348, 'SCIQUEST, INC.'),
                          (374, 382, '“Tenant”'),
                          (410, 438, 'Kroy Building Products, Inc.'),
                          (464, 470, '“Kroy”')])


if __name__ == "__main__":
    unittest.main()
