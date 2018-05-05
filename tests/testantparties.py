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


class TestParties(unittest.TestCase):

    def test_party(self):

        self.maxDiff = None

        prov_labels_map = annotate_doc('mytest/doc1.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(6617, 6636, 'LSH Auto UK Limited'),
                           (6736, 6750, 'the “Borrower"'),
                           (6763, 6776, 'HSBC BANK PLC'),
                           (6851, 6863, 'the “Lender"')])

        prov_labels_map = annotate_doc('mytest/doc2.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(1884, 1904, 'Volkswagen Bank GmbH'),
                           (2053, 2067, 'the "Borrower"'),
                           (2077, 2090, 'HSBC Bank pic'),
                           (2216, 2228, 'the "Lender"')])

        prov_labels_map = annotate_doc('mytest/doc3.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(7967, 7981, 'Tetra Pak Ltda'),
                           (8273, 8285, 'the Borrower'),
                           (8298, 8311, 'HSBC Bank pic'),
                           (8413, 8423, 'the Lender'),
                           (8436, 8473, 'HSBC Bank Brasil S.A - Banco Multiple'),
                           (8683, 8694, 'HSBC Brazil')])

        prov_labels_map = annotate_doc('mytest/doc4.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(59, 92, 'STAGECOACH TRANSPORT HOLDINGS PLC'),
                           (162, 182, 'the initial Borrower'),
                           (194, 207, 'HSBC BANK PLC'),
                           (247, 251, 'Bank')])

        prov_labels_map = annotate_doc('mytest/doc5.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
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

        prov_labels_map = annotate_doc('mytest/doc6.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(8427, 8438, 'AMEY UK PLC'),
                           (8475, 8488, 'the ’Company"'),
                           (8497, 8513, 'THE SUBSIDIARIES'),
                           (8611, 8636, 'the "Original Guarantors’'),
                           (8649, 8662, 'HSBC BANK PLC'),
                           (8674, 8686, 'the "Lender”')])

        prov_labels_map = annotate_doc('mytest/doc7.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(2727, 2736, 'Aviva pic'),
                           (2859, 2866, '"Aviva"'),
                           (2874, 2911, 'Aviva International Insurance Limited'),
                           (3035, 3059, 'Aviva Insurance  Limited'),
                           (3163, 3187, 'together “the Guarantor”'),
                           (3195, 3208, 'HSBC Bank pic'),
                           (3357, 3367, '"the Bank”')])

        prov_labels_map = annotate_doc('mytest/doc8.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(8380, 8405, 'LIBERTY INTERNATIONAL PLC'),
                           (8452, 8466, 'the "Borrower"'),
                           (8479, 8492, 'HSBC BANK PLC'),
                           (8494, 8506, 'the "Lender"')])

        prov_labels_map = annotate_doc('mytest/doc9.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(1897, 1914, 'HATTA ONE LIMITED'),
                           (2142, 2152, 'the Lessor'),
                           (2161, 2174, 'HSBC BANK PLC'),
                           (2291,
                            2364,
                            'together with its permitted assignees and  transferees, called the Lender'),
                           (2373, 2386, 'HSBC BANK PLC'),
                           (2526, 2534, 'as Agent'),
                           (2547, 2560, 'HSBC BANK PLC'),
                           (2710, 2727, 'as Security Agent')])

        prov_labels_map = annotate_doc('mytest/doc10.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(89, 131, 'Volkswagen Financial Services (UK) Limited'),
                           (135, 145, '“Borrower”'),
                           (155, 168, 'HSBC Bank pic'),
                           (172, 181, '“Lender”.')])

        prov_labels_map = annotate_doc('mytest/doc11.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(7462, 7474, 'DELOITTE LLP'),
                           (7576, 7590, 'the "Borrower"'),
                           (7599, 7619, 'DELOITTE MCS LIMITED'),
                           (7699, 7723, 'the "Original Guarantor"'),
                           (7736, 7749, 'HSBC BANK PLC'),
                           (7751, 7763, 'the "Lender"')])


        prov_labels_map = annotate_doc('mytest/doc12.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                           [(2237, 2288, 'Johnson & Johnson Medikal Sanayi Ve Ticaret Limited'),
                            (2359, 2373, 'the “Borrower”'),
                            (2382, 2395, 'HSBC BANK PLC'),
                            (2434, 2444, 'the “Bank”')])

        prov_labels_map = annotate_doc('mytest/doc13.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(529, 556, 'Gulf Agency Company Limited'),
                           (608, 622, 'the “Borrower”'),
                           (632, 645, 'HSBC BANK PLC'),
                           (683, 693, 'the “Bank”')])

        prov_labels_map = annotate_doc('mytest/doc14.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(85, 98, 'UNILEVER N.V.'),
                           # missing 'unilever plc'
                           # missing unilever finance international B.V.'
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

        prov_labels_map = annotate_doc('mytest/doc15.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(1997, 2036, 'CHINA MERCHANTS SECURITIES (UK) LIMITED'),
                           (2146, 2162, 'the  ‘ Borrower"'),
                           (2175, 2188, 'HSBC BANK PLC'),
                           (2225, 2235, 'the “Bank”')])


        prov_labels_map = annotate_doc('mytest/doc16.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(74, 93, 'GLAXOSMITHKLINE pic'),
                           (95, 109, 'the "Borrower"'),
                           (124, 137, 'HSBC BANK PLC'),
                           (139, 149, 'the "Bank"')])


        # box
        prov_labels_map = annotate_doc('mytest/doc100.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          # missing 'box', but was in the party_line!?
                          [(224, 238, 'Documents Inc.'),
                           (266, 275, '”Partner”'),
                           (367, 382, 'Box and Partner'),
                           (419, 428, 'a “Party”')])

        prov_labels_map = annotate_doc('mytest/doc101.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [(158, 186, 'CUSTOMER THREE HOLDINGS B.V.'),
                           (361, 365, '"C3"'),
                           (483, 499, 'Box.com (UK) Ltd'),
                           (601, 607, '"Box "')])

        """
        prov_labels_map = annotate_doc('mytest/doc102.txt')
        party_list = get_party_list(prov_labels_map)
        self.assertEquals(party_list,
                          [])
        """


if __name__ == "__main__":
    unittest.main()
