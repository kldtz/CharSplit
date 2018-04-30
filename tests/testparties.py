#!/usr/bin/env python3

import unittest

from kirke.ebrules import parties
from kirke.docstruct import partyutils


class TestParties(unittest.TestCase):

    def test_is_org_suffix(self):
        line = 'pic'
        self.assertTrue(partyutils.is_org_suffix(line))

        line = 'n.a.'
        self.assertTrue(partyutils.is_org_suffix(line))

        line = 'n.a'
        self.assertTrue(partyutils.is_org_suffix(line))                


    def test_find_non_lc_party_name(self):
        line = 'Volkswagen Bank GmbH, a company incorporated under'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 20), 22))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Volkswagen Bank GmbH')

        line = 'HSBC Bank pic, a bank incorporated under'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 13), 15))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'HSBC Bank pic')


        line = 'HSBC Bank pic'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 13), 13))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'HSBC Bank pic')


        line = 'HSBC Bank pic '
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 13), 13))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'HSBC Bank pic')

        line = 'HSBC Bank pic a'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 13), 14))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'HSBC Bank pic')

        line = 'Johnson & Johnson, a bank incorporated under'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 17), 19))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Johnson & Johnson')

        line = 'Johnson and Johnson, a bank incorporated under'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 19), 21))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Johnson and Johnson')

        line = 'Johnson 5, a bank incorporated under'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 9), 11))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Johnson 5')

        # not sure why 'Citibank, N.A' instead of 'Citibank, N.A.'
        line = 'Citibank, N.A. is smaller'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 14), 15))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Citibank, N.A.')                

        line = 'Citibank Bank, N.A. is smaller'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 19), 20))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Citibank Bank, N.A.')

        line = 'Citibank Bank, n.a. is smaller'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 19), 20))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Citibank Bank, n.a.')                

        line = 'Business Marketing Services, Inc, One Broadway Street,'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 32), 34))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Business Marketing Services, Inc')


        line = 'Business Marketing Services, Inc. One Broadway Street,'    
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 33), 34))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Business Marketing Services, Inc.')


        line = 'Business Marketing Services, Inc., One Broadway Street,'    
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 33), 35))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Business Marketing Services, Inc.')

        line = 'Business Marketing Services, Inc,. One Broadway Street,'    
        result = partyutils.find_non_lc_party_name(line)
        # we stop at ". One ..."
        self.assertEqual(result,
                         ((0, 32), 33))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Business Marketing Services, Inc')

        line = 'Business Marketing Services, Inc'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 32), 32))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Business Marketing Services, Inc')

        line = 'Business Marketing Services, Incor'
        result = partyutils.find_non_lc_party_name(line)
        self.assertEqual(result,
                         ((0, 34), 34))
        (start, end), other_start = result
        self.assertEqual(line[start:end],
                         'Business Marketing Services, Incor')
        


class TestPartyUtils(unittest.TestCase):

    def test_get_suffix_mat_list(self):
        line = 'Volkswagen Bank GmbH, a company incorporated under'
        mat_list = partyutils.get_org_suffix_mat_list(line)
        st_list = [line[mat.start():mat.end()] for mat in mat_list]
        self.assertEqual(st_list,
                         ['GmbH'])

        line = 'Volkswagen Bank GmbH, a company incorporated under mgm from I.B.M. Corp.'
        mat_list = partyutils.get_org_suffix_mat_list(line)
        st_list = [line[mat.start():mat.end()] for mat in mat_list]
        self.assertEqual(st_list,
                         ['GmbH', 'Corp.'])

        line = 'visiting xxx Group, Ltd.'
        mat_list = partyutils.get_org_suffix_mat_list(line)
        st_list = [line[mat.start():mat.end()] for mat in mat_list]
        self.assertEqual(st_list,
                         ['Ltd.'])        


if __name__ == "__main__":
    unittest.main()
