#!/usr/bin/env python3

import unittest

from kirke.utils import strutils
from kirke.utils import stopwordutils

    
class TestStrUtils(unittest.TestCase):

    def test_split_words(self):
        self.assertEquals(strutils.split_words('123-456'), ['123', '456'])
        self.assertEquals(strutils.split_words('page 3'), ['page', '3'])
        self.assertEquals(strutils.split_words('page-3'), ['page', '3'])
        self.assertEquals(strutils.split_words('I like to eat, but not to drink.'),
                          ['I', 'like', 'to', 'eat', 'but', 'not', 'to', 'drink'])

        self.assertEquals(strutils.split_words('He has a Ph.D.'),
                          ['He', 'has', 'a', 'Ph.D.'])
        self.assertEquals(strutils.split_words('He has a Ph.D'),
                          ['He', 'has', 'a', 'Ph.D'])
        self.assertEquals(strutils.split_words('He likes I.B.M.'),
                          ['He', 'likes', 'I.B.M.'])
        self.assertEquals(strutils.split_words('He likes I.B.M'),
                          ['He', 'likes', 'I.B.M'])
        self.assertEquals(strutils.split_words('His name is Matt A. Jacobson.'),
                          ['His', 'name', 'is', 'Matt', 'A.', 'Jacobson'])
        self.assertEquals(strutils.split_words('His name is Matt A Jacobson.'),
                          ['His', 'name', 'is', 'Matt', 'A', 'Jacobson'])
        self.assertEquals(strutils.split_words('- test kits'),
                          ['test', 'kits'])

    def test_extract_numbers(self):
        self.assertEquals(strutils.extract_numbers('123-456'), ['123', '456'])
        self.assertEquals(strutils.extract_numbers('3 42.00 42.00 69.33'), ['3', '42.00', '42.00', '69.33'])

    def test_count_numbers(self):
        self.assertEquals(strutils.count_numbers('123-456'), 2)
        self.assertEquals(strutils.count_numbers('3 42.00 42.00 69.33'), 4)

    def test_split_with_offsets(self):
        self.assertEquals(strutils.split_with_offsets('Digital Realty Trust, L.P., a Maryland limited partnership (the “Operating Partnership”)'),
                          [(0, 7, 'Digital'),
                           (8, 14, 'Realty'),
                           (15, 21, 'Trust,'),
                           (22, 27, 'L.P.,'),
                           (28, 29, 'a'),
                           (30, 38, 'Maryland'),
                           (39, 46, 'limited'),
                           (47, 58, 'partnership'),
                           (59, 63, '(the'),
                           (64, 74, '“Operating'),
                           (75, 88, 'Partnership”)')])

    def test_split_with_offsets_xpans(self):
        self.assertEquals(strutils.split_with_offsets_xparens('Digital Realty Trust, L.P., a Maryland limited partnership (the “Operating Partnership”)'),
                          [(0, 7, 'Digital'),
                           (8, 14, 'Realty'),
                           (15, 21, 'Trust,'),
                           (22, 27, 'L.P.,'),
                           (28, 29, 'a'),
                           (30, 38, 'Maryland'),
                           (39, 46, 'limited'),
                           (47, 58, 'partnership'),
                           (59, 60, '('),
                           (60, 63, 'the'),
                           (64, 74, '“Operating'),
                           (75, 87, 'Partnership”'),
                           (87, 88, ')')])

if __name__ == "__main__":
    unittest.main()

