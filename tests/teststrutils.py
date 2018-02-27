#!/usr/bin/env python3

import unittest

from kirke.utils import strutils


class TestStrUtils(unittest.TestCase):

    def test_split_words(self):
        self.assertEqual(strutils.split_words('123-456'), ['123', '456'])
        self.assertEqual(strutils.split_words('page 3'), ['page', '3'])
        self.assertEqual(strutils.split_words('page-3'), ['page', '3'])
        self.assertEqual(strutils.split_words('I like to eat, but not to drink.'),
                         ['I', 'like', 'to', 'eat', 'but', 'not', 'to', 'drink'])

        self.assertEqual(strutils.split_words('He has a Ph.D.'),
                         ['He', 'has', 'a', 'Ph.D.'])
        self.assertEqual(strutils.split_words('He has a Ph.D'),
                         ['He', 'has', 'a', 'Ph.D'])
        self.assertEqual(strutils.split_words('He likes I.B.M.'),
                         ['He', 'likes', 'I.B.M.'])
        self.assertEqual(strutils.split_words('He likes I.B.M'),
                         ['He', 'likes', 'I.B.M'])
        self.assertEqual(strutils.split_words('His name is Matt A. Jacobson.'),
                         ['His', 'name', 'is', 'Matt', 'A.', 'Jacobson'])
        self.assertEqual(strutils.split_words('His name is Matt A Jacobson.'),
                         ['His', 'name', 'is', 'Matt', 'A', 'Jacobson'])
        self.assertEqual(strutils.split_words('- test kits'),
                         ['test', 'kits'])

    def test_extract_numbers(self):
        self.assertEqual(strutils.extract_numbers('123-456'), ['123', '456'])
        self.assertEqual(strutils.extract_numbers('3 42.00 42.00 69.33'),
                         ['3', '42.00', '42.00', '69.33'])

    def test_count_numbers(self):
        self.assertEqual(strutils.count_numbers('123-456'), 2)
        self.assertEqual(strutils.count_numbers('3 42.00 42.00 69.33'), 4)

    def test_using_split2(self):
        self.assertEqual(strutils.using_split2('a'),
                         [(0, 1, 'a')])
        self.assertEqual(strutils.using_split2('one two three'),
                         [(0, 3, 'one'), (4, 7, 'two'), (8, 13, 'three')])
        self.assertEqual(strutils.using_split2('  one two three  '),
                         [(2, 5, 'one'), (6, 9, 'two'), (10, 15, 'three')])
        self.assertEqual(strutils.using_split2(''),
                         [])
        self.assertEqual(strutils.using_split2(' '),
                         [])

    def test_split_with_offsets(self):
        # pylint: disable=line-too-long
        astr = 'Digital Realty Trust, L.P., a Maryland limited partnership (the “Operating Partnership”)'
        self.assertEqual(strutils.split_with_offsets(astr),
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
        # pylint: disable=line-too-long
        astr = 'Digital Realty Trust, L.P., a Maryland limited partnership (the “Operating Partnership”)'
        self.assertEqual(strutils.split_with_offsets_xparens(astr),
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
