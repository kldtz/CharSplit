#!/usr/bin/env python

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

if __name__ == "__main__":
    unittest.main()

