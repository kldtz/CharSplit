#!/usr/bin/env python3

import unittest

from kirke.utils import strutils
from kirke.utils import stopwordutils

    
class TestStopwordsUtils(unittest.TestCase):

    def test_is_line_title_non_stopwords(self):
        self.assertTrue(stopwordutils.is_line_title_non_stopwords('This Is a Good THING.'))
        self.assertFalse(stopwordutils.is_line_title_non_stopwords('This Is a good THING.'))


if __name__ == "__main__":
    unittest.main()

