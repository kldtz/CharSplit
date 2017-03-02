#!/usr/bin/env python

import unittest
from utils.sentutils import is_page_number_st

class TestStrUtils(unittest.TestCase):

    def test_is_page_number_st(self):
        "Test is_page_number_st()"
        self.assertTrue(is_page_number_st('3'))
        self.assertTrue(is_page_number_st('-3-'))
        self.assertTrue(is_page_number_st('page -3-'))
        self.assertTrue(is_page_number_st('page 3'))
        self.assertTrue(is_page_number_st('PAge 3'))
        self.assertFalse(is_page_number_st('Page 3a'))                                

if __name__ == "__main__":
    unittest.main()

