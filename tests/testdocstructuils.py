#!/usr/bin/env python

import unittest

from kirke.docstruct.docstructutils import is_line_page_num


class TestStrUtils(unittest.TestCase):

    def test_is_line_page_num(self):
        "Test is_line_page_num()"
        self.assertTrue(is_line_page_num('3'))
        self.assertTrue(is_line_page_num('-3-'))
        self.assertTrue(is_line_page_num('page -3-'))
        self.assertTrue(is_line_page_num('page 3'))
        self.assertTrue(is_line_page_num('PAge 3'))
        self.assertFalse(is_line_page_num('Page 3a'))                                

if __name__ == "__main__":
    unittest.main()

