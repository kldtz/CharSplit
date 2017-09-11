#!/usr/bin/env python

import unittest

from kirke.docstruct.docstructutils import is_line_page_num, is_line_signature_prefix


class TestStrUtils(unittest.TestCase):

    def test_is_line_page_num(self):
        "Test is_line_page_num()"
        self.assertTrue(is_line_page_num('3'))
        self.assertTrue(is_line_page_num('-3-'))
        self.assertTrue(is_line_page_num('page -3-'))
        self.assertTrue(is_line_page_num('page 3'))
        self.assertTrue(is_line_page_num('PAge 3'))
        self.assertFalse(is_line_page_num('Page 3a'))

    def test_is_line_signature_prefix(self):
        "Test is_line_signature_prefix()"
        self.assertTrue(is_line_signature_prefix('By: _'))
        self.assertTrue(is_line_signature_prefix('By:'))
        self.assertTrue(is_line_signature_prefix('By_: _'))
        self.assertTrue(is_line_signature_prefix('Name.~: _'))
        self.assertTrue(is_line_signature_prefix('Title_: _'))
        self.assertTrue(is_line_signature_prefix('Title_ : _'))
        self.assertFalse(is_line_signature_prefix('Title_jjj : _'))
        self.assertTrue(is_line_signature_prefix('Title_jj: asdf asdfasdf:_'))

if __name__ == "__main__":
    unittest.main()

