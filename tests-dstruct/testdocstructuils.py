#!/usr/bin/env python3

import unittest

from kirke.docstruct.docstructutils import is_line_page_num, is_line_signature_prefix, is_line_title


class TestStrUtils(unittest.TestCase):

    def test_is_line_page_num(self):
        "Test is_line_page_num()"
        self.assertTrue(is_line_page_num('3'))
        self.assertTrue(is_line_page_num('-3-'))
        self.assertTrue(is_line_page_num('page -3-'))
        self.assertTrue(is_line_page_num('page 3'))
        self.assertTrue(is_line_page_num('PAge 3'))
        self.assertFalse(is_line_page_num('Page 3a'))

    def test_is_line_page_num_roman(self):
        self.assertTrue(is_line_page_num('i'))
        self.assertTrue(is_line_page_num('ii'))
        self.assertTrue(is_line_page_num('iii'))
        self.assertTrue(is_line_page_num('iv'))
        self.assertTrue(is_line_page_num('v'))
        self.assertTrue(is_line_page_num('vi'))
        self.assertTrue(is_line_page_num('x'))
        self.assertTrue(is_line_page_num('xi'))
        
        self.assertTrue(is_line_page_num('Page i'))
        self.assertTrue(is_line_page_num('Page ii'))
        self.assertTrue(is_line_page_num('Page iii'))
        self.assertTrue(is_line_page_num('Page iv'))
        self.assertTrue(is_line_page_num('Page v'))
        self.assertTrue(is_line_page_num('Page vi'))
        self.assertTrue(is_line_page_num('Page x'))
        self.assertTrue(is_line_page_num('Page xi'))                                        

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

    def test_is_line_title(self):
        "Test is_line_title()"
        self.assertTrue(is_line_title('New York State Department of Taxation and Finance'))
        self.assertTrue(is_line_title('New York State and Local Sales and Use Tax ST-121'))
        self.assertTrue(is_line_title('-—J , Exempt Use Certificate'))

if __name__ == "__main__":
    unittest.main()

