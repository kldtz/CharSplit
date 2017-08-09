#!/usr/bin/env python

import unittest

from kirke.docstruct import footerutils

def parse_page_num(line, prev_line='', prev_line_idx=-1):
    sechead_type, prefix, sechead, split_idx = secheadutils.extract_sechead_v4(line,
                                                                               prev_line=prev_line,
                                                                               prev_line_idx=prev_line_idx)
    return prefix, sechead
    
class TestSecHeadUtils(unittest.TestCase):

    def test_classify_pagenum(self):
        "Test pagenum"

        self.assertTrue(footerutils.classify_line_page_number('3'))
        self.assertTrue(footerutils.classify_line_page_number('page 3'))
        self.assertTrue(footerutils.classify_line_page_number('page 3 of 10'))
        self.assertFalse(footerutils.classify_line_page_number('11-6-2009'))

        self.assertTrue(footerutils.classify_line_page_number('3'))
        self.assertTrue(footerutils.classify_line_page_number('-3-'))
        self.assertTrue(footerutils.classify_line_page_number('page -3-'))
        self.assertTrue(footerutils.classify_line_page_number('page 3'))
        self.assertTrue(footerutils.classify_line_page_number('PAge 3'))
        self.assertFalse(footerutils.classify_line_page_number('Page 3a'))


if __name__ == "__main__":
    unittest.main()

