#!/usr/bin/env python3

import unittest

from kirke.utils import strutils
from kirke.utils import stopwordutils

    
class TestStopwordsUtils(unittest.TestCase):

    def test_is_line_title_non_stopwords(self):
        self.assertTrue(stopwordutils.is_line_title_non_stopwords('This Is a Good THING.'))
        self.assertFalse(stopwordutils.is_line_title_non_stopwords('This Is a good THING.'))

    def test_getnonstopwords_gt_len1(self):
        line = 'This Is a Good THING.'
        self.assertEqual(['good', 'thing'],
                         stopwordutils.get_nonstopwords_gt_len1(line))

        line = "7. “지체상금”이라 함은 수급사업자가 납품기일에 목적물을 납품하지 않을 경우  원사업자에게 지급해야 할 손해배상금을 말한다."
        self.assertEqual(['지체상금', '이라', '함은', '수급사업자가', '납품기일에',
                          '목적물을', '납품하지', '않을', '경우', '원사업자에게',
                          '지급해야', '손해배상금을', '말한다'],
                         stopwordutils.get_nonstopwords_gt_len1(line))


if __name__ == "__main__":
    unittest.main()

