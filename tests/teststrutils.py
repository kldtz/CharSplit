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

    def test_get_consecutive_one_char_parens_mats(self):
        # pylint: disable=line-too-long
        line = '1) aba bd (b) a2df'
        mat_list = strutils.get_consecutive_one_char_parens_mats(line)
        print("mat_list = {}".format(mat_list))
        self.assertEqual(len(mat_list), 2)

        line2 = '(a) aba bd ii) a2df'
        mat_list2 = strutils.get_consecutive_one_char_parens_mats(line2)
        print("mat_list2 = {}".format(mat_list2))
        self.assertEqual(len(mat_list2), 2)

        line2 = '(toma) aba bd ii) a2df'
        mat_list2 = strutils.get_consecutive_one_char_parens_mats(line2)
        print("mat_list2 = {}".format(mat_list2))
        self.assertEqual(len(mat_list2), 0)

    def test_word_comma_tokenizer(self):
        line = '1) aba Bd (b) a2df.'
        se_tok_list = list(strutils.word_comma_tokenize(line))
        self.assertEqual(se_tok_list, [(0, 1, '1'), (3, 6, 'aba'), (7, 9, 'Bd'), (11, 12, 'b'), (14, 18, 'a2df')])

        line = 'I.B.M. and Dell Inc., are in a war, battle, and cold-war.'
        se_tok_list = list(strutils.word_comma_tokenize(line))
        self.assertEqual(se_tok_list, [(0, 6, 'I.B.M.'), (7, 10, 'and'), (11, 15, 'Dell'),
                                       (16, 20, 'Inc.'), (20, 21, ','), (22, 25, 'are'),
                                       (26, 28, 'in'), (29, 30, 'a'), (31, 34, 'war'),
                                       (34, 35, ','), (36, 42, 'battle'), (42, 43, ','),
                                       (44, 47, 'and'), (48, 52, 'cold'), (53, 56, 'war')])


    def test_find_previous_word(self):
        line = '1) aba bd (b) a2df'
        start, end, word = strutils.find_previous_word(line, 3)
        self.assertEqual(word, '1')

        start, end, word = strutils.find_previous_word(line, 2)
        self.assertEqual(word, '1')

        start, end, word = strutils.find_previous_word(line, 1)
        self.assertEqual(word, '1')

        start, end, word = strutils.find_previous_word(line, 0)
        self.assertEqual(start, -1)

        start, end, word = strutils.find_previous_word(line, 30)
        self.assertEqual(start, -1)

        start, end, word = strutils.find_previous_word(line, 11)
        self.assertEqual(word, 'bd')

        start, end, word = strutils.find_previous_word(line, 15)
        self.assertEqual(word, 'b')

        start, end, word = strutils.find_previous_word(line, 6)
        self.assertEqual(word, 'aba')




if __name__ == "__main__":
    unittest.main()
