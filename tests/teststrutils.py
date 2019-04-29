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

    def test_find_numbers(self):
        self.assertEqual(strutils.find_numbers('123-456'), ['123', '-456'])
        self.assertEqual(strutils.find_numbers('3 42.00 42.00 69.33'),
                         ['3', '42.00', '42.00', '69.33'])

    def test_count_numbers(self):
        self.assertEqual(strutils.count_numbers('123-456'), 2)
        self.assertEqual(strutils.count_numbers('3 42.00 42.00 69.33'), 4)

        line = '1) aba bd (b) a2df'
        self.assertEqual(strutils.count_numbers(line), 1)

        line = '1) 2.3 bd 4 0.4 (b) a2df'
        self.assertEqual(strutils.count_numbers(line), 4)

        line = '1) 2.3 bd 4 0.4 (b) 1,800,000 23.4 a2df'
        self.assertEqual(strutils.count_numbers(line), 6)

        line = '1) 2.3 bd 4 0.4 (b) 1,800,000 23.4 a2df a-3f'
        self.assertEqual(strutils.count_numbers(line), 6)


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

    """
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
    """


    def test_find_itemized_paren_mats(self):
        line = '(a) The Princeton Review, Inc. (the “Issuer”), (b) the Collateral Agent ' \
               '(c) the Purchasers party hereto and (d) the Guarantors party hereto.'
        mat_list = strutils.find_itemized_paren_mats(line)
        st_list = []
        for mat in mat_list:
            st_list.append(line[mat.start():mat.end()])
        self.assertEqual(st_list,
                         ['(a) ', '(b) ', '(c) ', '(d) '])

        line = 'a) The Princeton Review, Inc. (the “Issuer”), b) the Collateral Agent ' \
               'c) the Purchasers party hereto and d) the Guarantors party hereto.'
        mat_list = strutils.find_itemized_paren_mats(line)
        st_list = []
        for mat in mat_list:
            st_list.append(line[mat.start():mat.end()])
        self.assertEqual(st_list,
                         ['a) ', ' b) ', ' c) ', ' d) '])


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

    def test_text_strip(self):
        line = ''
        se_tuple = strutils.text_strip(start=0, end=len(line), text=line)
        self.assertEqual(se_tuple, (0, 0))

        line = '1'
        se_tuple = strutils.text_strip(start=0, end=len(line), text=line)
        self.assertEqual(se_tuple, (0, 1))

        line = ' 1'
        se_tuple = strutils.text_strip(start=0, end=len(line), text=line)
        self.assertEqual(se_tuple, (1, 2))

        line = ' 1 '
        se_tuple = strutils.text_strip(start=0, end=len(line), text=line)
        self.assertEqual(se_tuple, (1, 2))

        line = ' 1 2 '
        se_tuple = strutils.text_strip(start=0, end=len(line), text=line)
        self.assertEqual(se_tuple, (1, 4))

        line = ' 1  2 '
        se_tuple = strutils.text_strip(start=0, end=len(line), text=line)
        self.assertEqual(se_tuple, (1, 5))

    def normalize_spaces(self):
        line = '60,161.40\n\n\xa0\n\n\xa0\n\n$'
        out_line = strutils.normalize_spaces(line)
        self.assertEqual('60,161.40\n\n \n\n \n\n$',
                         out_line)

if __name__ == "__main__":
    unittest.main()
