#!/usr/bin/env python3

import unittest
# import pprint
import copy
import shutil
# pylint: disable=unused-import
from typing import Any, Dict, Set

from kirke.docstruct import pdftxtparser
from kirke.utils import docworddiff, ebantdoc4, osutils, txtreader


WORK_DIR = 'dir-work'

osutils.mkpath(WORK_DIR)


class TestDocStructUtils(unittest.TestCase):

    def test_two_column_docs_8410(self):
        txt_base_name = '8410.txt'
        txt_fname = 'twoColTxt/{}'.format(txt_base_name)
        ebantdoc4.clear_cache(txt_fname, work_dir=WORK_DIR)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            print('page.num_column = {}'.format(apage.num_column))
            multi_col_list.append(apage.is_multi_column_pformat())
        self.assertEqual(multi_col_list, [True])

    def test_two_column_docs_8411(self):
        txt_base_name = '8411.txt'
        txt_fname = 'twoColTxt/{}'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())
        self.assertEqual(multi_col_list, [True, True, False])

    def test_two_column_docs_8412(self):
        txt_base_name = '8412.txt'
        txt_fname = 'twoColTxt/{}'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())
        self.assertEqual(multi_col_list,
                         [True, True, True, True, False, False])

    def test_two_column_docs_8413(self):
        txt_base_name = '8413.txt'
        txt_fname = 'twoColTxt/{}'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())
        self.assertEqual(multi_col_list,
                         [True, True, True])

    def test_two_column_docs_8414(self):
        txt_base_name = '8414.txt'
        txt_fname = 'twoColTxt/{}'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())
        self.assertEqual(multi_col_list,
                         [True, False])

    def test_two_column_docs_8415(self):
        txt_base_name = '8415.txt'
        txt_fname = 'twoColTxt/{}'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())
        self.assertEqual(multi_col_list,
                         [True, True, True, True, True, True, True, True,
                          False, False, False, False, False, False])

    def test_two_column_docs_ashtabula(self):
        txt_base_name = 'ashtabula'
        txt_fname = 'data-rate-table/{}.txt'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        gold_list = [False] * 46
        gold_list[40] = True  # page 41, 43
        gold_list[42] = True

        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())

        self.assertEquals(gold_list, multi_col_list)

    def test_two_column_docs_carousel(self):
        self.maxDiff = None
        txt_base_name = 'carousel'
        txt_fname = 'data-rate-table/{}.txt'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        gold_list = [False] * 99
        gold_list[91] = True  # page 92, it's not normal text
        gold_list[98] = True  # page 99, it's not normal text

        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())

        for val_i, aval in enumerate(multi_col_list):
            if aval:
                print('val_i = {}'.format(val_i))

        self.assertEqual(gold_list, multi_col_list)
        # self.assertFalse(any(multi_col_list))

    def test_two_column_docs_cedarbluff(self):
        self.maxDiff = None
        txt_base_name = 'cedarbluff'
        txt_fname = 'data-rate-table/{}.txt'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        gold_list = [False] * 94
        # gold_list[91] = True  # page 92, it's not normal text
        # gold_list[98] = True  # page 99, it's not normal text

        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())

        for val_i, aval in enumerate(multi_col_list):
            if aval:
                print('val_i = {}'.format(val_i))

        self.assertEqual(gold_list, multi_col_list)
        # self.assertFalse(any(multi_col_list))

    def test_two_column_docs_cornbelt(self):
        self.maxDiff = None
        txt_base_name = 'cornbelt'
        txt_fname = 'data-rate-table/{}.txt'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        gold_list = [False] * 44
        # gold_list[91] = True  # page 92, it's not normal text
        # gold_list[98] = True  # page 99, it's not normal text

        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())

        for val_i, aval in enumerate(multi_col_list):
            if aval:
                print('val_i = {}'.format(val_i))

        self.assertEqual(gold_list, multi_col_list)
        # self.assertFalse(any(multi_col_list))

    def test_two_column_docs_crystallake(self):
        self.maxDiff = None
        txt_base_name = 'crystallake'
        txt_fname = 'data-rate-table/{}.txt'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        gold_list = [False] * 58
        gold_list[57] = True  # page 92, it's not normal text
        # gold_list[98] = True  # page 99, it's not normal text

        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())

        for val_i, aval in enumerate(multi_col_list):
            if aval:
                print('val_i = {}'.format(val_i))

        self.assertEqual(gold_list, multi_col_list)
        # self.assertFalse(any(multi_col_list))

    def test_two_column_docs_daycounty(self):
        self.maxDiff = None
        txt_base_name = 'daycounty'
        txt_fname = 'data-rate-table/{}.txt'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        gold_list = [False] * 48
        gold_list[45] = True  # page 92, it's not normal text
        # gold_list[98] = True  # page 99, it's not normal text

        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())

        for val_i, aval in enumerate(multi_col_list):
            if aval:
                print('val_i = {}'.format(val_i))

        self.assertEqual(gold_list, multi_col_list)
        # self.assertFalse(any(multi_col_list))

    def test_two_column_docs_highwinds(self):
        self.maxDiff = None
        txt_base_name = 'highwinds'
        txt_fname = 'data-rate-table/{}.txt'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        gold_list = [False] * 92
        gold_list[57] = True
        gold_list[60] = True
        gold_list[69] = True
        gold_list[71] = True
        gold_list[72] = True
        gold_list[74] = True
        gold_list[84] = True
        gold_list[85] = True

        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())

        for val_i, aval in enumerate(multi_col_list):
            if aval:
                print('val_i = {}'.format(val_i))

        self.assertEqual(gold_list, multi_col_list)
        # self.assertFalse(any(multi_col_list))


    def test_two_column_docs_windenergy(self):
        self.maxDiff = None
        txt_base_name = 'windenergy'
        txt_fname = 'data-rate-table/{}.txt'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        gold_list = [False] * 108
        gold_list[79] = True
        gold_list[80] = True
        gold_list[81] = True

        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column_pformat())

        for val_i, aval in enumerate(multi_col_list):
            if aval:
                print('val_i = {}'.format(val_i))

        self.assertEqual(gold_list, multi_col_list)


if __name__ == "__main__":
    unittest.main()
