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
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column)
        self.assertEqual([True], multi_col_list)

    def test_two_column_docs_8411(self):        
        txt_base_name = '8411.txt'
        txt_fname = 'twoColTxt/{}'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column)
        self.assertEqual([True, True, False], multi_col_list)

    def test_two_column_docs_8412(self):                
        txt_base_name = '8412.txt'
        txt_fname = 'twoColTxt/{}'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column)
        self.assertEqual([True, True, True, True, False, False],
                         multi_col_list)

    def test_two_column_docs_8413(self):                        
        txt_base_name = '8413.txt'
        txt_fname = 'twoColTxt/{}'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column)
        self.assertEqual([True, True, True],
                         multi_col_list)

    def test_two_column_docs_8414(self):                        
        txt_base_name = '8414.txt'
        txt_fname = 'twoColTxt/{}'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column)
        self.assertEqual([True, False],
                         multi_col_list)

    def test_two_column_docs_8415(self):                        
        txt_base_name = '8415.txt'
        txt_fname = 'twoColTxt/{}'.format(txt_base_name)
        pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
        multi_col_list = []
        for apage in pdf_text_doc.page_list:
            multi_col_list.append(apage.is_multi_column)
        self.assertEqual([True, True, True, True, True, True, True, True,
                         False, False, False, False, False, False],
                         multi_col_list)

    def test_one_column_docs(self):
        doc_id_list = ['ashtabula', 'carousel', 'cedarbluff', 'cornbelt',
                       'crystallake', 'daycounty', 'highwinds', 'windenergy']
        for doc_id in doc_id_list:
            txt_fname = 'data-rate-table/{}.txt'.format(doc_id)

            pdf_text_doc = pdftxtparser.parse_document(txt_fname, work_dir=WORK_DIR)
            multi_col_list = []
            for apage in pdf_text_doc.page_list:
                multi_col_list.append(apage.is_multi_column)

        self.assertFalse(any(multi_col_list))
        

        
if __name__ == "__main__":
    unittest.main()
