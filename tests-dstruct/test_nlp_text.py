#!/usr/bin/env python3

import unittest
# import pprint
import copy
import os
import shutil
# pylint: disable=unused-import
from typing import Any, Dict, Set

from kirke.eblearn import ebrunner

from kirke.docstruct import pdftxtparser
from kirke.utils import docworddiff, ebantdoc4, osutils, txtreader
from kirke.utils.ebantdoc4 import pdf_to_ebantdoc, get_nlp_file_name


WORK_DIR = 'dir-work'
osutils.mkpath(WORK_DIR)


class TestNLPText(unittest.TestCase):

    def test_nlp_text_1(self):
        txt_base_name = 'trilinc.txt'
        doc_id = txt_base_name.replace('.txt', '')
        txt_fname = '{}/{}'.format(WORK_DIR, txt_base_name)
        offsets_base_name = txt_base_name.replace('.txt', '.offsets.json')
        offsets_fname = '{}/{}'.format(WORK_DIR, offsets_base_name)
        pdfxml_base_name = txt_base_name.replace('.txt', '.pdf.xml')        
        pdfxml_fname = '{}/{}'.format(WORK_DIR, pdfxml_base_name)
        shutil.copy2('dir-test-doc/{}'.format(txt_base_name), txt_fname)
        shutil.copy2('dir-test-doc/{}'.format(offsets_base_name), offsets_fname)
        if os.path.exists('dir-test-doc/{}'.format(pdfxml_base_name)):        
            shutil.copy2('dir-test-doc/{}'.format(pdfxml_base_name), pdfxml_fname)        

        ebantdoc = pdf_to_ebantdoc(txt_fname,
                                   offsets_fname,
                                   pdfxml_fname,
                                   work_dir=WORK_DIR)
        nlptxt_md5 = osutils.get_text_md5(ebantdoc.get_nlp_text())
        nlptxt_file_name = get_nlp_file_name(doc_id,
                                             nlptxt_md5=nlptxt_md5,
                                             work_dir=WORK_DIR)
        same_list, diff_list = docworddiff.diff_word_lists('{}/{}'.format(WORK_DIR, txt_base_name),
                                                           nlptxt_file_name)
        self.assertEqual(len(same_list), 1974)
        self.assertEqual(len(diff_list), 0)

    def test_nlp_text_2(self):
        txt_base_name = 'carousel.txt'
        doc_id = txt_base_name.replace('.txt', '')
        txt_fname = '{}/{}'.format(WORK_DIR, txt_base_name)
        offsets_base_name = txt_base_name.replace('.txt', '.offsets.json')
        offsets_fname = '{}/{}'.format(WORK_DIR, offsets_base_name)
        pdfxml_base_name = txt_base_name.replace('.txt', '.pdf.xml')        
        pdfxml_fname = '{}/{}'.format(WORK_DIR, pdfxml_base_name)        
        shutil.copy2('dir-test-doc/{}'.format(txt_base_name), txt_fname)
        shutil.copy2('dir-test-doc/{}'.format(offsets_base_name), offsets_fname)
        if os.path.exists('dir-test-doc/{}'.format(pdfxml_base_name)):
            shutil.copy2('dir-test-doc/{}'.format(pdfxml_base_name), pdfxml_fname)
        
        ebantdoc = pdf_to_ebantdoc(txt_fname,
                                   offsets_fname,
                                   pdfxml_fname,
                                   work_dir=WORK_DIR)
        nlptxt_md5 = osutils.get_text_md5(ebantdoc.get_nlp_text())
        nlptxt_file_name = get_nlp_file_name(doc_id,
                                             nlptxt_md5=nlptxt_md5,
                                             work_dir=WORK_DIR)
        same_list, diff_list = docworddiff.diff_word_lists('{}/{}'.format(WORK_DIR, txt_base_name),
                                                           nlptxt_file_name)

        self.assertEqual(len(same_list), 5963)
        self.assertEqual(len(diff_list), 0)

    def test_is_continued_page_1(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        txt_base_name = 'trilinc.txt'
        txt_fname = '{}/{}'.format(WORK_DIR, txt_base_name)
        offsets_base_name = txt_base_name.replace('.txt', '.offsets.json')
        offsets_fname = '{}/{}'.format(WORK_DIR, offsets_base_name)
        shutil.copy2('dir-test-doc/{}'.format(txt_base_name), txt_fname)
        shutil.copy2('dir-test-doc/{}'.format(offsets_base_name), offsets_fname)

        pdf_text_doc = pdftxtparser.parse_document(txt_fname,
                                                   work_dir=WORK_DIR)

        is_continued_list = []  # type: List[bool]
        for apage in pdf_text_doc.page_list:
            is_continued_list.append(apage.is_continued_para_from_prev_page)

        print("is_continued_list:")
        print(is_continued_list)

        gold_list = [False, False, False, False, False, False, False,
                     False, False, False, False, False, True, False,
                     True, False, False, True, False, False]

        self.assertEqual(is_continued_list, gold_list)


    def test_is_continued_page_2(self):
        # pylint: disable=invalid-name
        self.maxDiff = None

        txt_base_name = 'carousel.txt'
        txt_fname = '{}/{}'.format(WORK_DIR, txt_base_name)
        offsets_base_name = txt_base_name.replace('.txt', '.offsets.json')
        offsets_fname = '{}/{}'.format(WORK_DIR, offsets_base_name)
        shutil.copy2('dir-test-doc/{}'.format(txt_base_name), txt_fname)
        shutil.copy2('dir-test-doc/{}'.format(offsets_base_name), offsets_fname)

        pdf_text_doc = pdftxtparser.parse_document(txt_fname,
                                                   work_dir=WORK_DIR)
        is_continued_list = []  # type: List[bool]
        for apage in pdf_text_doc.page_list:
            is_continued_list.append(apage.is_continued_para_from_prev_page)

        print("is_continued_list:")
        print(is_continued_list)

        gold_list = [False, False, False, False, False, False, False,
                     True, False, False, False, True, False, False, False,
                     True, False, True, False, True, False, False, False,
                     False, False, True, False, True, False, False, True,
                     False, True, False, False, True, True, False, False,
                     True, True, True, False, True, True, False, False,
                     True, False, False, False, True, True, True, False,
                     True, False, True, False, True, True, False, False, True,
                     False, False, False, False, True, False, False, True,
                     True, False, False, False, False, False, False, True,
                     True, False, False, False, False, False, False, False,
                     False, False, False, False, False, False, True, False,
                     True, False, False]

        for pnum, (is_continued, gval) in enumerate(zip(is_continued_list, gold_list), 1):
            if is_continued != gval:
                print("page {}, is_continued = {}, gold = {}".format(pnum,
                                                                     is_continued,
                                                                     gval))

        self.assertEqual(is_continued_list, gold_list)


if __name__ == "__main__":
    unittest.main()
