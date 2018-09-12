#!/usr/bin/env python3

import unittest

from kirke.docstruct import pdftxtparser
from kirke.utils import docworddiff
from kirke.utils.ebantdoc4 import CORENLP_JSON_VERSION

WORK_DIR = 'dir-work'

class TestNLPTxt(unittest.TestCase):

    def test_nlptxt_demo_8285(self):
        "Test NLP txt has same set of words"

        doc_id = '8285'
        fname = 'demo-txt/{}.txt'.format(doc_id)
        nlp_fname = 'dir-work/{}.nlp.v{}.txt'.format(doc_id,
                                                     CORENLP_JSON_VERSION)
        unused_pdf_doc = pdftxtparser.parse_document(fname,
                                                     work_dir=WORK_DIR,
                                                     nlptxt_file_name=nlp_fname)
        same_list, diff_list = docworddiff.diff_word_lists(fname, nlp_fname)
        # to verify that two docs have the right same number
        self.assertEqual(len(same_list), 1712)
        self.assertEqual(len(diff_list), 0)


    def test_nlptxt_demo_8300(self):
        "Test NLP txt has same set of words"

        doc_id = '8300'
        fname = 'demo-txt/{}.txt'.format(doc_id)
        nlp_fname = 'dir-work/{}.nlp.v{}.txt'.format(doc_id,
                                                     CORENLP_JSON_VERSION)
        unused_pdf_doc = pdftxtparser.parse_document(fname,
                                                     work_dir=WORK_DIR,
                                                     nlptxt_file_name=nlp_fname)
        same_list, diff_list = docworddiff.diff_word_lists(fname, nlp_fname)
        # to verify that two docs have the right same number
        self.assertEqual(len(same_list), 2009)
        self.assertEqual(len(diff_list), 0)


    def test_nlptxt_demo_txt(self):
        "Test NLP txt has same set of words"

        # this only go up to 8299
        for doc_id in range(8286, 8300):
            fname = 'demo-txt/{}.txt'.format(doc_id)
            nlp_fname = 'dir-work/{}.nlp.v{}.txt'.format(doc_id,
                                                         CORENLP_JSON_VERSION)
            unused_pdf_doc = pdftxtparser.parse_document(fname,
                                                         work_dir=WORK_DIR,
                                                         nlptxt_file_name=nlp_fname)
            same_list, diff_list = docworddiff.diff_word_lists(fname, nlp_fname)
            # just need to know same is not 0
            self.assertGreater(len(same_list), 100)
            self.assertEqual(len(diff_list), 0)


if __name__ == "__main__":
    unittest.main()
