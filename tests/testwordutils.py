#!/usr/bin/env python

import unittest

from kirke.utils import wordutils

class TestWordUtils(unittest.TestCase):

    def test_get_all_words_ge_2chars(self):
        "Test get_all_words_ge_2chars()"

        self.assertEqual(wordutils.get_all_words_ge_2chars("Limited inc has a big sign."),
                         ["Limited", "inc", "has", "big", "sign"])
        self.assertEqual(wordutils.get_all_words_ge_2chars("Limited inc has a big sign.  Limited inc has a big sign."),
                         ["Limited", "inc", "has", "big", "sign",
                          "Limited", "inc", "has", "big", "sign"])
        self.assertEqual(wordutils.get_all_words_ge_2chars("International Business Machine."),
                         ["International", "Business", "Machine"])
        self.assertEqual(wordutils.get_all_words_ge_2chars("International Business Machine, 2 p.m. am"),
                         ["International", "Business", "Machine", "am"])

    def test_is_word_overlap_66p(self):
        "Test is_word_overlap_66p()"

        self.assertTrue(wordutils.is_word_overlap_ge_66p("International Business Machine.",
                                                         "International Business"))
        self.assertTrue(wordutils.is_word_overlap_ge_66p("International Business",
                                                         "International Business Machine."))
        self.assertTrue(wordutils.is_word_overlap_ge_66p("Happy Hour",
                                                         "Happy Hour"))
        self.assertTrue(wordutils.is_word_overlap_ge_66p("Company",
                                                         "Company"))
        self.assertFalse(wordutils.is_word_overlap_ge_66p("Company",
                                                          ""))
        self.assertFalse(wordutils.is_word_overlap_ge_66p("", ""))
        self.assertFalse(wordutils.is_word_overlap_ge_66p("a b c", "a b c"))
        self.assertFalse(wordutils.is_word_overlap_ge_66p("a b c", "a"))
        self.assertFalse(wordutils.is_word_overlap_ge_66p("Company",
                                                          "CompanyB"))
        self.assertTrue(wordutils.is_word_overlap_ge_66p("International Business Machine",
                                                         "International Business Machine Corp."))
        self.assertTrue(wordutils.is_word_overlap_ge_66p("International Business Machine.",
                                                         "International Business Machine Corp."))

if __name__ == "__main__":
    unittest.main()

