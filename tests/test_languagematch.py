# pylint: disable=missing-docstring,wildcard-import
import unittest
from kirke.nlputil.languagematch import *

class TestCanonicalize(unittest.TestCase):

    def test_canonicalize(self):
        self.assertEqual(canonicalize(None, 'en'), 'en')
        self.assertEqual(canonicalize('', 'en'), 'en')
        self.assertEqual(canonicalize('en-US', 'en'), 'en-us')
        self.assertEqual(canonicalize('en_US', 'en'), 'en-us')

class TestLanguageBasicFilterMatch(unittest.TestCase):

    def test_simple_match(self):
        self.assertTrue(language_basic_filter_match('', ''))
        self.assertTrue(language_basic_filter_match('en', ''))
        self.assertTrue(language_basic_filter_match(None, 'en'))
        self.assertTrue(language_basic_filter_match('en-us', 'en_us'))
        self.assertFalse(language_basic_filter_match('en', 'de'))
        self.assertFalse(language_basic_filter_match('', 'de'))

    def test_equality_match(self):
        self.assertTrue('de', 'de')

    def test_extension_match(self):
        # We need generic English, we only have American English: reject.
        self.assertFalse(language_lookup_match('en', 'en-us'))
        # We need American English, we have generic English: accept.
        self.assertTrue(language_lookup_match('en-us', 'en'))
        # Total mismatch
        self.assertFalse(language_lookup_match('en-us', 'en-uk'))

    def test_star_match(self):
        self.assertTrue(language_lookup_match('*', 'ja'))
        self.assertFalse(language_lookup_match('en-*', 'en'))


class TestLanguageLookupMatch(unittest.TestCase):

    def test_simple_match(self):
        self.assertTrue(language_lookup_match('', ''))
        self.assertTrue(language_lookup_match('en', ''))
        self.assertTrue(language_lookup_match(None, 'en'))
        self.assertTrue(language_lookup_match('en-us', 'en_us'))
        self.assertFalse(language_lookup_match('en', 'de'))
        self.assertFalse(language_lookup_match('', 'de'))

    def test_reduction_match(self):
        # We need something en-us, en is the best we can do: accept.
        self.assertTrue(language_lookup_match('en-us', 'en'))
        # We need something en, we have en-us: too specific, reject.
        self.assertFalse(language_lookup_match('en', 'en-us'))
        # US English and UK English just don't match
        self.assertFalse(language_lookup_match('en-us', 'en-uk'))

    def test_star_match(self):
        self.assertTrue(language_lookup_match('*', 'ja'))
        self.assertFalse(language_lookup_match('en-*', 'en'))

    def test_singleton_match(self):
        # We need NYC English, we only have generic English: reject.
        self.assertFalse(language_lookup_match('en-x-nyc', 'en'))
        # Misuse of singleton
        self.assertTrue(language_lookup_match('en-x-nyc', 'en-x'))
