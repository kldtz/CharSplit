import unittest
from kirke.nlputil.languagematch import *

class TestLanguageLookupMatch:

    def simple_match(self):
        self.assertTrue(needed='', available='')
        self.assertTrue(needed='en', available='')
        self.assertTrue(needed=None, available='en')
        self.assertTrue(needed='en-us', available='en_us')
        self.assertFalse(needed='en', available='de')
        self.assertFalse(needed='', available='de')

    def reduction_match(self):
        self.assertTrue(needed='en', available='en-us')
        self.assertFalse(needed='en-us', available='en')

    def star_match(self):
        self.assertTrue(needed='*', available='ja')
        self.assertFalse(needed='en-*', available='en')

    def singleton_match(self):
      self.assertTrue(needed='en', available='en-x-nyc')
      self.assertFalse(needed='en-x', available='en-x-nyc')

    def gonna_fail(self):
        self.assertTrue(False)
