#!/usr/bin/env python3

import unittest

from kirke.utils import nlputils

class TestNLPUtils(unittest.TestCase):

    def test_first_sentence(self):
        "Test first_sentence"

        line = 'This Non-Disclosure Agreement (“Agreement”), effective as of the last signature  date below, (“Effective Date”), is by and between Partner 4, LLC, a Virginia Corporation  having its headquarters located at 999 Parkview Drive, West Church, VA 22099, on  behalf of itself, its subsidiaries and Affiliates, (collectively, “P4”), and Box, Inc., a  Delaware Corporation having its headquarters located at 900 Jefferson Ave,  Redwood City, CA 94063, on behalf of itself and its subsidiaries and Affiliates  (collectively, “Supplier”).  The term “Affiliates” shall mean those entities controlled  by, which control or which are under common control with an identified named Party.  Such entity shall be deemed to be an Affiliate only so long as such control exists.  Such control means: (i) direct or indirect ownership or control (now or hereafter) of  more than fifty percent (50%)'
        self.assertEqual(nlputils.first_sentence(line),
                         'This Non-Disclosure Agreement (“Agreement”), effective as of the last signature  date below, (“Effective Date”), is by and between Partner 4, LLC, a Virginia Corporation  having its headquarters located at 999 Parkview Drive, West Church, VA 22099, on  behalf of itself, its subsidiaries and Affiliates, (collectively, “P4”), and Box, Inc., a  Delaware Corporation having its headquarters located at 900 Jefferson Ave,  Redwood City, CA 94063, on behalf of itself and its subsidiaries and Affiliates  (collectively, “Supplier”).')

        line = 'This Non-Disclosure Agreement (“Agreement”), effective as'
        self.assertEqual(nlputils.first_sentence(line),
                         line)

    def test_sent_tokenize(self):
        self.maxDiff = None
        
        line = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).  James'
        self.assertEqual(nlputils.sent_tokenize(line),
                         [(0, 38), (40, 68), (69, 78), (80, 85)])

        line = 'This Non-Disclosure Agreement (“Agreement”), effective as of the last signature  date below, (“Effective Date”), is by and between Partner 4, LLC, a Virginia Corporation  having its headquarters located at 999 Parkview Drive, West Church, VA 22099, on  behalf of itself, its subsidiaries and Affiliates, (collectively, “P4”), and Box, Inc., a  Delaware Corporation having its headquarters located at 900 Jefferson Ave,  Redwood City, CA 94063, on behalf of itself and its subsidiaries and Affiliates  (collectively, “Supplier”).  The term “Affiliates” shall mean those entities controlled  by, which control or which are under common control with an identified named Party.  Such entity shall be deemed to be an Affiliate only so long as such control exists.  Such control means: (i) direct or indirect ownership or control (now or hereafter) of  more than fifty percent (50%)'
        token_list = [line[start:end]
                      for start, end in nlputils.sent_tokenize(line)]        
        self.assertEqual(token_list,
                         ['This Non-Disclosure Agreement (“Agreement”), effective as of the last signature  date below, (“Effective Date”), is by and between Partner 4, LLC, a Virginia Corporation  having its headquarters located at 999 Parkview Drive, West Church, VA 22099, on  behalf of itself, its subsidiaries and Affiliates, (collectively, “P4”), and Box, Inc., a  Delaware Corporation having its headquarters located at 900 Jefferson Ave,  Redwood City, CA 94063, on behalf of itself and its subsidiaries and Affiliates  (collectively, “Supplier”).',
                          'The term “Affiliates” shall mean those entities controlled  by, which control or which are under common control with an identified named Party.',
                          'Such entity shall be deemed to be an Affiliate only so long as such control exists.',
                          'Such control means: (i) direct or indirect ownership or control (now or hereafter) of  more than fifty percent (50%)'])


    def test_tokenize(self):

        line = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).  James'
        self.assertEqual(nlputils.tokenize(line),
                         ['Good',
                          'muffins',
                          'cost',
                          '$',
                          '3.88',
                          'in',
                          'New',
                          '(',
                          'York',
                          ')',
                          '.',
                          'Please',
                          '(',
                          'buy',
                          ')',
                          'me',
                          'two',
                          'of',
                          'them.',
                          '(',
                          'Thanks',
                          ')',
                          '.',
                          'James'])

    def test_span_tokenize(self):

        line = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).  James'
        token_list = [line[start:end]
                      for start, end in nlputils.span_tokenize(line)]        
        self.assertEqual(token_list,
                         ['Good',
                          'muffins',
                          'cost',
                          '$',
                          '3.88',
                          'in',
                          'New',
                          '(',
                          'York',
                          ')',
                          '.',
                          'Please',
                          '(',
                          'buy',
                          ')',
                          'me',
                          'two',
                          'of',
                          'them.',
                          '(',
                          'Thanks',
                          ')',
                          '.',
                          'James'])

    def test_text_tokenize(self):
                
        line = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).  James'
        self.assertEqual(nlputils.text_tokenize(line),
                         ['Good',
                          'muffins',
                          'cost',
                          '$',
                          '3.88',
                          'in',
                          'New',
                          '(',
                          'York',
                          ')',
                          '.',
                          'Please',
                          '(',
                          'buy',
                          ')',
                          'me',
                          'two',
                          'of',
                          'them',
                          '.',
                          '(',
                          'Thanks',
                          ')',
                          '.',
                          'James'])


    def test_text_span_tokenize(self):
                
        line = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).  James'
        token_list = [line[start:end]
                      for start, end in nlputils.text_span_tokenize(line)]
        self.assertEqual(token_list,
                         ['Good',
                          'muffins',
                          'cost',
                          '$',
                          '3.88',
                          'in',
                          'New',
                          '(',
                          'York',
                          ')',
                          '.',
                          'Please',
                          '(',
                          'buy',
                          ')',
                          'me',
                          'two',
                          'of',
                          'them',
                          '.',
                          '(',
                          'Thanks',
                          ')',
                          '.',
                          'James'])

    def test_word_punct_tokenize(self):

        line = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).  James'
        self.assertEqual(nlputils.word_punct_tokenize(line),
                         ['Good',
                          'muffins',
                          'cost',
                          '$',
                          '3',
                          '.',
                          '88',
                          'in',
                          'New',
                          '(',
                          'York',
                          ').',
                          'Please',
                          '(',
                          'buy',
                          ')',
                          'me',
                          'two',
                          'of',
                          'them',
                          '.',
                          '(',
                          'Thanks',
                          ').',
                          'James'])

if __name__ == "__main__":
    unittest.main()

