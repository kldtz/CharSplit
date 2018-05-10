#!/usr/bin/env python3

import unittest

from kirke.utils import nlputils

class TestNLPUtils(unittest.TestCase):

    def test_word_comma_tokenizer(self):
        line = '1) aba Bd (b) a2df.'
        se_tok_list = list(nlputils.word_comma_tokenize(line))
        self.assertEqual(se_tok_list, [(0, 1, '1'), (3, 6, 'aba'), (7, 9, 'Bd'), (11, 12, 'b'), (14, 18, 'a2df')])

        line = 'I.B.M. and Dell Inc., are in a war, battle, and cold-war.'
        se_tok_list = list(nlputils.word_comma_tokenize(line))
        self.assertEqual(se_tok_list, [(0, 6, 'I.B.M.'), (7, 10, 'and'), (11, 15, 'Dell'),
                                       (16, 20, 'Inc.'), (20, 21, ','), (22, 25, 'are'),
                                       (26, 28, 'in'), (29, 30, 'a'), (31, 34, 'war'),
                                       (34, 35, ','), (36, 42, 'battle'), (42, 43, ','),
                                       (44, 47, 'and'), (48, 52, 'cold'), (53, 56, 'war')])

    def test_get_suffix_mat_list(self):
        line = 'Volkswagen Bank GmbH, a company incorporated under'
        mat_list = nlputils.get_org_suffix_mat_list(line)
        st_list = [line[mat.start():mat.end()] for mat in mat_list]
        self.assertEqual(st_list,
                         ['GmbH'])

        line = 'Volkswagen Bank GmbH, a company incorporated under mgm from I.B.M. Corp.'
        mat_list = nlputils.get_org_suffix_mat_list(line)
        st_list = [line[mat.start():mat.end()] for mat in mat_list]
        self.assertEqual(st_list,
                         ['GmbH', 'Corp.'])

        line = 'visiting xxx Group, Ltd.'
        mat_list = nlputils.get_org_suffix_mat_list(line)
        st_list = [line[mat.start():mat.end()] for mat in mat_list]
        self.assertEqual(st_list,
                         ['Ltd.'])

        line = 'Citibank Bank, N.A. is a bank incorporated under New York Law'
        se_term_list = nlputils.find_known_terms(line)
        tag_list = []
        st_list = []
        for se_term in se_term_list:
            start, end, ttype = se_term
            st_list.append(line[start:end])
            tag_list.append(ttype)
        self.assertEqual(st_list,
                         ['N.A.'])
        self.assertEqual(tag_list,
                         ['xORGP'])


        line = 'Citibank Bank, N.A. is a Bank incorporated under New York Law'
        se_term_list = nlputils.find_known_terms(line)
        tag_list = []
        st_list = []
        for se_term in se_term_list:
            start, end, ttype = se_term
            st_list.append(line[start:end])
            tag_list.append(ttype)
        self.assertEqual(st_list,
                         ['N.A.', 'incorporated'])
        self.assertEqual(tag_list,
                         ['xORGP', 'xORGP'])



    def test_is_org_suffix(self):
        line = 'pic'
        self.assertTrue(nlputils.is_org_suffix(line))

        line = 'n.a.'
        self.assertTrue(nlputils.is_org_suffix(line))

        line = 'n.a'
        self.assertTrue(nlputils.is_org_suffix(line))

        line = 'Limited'
        self.assertTrue(nlputils.is_org_suffix(line))

        line = 'limited'
        self.assertTrue(nlputils.is_org_suffix(line))

        line = 'limited.'
        self.assertFalse(nlputils.is_org_suffix(line))

        line = 'Limited.'
        self.assertFalse(nlputils.is_org_suffix(line))



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
        st_list = []
        se_list = nlputils.sent_tokenize(line)
        for start, end in se_list:
            st_list.append(line[start:end])
        # Note:  'them.' is not a sentence separator because of '('
        # Make this work will cause other issue, such as "...Inc.  (the purchase)"
        self.assertEqual(st_list,
                         ['Good muffins cost $3.88\nin New (York).',
                          'Please (buy) me\ntwo of them.\n(Thanks).',
                          'James'])

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
                          'them.',
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
                          'them.',
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


    def test_extract_orgs_term_offset(self):
        line = 'The Princeton Review, Inc. (the “Issuer”), '
        phrased_sent = nlputils.PhrasedSent(line, is_chopped=True)
        parties_term_offset = phrased_sent.extract_orgs_term_offset()
        st_list = []
        if parties_term_offset:
            parties_offset, term_offset = parties_term_offset
            for party_offset in parties_offset:
                start, end = party_offset
                st_list.append(line[start:end])
            if term_offset:
                st_list.append(line[term_offset[0]:term_offset[1]])
        #for i, astr in enumerate(st_list):
        #    print("party #{}\t[{}]".format(i, astr))
        self.assertEqual(st_list,
                         ['The Princeton Review, Inc.', 'the “Issuer”'])

if __name__ == "__main__":
    unittest.main()

