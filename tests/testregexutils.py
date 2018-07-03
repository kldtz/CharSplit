#!/usr/bin/env python3

import unittest
from typing import List

from kirke.utils import regexutils

def find_phrase(line: str, phrase: str) -> List[str]:
    return regexutils.find_phrase(line, phrase)

def find_phrases(line: str, phrases: str) -> List[str]:
    return regexutils.find_phrases(line, phrases)

class TestRegexUtil(unittest.TestCase):

    def test_(self):
        "Test get_all_words_ge_2chars()"

        self.assertEquals(find_phrase('I negotiated a contract with IBM',
                                      'IBM'),
                          ['IBM'])

        self.assertEquals(find_phrases('I negotiated a contract with IBM',
                                       ['IBM']),
                          ['IBM'])
        self.assertEquals(find_phrases('I negotiated a contract with I. B. M.',
                                       ['I.B.M.']),
                          ['I. B. M.'])
        self.assertEquals(find_phrases('I negotiated a contract with IBM',
                                       ['I.B.M.']),
                          ['IBM'])
        self.assertEquals(find_phrases('I negotiated a contract with IBM.',
                                       ['I.B.M.']),
                          ['IBM.'])
        self.assertEquals(find_phrases('I negotiated a contract with IBM Corp.',
                                       ['I.B.M. Corp.', 'I.B.M.']),
                          ['IBM Corp.'])

        self.assertEquals(find_phrases('I negotiated a contract with IBM Corp. and I.B.M. subsidiaries.',
                                       ['I.B.M. Corp.', 'I.B.M.']),
                          ['IBM Corp.', 'I.B.M.'])

        # test the use of sorted() in creating the pattern, ensure longest match first
        self.assertEquals(find_phrases('I negotiated a contract with IBM Corp. and I.B.M. subsidiaries.',
                                       ['I.B.M.', 'I.B.M. Corp.']),
                          ['IBM Corp.', 'I.B.M.'])

if __name__ == "__main__":
    unittest.main()

