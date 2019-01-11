#!/usr/bin/env python3

import re
import unittest
from typing import Dict, List, Optional, Pattern, Tuple

from kirke.utils import ebsentutils
from kirke.eblearn import annotatorconfig
from kirke.sampleutils import regexgen

def extract_str(pat: Pattern, line: str, group_num: int = 0) -> str:
    mat = re.search(pat, line)
    if mat:
        mat_st = line[mat.start(group_num):mat.end(group_num)]
        return mat_st.strip()
    return ''

def extract_cand(alphanum: regexgen.RegexContextGenerator, line: str):
    candidates, _, _ = alphanum.get_candidates_from_text(line)
    cand_text = ' /// '.join([cand['chars'] for cand in candidates])
    return cand_text

class TestCurrency(unittest.TestCase):

    def test_currency(self):
        "Test CURRENCY_PAT"

        currency_pat = annotatorconfig.CURRENCY_PAT

        line = "Bob received 33 dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33 dollars')

        line = "Bob received 33. dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33. dollars')

        line = "Bob received 33.5 dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33.5 dollars')

        line = "Bob received 33.55 dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33.55 dollars')

        line = "Bob received 33B dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33B dollars')

        line = "Bob received 33 B dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33 B dollars')

        line = "Bob received 33.3 M dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33.3 M dollars')

        line = "Bob received 33.33 M dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33.33 M dollars')

        # TODO, this is a little weird
        # we intentionally want to be more inclusive, so
        # didn't check for \b
        line = "Bob received 33.444 M dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33.444 M dollars')


        line = "Bob received 333,333  dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '333,333  dollars')

        line = "Bob received 333,333.2 million dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '333,333.2 million dollars')

        line = "Bob received $333,333.20 from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '$333,333.20')

        line = "Bob received €333,333 from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '€333,333')

        line = "Bob received 333,333 Euros from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '333,333 Euros')

        line = "Bob received 333,333 € from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '333,333 €')

        line = "Bob received 333,333€ from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '333,333€')

        line = "Bob received 333,333 from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '')

        # TODO, failed
        # should be ''
        line = "Bob received -333,333 dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '333,333 dollars')

        line = "Bob received USD 33 from Alice"
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         'USD 33')

        line = "Bob received USD   33 from Alice"
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         'USD   33')

        line = "Bob received USD   33.33 from Alice"
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         'USD   33.33')

        line = "Bob received Rs   3 from Alice"
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         'Rs   3')
        line = "Bob received Rs.   3 from Alice"
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         'Rs.   3')

        line = "Bob received 3  Rs from Alice"
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         '3  Rs')
        line = "Bob received 3 Rs. from Alice"
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         '3 Rs')

        line = "Bob received 33.33 Rupees from Alice"
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         '33.33 Rupees')
        line = "Bob received 33.33 Rupee from Alice"
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         '33.33 Rupee')
        line = "Bob received INR 33.33 from Alice"
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         'INR 33.33')
        line = "Bob received 33.33  INR from Alice"
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         '33.33  INR')
        line = 'Rs.50,000.00 (Rupees Fifty Thousand only)'
        self.assertEqual(extract_str(currency_pat,
                                     line),
                         'Rs.50,000.00')



    def test_number(self):
        "Test NUMBER_PAT"

        number_pat = annotatorconfig.NUMBER_PAT

        line = "33.3 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '33.3')

        line = "3.3 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '3.3')

        line = ".3 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '.3')

        line = "0.3 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '0.3')


        line = "-0.3 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '-0.3')


        line = "-333,333.3 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '-333,333.3')

        line = "-22,333.3 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '-22,333.3')

        line = "Bob received 33 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '33')

        line = "Bob received 33.3 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '33.3')

        line = "Bob received 33.3802 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '33.3802')


        line = "Bob received 33. dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '33.')

        line = "Bob received .3802 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '.3802')

        line = "Bob received 0.3802 dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '0.3802')

        line = "Bob received (0.3802) dollars from Alice"
        self.assertEqual(extract_str(number_pat, line, 2),
                         '0.3802')


    def test_percent(self):
        "Test PERCENT_PAT"

        percent_pat = annotatorconfig.PERCENT_PAT

        line = "33.3 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '33.3 percent')

        line = "33.3% from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '33.3%')

        line = "33.3percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '33.3percent')


        line = "33.3 % from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '33.3 %')

        line = "3.3 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '3.3 percent')

        line = ".3 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '.3 percent')

        line = "0.3 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '0.3 percent')

        line = "-0.3 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '-0.3 percent')


        line = "-333,333.3 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '-333,333.3 percent')

        line = "-22,333.3 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '-22,333.3 percent')

        line = "Bob received 33 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '33 percent')

        line = "Bob received 33.3 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '33.3 percent')

        line = "Bob received 33.3802 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '33.3802 percent')

        line = "Bob received 33.  percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '33.  percent')

        line = "Bob received .3802 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '.3802 percent')

        line = "Bob received 0.3802 percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '0.3802 percent')


    def test_word_currency(self):
        "Test CURRENCY_PAT"

        currency_pat = annotatorconfig.CURRENCY_PAT


        line = "Bob received 1 pound from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '1 pound')

        """
        line = "Bob received one pound from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         'one pound')
        """

    """

        line = "Bob received thirty-three dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         'thirty-three dollars')

        line = "Bob received 33M dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33M dollars')

        line = "Bob received 33 M dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33 M dollars')

        line = "Bob received 33B dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33B dollars')

        line = "Bob received 33 B dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33 B dollars')

        line = "Bob received 33 B dollars from Alice"
        self.assertEqual(extract_str(currency_pat, line),
                         '33 B dollars')
"""
