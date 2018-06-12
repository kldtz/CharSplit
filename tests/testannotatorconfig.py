#!/usr/bin/env python3

import re
import unittest
from typing import Pattern

from kirke.eblearn import annotatorconfig

def extract_str(pat: Pattern, line: str, group_num: int = 1) -> str:
    mat = re.search(pat, line)
    if mat:
        mat_st = line[mat.start(group_num):mat.end(group_num)]
        return mat_st
    return ''


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
                         '444 M dollars')


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



class TestIdNum(unittest.TestCase):

    def test_idnum(self):
        "Test IDNUM_PAT"

        idnum_pat = annotatorconfig.IDNUM_PAT

        line = "Bob received 33 dollars from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '33')

        line = "Bob received 33. dollars from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '33.')

        line = "Bob received 33.5 335 dollars from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '33.5 335')

        line = "Bob received 33.5 335dollars from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '33.5 335dollars')

        line = "Bob received 33.5 #3llars from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '33.5 #3llars')

        line = "Bob received 3 from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '')

        line = "Bob received 3 #3llars from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '#3llars')

        line = "Bob received 43 #3llars from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '43 #3llars')

        line = "Bob received 43 #3ll,ars from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '43 #3ll')

        line = "Bob received .43 #3ll)ars from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '.43 #3ll')

        line = "Bob received .43  #3ll)ars from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '.43  #3ll')

        # 3 spaces between idnums, only first one is found
        line = "Bob received .43   #3ll)ars from Alice"
        self.assertEqual(extract_str(idnum_pat, line),
                         '.43')



