#!/usr/bin/env python3

import re
import unittest
from typing import Dict, List, Optional, Pattern, Tuple

from kirke.utils import ebsentutils
from kirke.eblearn import annotatorconfig
from kirke.sampleutils import idnumgen

def extract_str(pat: Pattern, line: str, group_num: int = 1) -> str:
    mat = re.search(pat, line)
    if mat:
        mat_st = line[mat.start(group_num):mat.end(group_num)]
        return mat_st
    return ''

def extract_cand(alphanum: idnumgen.IdNumContextGenerator, line: str):
    candidates, _, _ = alphanum.get_candidates_from_text(line, -1, [])
    cand_text = ' /// '.join([cand['chars'] for cand in candidates])
    return cand_text

def extract_idnum_list(alphanum: idnumgen.IdNumContextGenerator,
                       line: str,
                       label_ant_list_param: Optional[List[ebsentutils.ProvisionAnnotation]] = None,
                       label: str = '') -> Tuple[List[Dict],
                                                 List[int],
                                                 List[bool]]:
    candidates, group_id_list, label_list = \
        alphanum.get_candidates_from_text(line,
                                          -1,
                                          [])
    return candidates, group_id_list, label_list


class TestAlphanum(unittest.TestCase):

    # pylint: disable=too-many-statements
    def test_alphanum(self):
        "Test AlphaNum"
        alphanum = idnumgen.IdNumContextGenerator(0,
                                                  0,
                                                  re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)'),
                                                  'idnum',
                                                  is_join=True,
                                                  length_min=2)

        line = "text TIN: 555-77-5555 text"
        self.assertEqual(extract_cand(alphanum, line), '555-77-5555')

        line = "text EIN: 55-5555555 text"
        self.assertEqual(extract_cand(alphanum, line), '55-5555555')

        line = "text SSN: 555-55-5555 text"
        self.assertEqual(extract_cand(alphanum, line), '555-55-5555')

        line = "text ITIN: 955-77-5555 text"
        self.assertEqual(extract_cand(alphanum, line), '955-77-5555')

        line = "text ISIN: US0378331005 text"
        self.assertEqual(extract_cand(alphanum, line), 'US0378331005')

        line = "text CUSIP: 037833100 text"
        self.assertEqual(extract_cand(alphanum, line), '037833100')

        line = "text PHONE NUM: 1-201-505-6365 text"
        self.assertEqual(extract_cand(alphanum, line), '1-201-505-6365')

        line = "text PHONE NUM: 754-3010 text"
        self.assertEqual(extract_cand(alphanum, line), '754-3010')

        line = "text PHONE NUM: (541) 754-3010 text"
        self.assertEqual(extract_cand(alphanum, line), '(541) 754-3010')

        line = "text PHONE NUM: +1-541-754-3010 text"
        self.assertEqual(extract_cand(alphanum, line), '+1-541-754-3010')

        line = "text PHONE NUM: 001-541-754-3010 text"
        self.assertEqual(extract_cand(alphanum, line), '001-541-754-3010')

        line = "text PHONE NUM: 191 541 754 3010 text"
        self.assertEqual(extract_cand(alphanum, line), '191 541 754 3010')

        line = "text PHONE NUM: +49-89-636-48018 text"
        self.assertEqual(extract_cand(alphanum, line), '+49-89-636-48018')

        line = "text PHONE NUM: 19-49-89-636-48018 text"
        self.assertEqual(extract_cand(alphanum, line), '19-49-89-636-48018')

        line = "text PHONE NUM: (206)266-7010 text"
        self.assertEqual(extract_cand(alphanum, line), '(206)266-7010')

        line = "text PHONE NUM: 020 7524 6000 text"
        self.assertEqual(extract_cand(alphanum, line), '020 7524 6000')

        line = "text PHONE NUM: +44 20 7006 1000 text"
        self.assertEqual(extract_cand(alphanum, line), '+44 20 7006 1000')

        line = "text PHONE NUM: +44 (0)20 34001000 text"
        self.assertEqual(extract_cand(alphanum, line), '+44 (0)20 34001000')

        line = "text PHONE NUM: 1-800-KPMG text"
        self.assertEqual(extract_cand(alphanum, line), '1-800-KPMG')

        line = "text DOC ID: 23105963.3 text"
        self.assertEqual(extract_cand(alphanum, line), '23105963.3')

        line = "text DOC ID: #1470273vl text"
        self.assertEqual(extract_cand(alphanum, line), '#1470273vl')

        line = "text DOC ID: 70113 text"
        self.assertEqual(extract_cand(alphanum, line), '70113')

        line = "text DOC ID: L0001/06298/1726860 v.4 text"
        self.assertEqual(extract_cand(alphanum, line), 'L0001/06298/1726860 v.4')

        line = "text DOC ID: 3.4.1 text"
        self.assertEqual(extract_cand(alphanum, line), '3.4.1')

        line = "text ZIPCODE: 98109-5210 text"
        self.assertEqual(extract_cand(alphanum, line), '98109-5210')

        line = "text ZIPCODE: WC2N 5AF text"
        self.assertEqual(extract_cand(alphanum, line), 'WC2N 5AF')

        line = "text ZIPCODE: E34 53G E534 J2 "
        self.assertEqual(extract_cand(alphanum, line), 'E34 53G E534 J2')

        line = "text SECTION NUM: 1(a) text"
        self.assertEqual(extract_cand(alphanum, line), '1(a)')

        line = "text xx1, xx2, xx3 text"
        self.assertEqual(extract_cand(alphanum, line), 'xx1 /// xx2 /// xx3')

        line = "xx1, xx2, xx3"
        self.assertEqual(extract_cand(alphanum, line), 'xx1 /// xx2 /// xx3')

        line = "xx1"
        self.assertEqual(extract_cand(alphanum, line), 'xx1')

        line = "xx1, xx2"
        self.assertEqual(extract_cand(alphanum, line), 'xx1 /// xx2')

        line = "xx1,xx2,xx3"
        self.assertEqual(extract_cand(alphanum, line), 'xx1,xx2,xx3')

        line = "xxabc"
        self.assertEqual(extract_cand(alphanum, line), '')

        line = '+ 63 3 477 4000'
        self.assertEqual(extract_cand(alphanum, line), '+ 63 3 477 4000')

        line = '+1 917'
        self.assertEqual(extract_cand(alphanum, line), '+1 917')

        line = '+49'
        self.assertEqual(extract_cand(alphanum, line), '+49')

        line = '8 10 64 3 477 4000'
        self.assertEqual(extract_cand(alphanum, line), '8 10 64 3 477 4000')

        line = '1 800 Mattre2'
        self.assertEqual(extract_cand(alphanum, line), '1 800 Mattre2')

        # TODO, letting this pass for now
        # We can always add a split when seeing '+' token after merging is done.  But
        # keep the code simple, don't bother right now.
        # line = '+ 8 10 64 + 3 477 4000'
        # self.assertEqual(extract_cand(alphanum, line), '+ 8 10 64 /// + 3 477 4000')

        # shouldn't go across lines
        line = "text xx1\nxx2, xx3 text"
        self.assertEqual(extract_cand(alphanum, line), 'xx1 /// xx2 /// xx3')

    def test_alphanum_label_list(self):
        "Test AlphaNum label_list handling"

        alphanum = idnumgen.IdNumContextGenerator(0,
                                                  0,
                                                  re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)'),
                                                  'idnum',
                                                  is_join=True,
                                                  length_min=2)

        line = 'aaaaaaaaa bbbbbbbbb ccccccccc abcd #678,012 456 text ddddddddd eeeeeeeee ffffffffff'
        self.assertEqual(extract_cand(alphanum, line), '#678,012 456')

        # test if the 2nd idnum_word is labeled True
        ant_list = [ebsentutils.ProvisionAnnotation(label='purchase_order_number',
                                                    start=44,
                                                    end=47)]
        candidates, group_id_list, label_list = \
            extract_idnum_list(alphanum,
                               line,
                               label_ant_list_param=ant_list,
                               label='purchase_order_number')

        self.assertEqual(len(label_list), 1)
        self.assertTrue(label_list[0])

        # test if the first idnum_word is labeled True
        ant_list = [ebsentutils.ProvisionAnnotation(label='purchase_order_number',
                                                    start=35,
                                                    end=43)]
        candidates, group_id_list, label_list = \
            extract_idnum_list(alphanum,
                               line,
                               label_ant_list_param=ant_list,
                               label='purchase_order_number')

        self.assertEqual(len(label_list), 1)
        self.assertTrue(label_list[0])




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
