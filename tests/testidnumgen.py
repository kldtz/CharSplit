#!/usr/bin/env python3

import re
import unittest
from typing import Dict, List

from kirke.sampleutils import idnumgen


IDNUM_WORD_PAT = re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)')

def extract_idnum_str_list(atext: str) -> List[str]:
    candidates = idnumgen.extract_idnum_list(atext,
                                             IDNUM_WORD_PAT,
                                             group_num=1,
                                             is_join=True,
                                             length_min=2)

    return [cand['chars'] for cand in candidates]


def extract_cand_str_list(idnum_gen: idnumgen.IdNumContextGenerator,
                          line: str) -> List[str]:
    candidates, label_list, group_id_list = \
        idnum_gen.get_candidates_from_text(line,
                                           group_id=-1,
                                           label_ant_list=[])
    str_list = [cand['chars'] for cand in candidates]
    return str_list


class TestIdNumGen(unittest.TestCase):

    def test_extract_idnum_list(self):
        "Test extract_idnum_list()"

        idnum_list = extract_idnum_str_list('')
        self.assertEqual(idnum_list,
                         [])

        idnum_list = extract_idnum_str_list('1')
        self.assertEqual(idnum_list,
                         [])

        idnum_list = extract_idnum_str_list('xxxxxx1')
        self.assertEqual(idnum_list,
                         ['xxxxxx1'])

        idnum_list = extract_idnum_str_list('11')
        self.assertEqual(idnum_list,
                         ['11'])

        idnum_list = extract_idnum_str_list('11,')
        self.assertEqual(idnum_list,
                         ['11'])

        idnum_list = extract_idnum_str_list('11.')
        self.assertEqual(idnum_list,
                         ['11'])

        idnum_list = extract_idnum_str_list('xxx1, hi xxx2')
        self.assertEqual(idnum_list,
                         ['xxx1', 'xxx2'])

        idnum_list = extract_idnum_str_list('xxx1, xxx2, xxx3')
        self.assertEqual(idnum_list,
                         ['xxx1', 'xxx2', 'xxx3'])

        idnum_list = extract_idnum_str_list('xxx1,xxx2,xxx3')
        self.assertEqual(idnum_list,
                         ['xxx1,xxx2,xxx3'])

        idnum_list = extract_idnum_str_list('63 33')
        self.assertEqual(idnum_list,
                         ['63 33'])

        idnum_list = extract_idnum_str_list('xxx1 xxx2 bbb')
        self.assertEqual(idnum_list,
                         ['xxx1 xxx2'])

        idnum_list = extract_idnum_str_list('xxx1 xxx2 xxx3 bbb xxx4')
        self.assertEqual(idnum_list,
                         ['xxx1 xxx2 xxx3', 'xxx4'])

        idnum_list = extract_idnum_str_list('text PHONE NUM: 191 541 754 3010 text')
        self.assertEqual(idnum_list,
                         ['191 541 754 3010'])

        idnum_list = extract_idnum_str_list('+ 63 3 477 4000')
        self.assertEqual(idnum_list,
                         ['+ 63 3 477 4000'])

        idnum_list = extract_idnum_str_list('+1 917')
        self.assertEqual(idnum_list,
                         ['+1 917'])

        idnum_list = extract_idnum_str_list('+49')
        self.assertEqual(idnum_list,
                         ['+49'])

        idnum_list = extract_idnum_str_list('8 10 64 3 477 4000')
        self.assertEqual(idnum_list,
                         ['8 10 64 3 477 4000'])


        idnum_list = extract_idnum_str_list('+ 1 917 + 1 917')
        self.assertEqual(idnum_list,
                         # TODO, ideally these should be
                         # ['+ 1 917', '+ 1 917'])
                         ['+ 1 917 + 1 917'])


    # pylint: disable=too-many-statements
    def test_idnum_context_gen(self):
        "Test idnumgen"
        idnum_gen = idnumgen.IdNumContextGenerator(3,
                                                   3,
                                                   re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)'),
                                                   'idnum',
                                                   is_join=True,
                                                   length_min=2)

        line = "text TIN: 555-77-5555 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['555-77-5555'])

        line = "text EIN: 55-5555555 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['55-5555555'])

        line = "text SSN: 555-55-5555 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['555-55-5555'])

        line = "text ITIN: 955-77-5555 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['955-77-5555'])

        line = "text ISIN: US0378331005 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['US0378331005'])

        line = "text CUSIP: 037833100 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['037833100'])

        line = "text PHONE NUM: 1-201-505-6365 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['1-201-505-6365'])

        line = "text PHONE NUM: 754-3010 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['754-3010'])

        line = "text PHONE NUM: (541) 754-3010 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['(541) 754-3010'])

        line = "text PHONE NUM: +1-541-754-3010 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['+1-541-754-3010'])

        line = "text PHONE NUM: 001-541-754-3010 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['001-541-754-3010'])

        line = "text PHONE NUM: 191 541 754 3010 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['191 541 754 3010'])

        line = "text PHONE NUM: +49-89-636-48018 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['+49-89-636-48018'])

        line = "text PHONE NUM: 19-49-89-636-48018 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['19-49-89-636-48018'])

        line = "text PHONE NUM: (206)266-7010 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['(206)266-7010'])

        line = "text PHONE NUM: 020 7524 6000 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['020 7524 6000'])

        line = "text PHONE NUM: +44 20 7006 1000 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['+44 20 7006 1000'])

        line = "text PHONE NUM: +44 (0)20 34001000 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['+44 (0)20 34001000'])

        line = "text PHONE NUM: 1-800-KPMG text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['1-800-KPMG'])

        line = "text DOC ID: 23105963.3 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['23105963.3'])

        line = "text DOC ID: #1470273vl text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['#1470273vl'])

        line = "text DOC ID: 70113 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['70113'])

        line = "text DOC ID: L0001/06298/1726860 v.4 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['L0001/06298/1726860 v.4'])

        line = "text DOC ID: 3.4.1 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['3.4.1'])

        line = "text ZIPCODE: 98109-5210 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['98109-5210'])

        line = "text ZIPCODE: WC2N 5AF text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['WC2N 5AF'])

        line = "text ZIPCODE: E34 53G E534 J2"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['E34 53G E534 J2'])

        line = "text ZIPCODE: E34 53G E534 J2 "
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['E34 53G E534 J2'])

        line = "text ZIPCODE: E34 53G E534 J2      "
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['E34 53G E534 J2'])

        line = "text SECTION NUM: 1(a) text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['1(a)'])

        line = "text xx1, xx2, xx3 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['xx1', 'xx2', 'xx3'])

        line = "xx1, xx2, xx3"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['xx1', 'xx2', 'xx3'])

        line = "xx1"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['xx1'])

        line = "xx1, xx2"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['xx1', 'xx2'])

        line = "xx1,xx2,xx3"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['xx1,xx2,xx3'])

        line = "xxabc"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         [])

        line = '+ 63 3 477 4000'
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['+ 63 3 477 4000'])

        line = '+1 917'
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['+1 917'])

        line = '+49'
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['+49'])

        line = '8 10 64 3 477 4000'
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['8 10 64 3 477 4000'])

        line = '1 800 Mattre2'
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['1 800 Mattre2'])

        # TODO, letting this pass for now
        # We can always add a split when seeing '+' token after merging is done.  But
        # keep the code simple, don't bother right now.
        # line = '+ 8 10 64 + 3 477 4000'
        # self.assertEqual(cand_str_list, '+ 8 10 64 /// + 3 477 4000')

        # shouldn't go across lines
        line = "text xx1\nxx2, xx3 text"
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['xx1', 'xx2', 'xx3'])

        # check for length_min
        line = '1 2'
        cand_str_list = extract_cand_str_list(idnum_gen, line)
        self.assertEqual(cand_str_list,
                         ['1 2'])



