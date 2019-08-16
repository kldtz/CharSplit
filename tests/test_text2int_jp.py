#!/usr/bin/env python3

import math
import unittest
from typing import Dict, List, Optional, Pattern, Tuple

from kirke.nlputil.text2int_jp import _text2number, extract_numbers, extract_number
from kirke.nlputil.text2int_jp import extract_fractions, extract_numbers_fractions
from kirke.utils.text2int import extract_roman_numbers


from kirke.utils import unicodeutils

class TestText2IntJP(unittest.TestCase):

    def test_text2number_arabic(self):
        "Test text2number()"

        line = ''
        self.assertEqual(_text2number(line), -1)

        line = '0'
        self.assertEqual(_text2number(line), 0)

        line = "33"
        self.assertEqual(_text2number(line), 33)

        line = '３３'
        line = unicodeutils.normalize_dbcs_sbcs(line)
        self.assertEqual(_text2number(line), 33)

        line = '３３．８'
        line = unicodeutils.normalize_dbcs_sbcs(line)
        self.assertEqual(_text2number(line), 33.8)

        line = '．８'
        line = unicodeutils.normalize_dbcs_sbcs(line)
        self.assertEqual(_text2number(line), 0.8)

        line = '０．８'
        line = unicodeutils.normalize_dbcs_sbcs(line)
        self.assertEqual(_text2number(line), 0.8)


    def test_text2number_words(self):

        line = '一百三十五'
        self.assertEqual(_text2number(line), 135)

        line = '十億零三十五'
        self.assertEqual(_text2number(line), 1000000035)

        line = '十億三十五'
        self.assertEqual(_text2number(line), 1000000035)

        line = '三十億零三十五'
        self.assertEqual(_text2number(line), 3000000035)

        line = '一千二百一十四万四千六十六'
        self.assertEqual(_text2number(line), 12144066)

        line = 'ゼロ'
        self.assertEqual(_text2number(line), 0)


    def test_text2number_comma(self):
        "Test text2number() with comma"

        line = "33,000,000"
        self.assertEqual(_text2number(line), 33000000)

        line = "33,000,000.0"
        self.assertEqual(_text2number(line), 33000000.0)

        line = "1,401"
        self.assertEqual(_text2number(line), 1401)

        line = "1.401"
        self.assertEqual(_text2number(line), 1.401)

        line = "14,01"
        self.assertEqual(_text2number(line), 14.01)

        line = "1101.401"
        self.assertEqual(_text2number(line), 1101.401)

        line = "1,101.401"
        self.assertEqual(_text2number(line), 1101.401)

        line = "1.101,401"
        self.assertEqual(_text2number(line), 1101.401)


    def test_extract_numbers_double_arabic(self):

        line = '２０１６年４月１３日まで'
        adict_list = extract_numbers(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(len(adict_list), 3)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 2016)
        adict = adict_list[1]
        self.assertEqual(adict['norm']['value'], 4)
        adict = adict_list[2]
        self.assertEqual(adict['norm']['value'], 13)


    def test_extract_numbers(self):
        "Test extract_numbers()"

        line = '二'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 2)

        line = '弐'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 2)

        line = '十三'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 13)

        line = '十参'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 13)

        line = '二千零四年四月二十三日まで'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 3)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 2004)
        adict = adict_list[1]
        self.assertEqual(adict['norm']['value'], 4)
        adict = adict_list[2]
        self.assertEqual(adict['norm']['value'], 23)

        line = '個人情報の整理、追加に関する支援保 護法（平成１５年５月３０日法律第５７号。' \
               'その後の改正を締結する。含む月の翌月む月の翌月。'
        adict_list = extract_numbers(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(len(adict_list), 4)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 15)
        adict = adict_list[1]
        self.assertEqual(adict['norm']['value'], 5)
        adict = adict_list[2]
        self.assertEqual(adict['norm']['value'], 30)
        adict = adict_list[3]
        self.assertEqual(adict['norm']['value'], 57)

        line = '二千二十三日まで'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 2023)

        # testing just '13' after 1000
        line = '二千十三日まで'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 2013)

        line = '十億零三十五'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 1000000035)

        line = '一千二百一十四万'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 12140000)

        line = '壱千弐百一十参万'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(12130000, adict['norm']['value'])

        line = '四千六十六'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 4066)

        line = '一千二百一十四'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 1214)

        line = '一千二百一十四万四千六十六'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 12144066)

        # numbers from DFIN's number description
        line = '三十一'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(31, adict['norm']['value'])

        line = '五十四'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(54, adict['norm']['value'])

        line = '七十七'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(77, adict['norm']['value'])

        line = '二十'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(20, adict['norm']['value'])

        line = '一万'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(10000, adict['norm']['value'])

        line = '一億'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(100000000, adict['norm']['value'])

        line = '一兆'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(1000000000000, adict['norm']['value'])

        line = '三百'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(300, adict['norm']['value'])

        line = '八千'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(8000, adict['norm']['value'])

        line = '八千三百'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(8300, adict['norm']['value'])

        line = '八千三百万'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(83000000, adict['norm']['value'])

        line = '四万三千七十六'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(43076, adict['norm']['value'])

        line = '七億六百二十四万九千二百二十二'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(706249222, adict['norm']['value'])

        line = '五百兆二万一'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(500000000020001, adict['norm']['value'])

        line = 'ゼロ'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(0, adict['norm']['value'])

        line = '二千五百六十二万千二百三十四'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(25621234, adict['norm']['value'])


    def test_extract_numbers_cn(self):

        # from https://resources.allsetlearning.com/chinese/grammar/Big_numbers_in_Chinese
        line = '五万两千一百五十二'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(52152, adict['norm']['value'])

        line = '二百九十一万四千六百八十'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(2914680, adict['norm']['value'])

        line = '七百八十九万零二百九十八'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(7890298, adict['norm']['value'])

        line = '两千七百二十一万四千八百九十六'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(27214896, adict['norm']['value'])

        line = '五千三百七十九万八千两百五十'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(53798250, adict['norm']['value'])

        line = '四亿一千四百二十九万四千一百八十二'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(414294182, adict['norm']['value'])

        line = '十三亿两千六百八十万'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(1326800000, adict['norm']['value'])

        line = '两百五十一亿五千八百三十六万七千二百'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(25158367200, adict['norm']['value'])


    def test_extract_numbers_as_digits(self):
        """Test when people read out each number as a digit."""
        line = '二零零四年四月十三日まで'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 3)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 2004)
        adict = adict_list[1]
        self.assertEqual(adict['norm']['value'], 4)
        adict = adict_list[2]
        self.assertEqual(adict['norm']['value'], 13)


    def test_extract_number(self):
        "Test extract_number()"

        line = '一千二百一十四万四千六十六'
        adict = extract_number(line)
        self.assertEqual(adict['norm']['value'], 12144066)

        line = '一千二百一十四万四千六十六日まで'
        adict = extract_number(line)
        self.assertEqual(adict['norm']['value'], 12144066)


    def test_extract_nunmbers_empty(self):
        "Test extract_numbers(), empty"

        line = ''
        adict_list = extract_numbers(line)
        print('adict_list: {}'.format(adict_list))
        self.assertEqual(0, len(adict_list))

        line = '日まで'
        adict_list = extract_numbers(line)
        print('adict_list: {}'.format(adict_list))
        self.assertEqual(0, len(adict_list))


    def test_extract_nunmber_empty(self):
        "Test extract_number(), empty"
        line = ''
        adict = extract_number(line)
        self.assertEqual({}, adict)

        line = '日まで'
        adict = extract_number(line)
        self.assertEqual({}, adict)


    def test_extract_fractions(self):
        """Test extract_fractions()"""

        line = '1/3'
        adict_list = extract_fractions(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        print('adict')
        print(adict)
        self.assertEqual(0.333, round(adict['norm']['value'], 3))

        line = '四分の三'
        adict_list = extract_fractions(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        print('adict')
        print(adict)
        self.assertEqual(0.75, adict['norm']['value'])

        line = '33 1/3'
        adict_list = extract_fractions(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        print('adict')
        print(adict)
        self.assertEqual(33.333, round(adict['norm']['value'], 3))


    def test_extract_numbers_fractions(self):
        """Test extract_numbers_fractions()"""

        line = '14.5 and 四分の三 and 33 1/3'
        adict_list = extract_numbers_fractions(line)
        self.assertEqual(len(adict_list), 3)
        adict = adict_list[0]
        self.assertEqual(14.5, adict['norm']['value'])
        adict = adict_list[1]
        self.assertEqual(0.75, adict['norm']['value'])
        adict = adict_list[2]
        self.assertEqual(33.333, round(adict['norm']['value'], 3))


    def test_extract_roman_numbers(self):

        line = 'i ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi xvii xviii xix xx'
        adict_list = extract_roman_numbers(line)
        self.assertEqual(len(adict_list), 20)
        for inum in range(20):
            adict = adict_list[inum]
            self.assertEqual(inum + 1, adict['norm']['value'])

        line = 'ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫ'
        adict_list = extract_roman_numbers(line)
        self.assertEqual(len(adict_list), 12)
        for inum in range(12):
            adict = adict_list[inum]
            self.assertEqual(inum + 1, adict['norm']['value'])

        line = 'I got three numbers'
        print('\nline: [{}]'.format(line))
        adict_list = extract_roman_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(1, adict['norm']['value'])

    def test_extract_numbers_extra_punct(self):

        # TODO
        # These are special unicode characters.
        # ABBYY probably will never generate those,
        # but a Word file might.  Might be problematic in
        # Japanese docs.
        # ⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛
        # ⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇
        line = '1. 2. 3. 4. 5. 6. 7. 8. 9. ' \
               '10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20.'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 20)
        for inum in range(20):
            adict = adict_list[inum]
            self.assertEqual(inum + 1, adict['norm']['value'])

        line = '(1) (2) (3) (4) (5) (6) (7) (8) (9) ' \
               '(10) (11) (12) (13) (14) (15) (16) (17) (18) (19) (20)'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 20)
        for inum in range(20):
            adict = adict_list[inum]
            self.assertEqual(inum + 1, adict['norm']['value'])

    def test_extract_numbers_float(self):
        # Thanks for Kathy Dunn for finding the examples

        # 42.195 km
        line = '四十二・一九五 キロメートル'
        adict_list = extract_numbers(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(42.195, adict['norm']['value'])


    def test_extract_numbers_with_spaces(self):
        # OCR for Japanese docs sometimes has extra spaces

        line = 'He has 10 0 dollars'
        adict_list = extract_numbers(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(100, adict['norm']['value'])


    def test_parse_numbers_in_fractions(self):
        """We must preserve the '百' in '百分の二', otherwise
           if we invalidate that, then there is no value for
           the fraction.
        """

        line = 'interest reate is 百分の二'
        adict_list = extract_numbers(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(len(adict_list), 2)
        adict = adict_list[0]
        self.assertEqual(100, adict['norm']['value'])
        adict = adict_list[1]
        self.assertEqual(2, adict['norm']['value'])


    def test_extract_numbers_spell_out(self):

        line = '二零零四'
        adict_list = extract_numbers(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(2004, adict['norm']['value'])

        line = '一九五'
        adict_list = extract_numbers(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(195, adict['norm']['value'])

        line = '二四'
        adict_list = extract_numbers(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(24, adict['norm']['value'])


    """
    def test_extract_numbers_mix_arabic_jp(self):

        line = '2,562万1,234'
        adict_list = extract_numbers(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(25621234, adict['norm']['value'])
    """
