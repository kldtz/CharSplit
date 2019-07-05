#!/usr/bin/env python3

import math
import unittest
from typing import Dict, List, Optional, Pattern, Tuple

from kirke.utils.text2int import text2number, extract_numbers, extract_number, extract_numbers_in_words
from kirke.utils.text2int import normalize_comma_period

class TestText2Int(unittest.TestCase):

    def test_text2number(self):
        "Test text2number()"

        line = "33"
        self.assertEqual(text2number(line), 33)

        line = ''
        self.assertEqual(text2number(line), -1)

        line = 'one hundred thirty five'
        self.assertEqual(text2number(line), 135)

        line = '1 hundred thirty five'
        self.assertEqual(text2number(line), 135)

        line = 'three billion and thirty five'
        self.assertEqual(text2number(line), 3000000035)

        line = '3 billion and thirty five'
        self.assertEqual(text2number(line), 3000000035)

        line = 'twelve million one hundred forty four thousand and sixty-six'
        self.assertEqual(text2number(line), 12144066)

        line = 'twelve million one hundred forty four thousand and Sixty-Six'
        self.assertEqual(text2number(line), 12144066)

        line = 'twelve million one hundred 44 thousand and 66'
        self.assertEqual(text2number(line), 12144066)

        line = 'twelve million 144 thousand and 66'
        self.assertEqual(text2number(line), 12144066)

        line = '12 million and 44 thousand and 66'
        self.assertEqual(text2number(line), 12044066)


    def test_normalize_comma_period(self):
        "Test normalize_comma_period() with comma"

        line = "33,000,000"
        self.assertEqual('33000000', normalize_comma_period(line))

        line = "33,0"
        self.assertEqual('33.0', normalize_comma_period(line))

        line = "33,12345"
        self.assertEqual('33.12345', normalize_comma_period(line))

        line = "33.12345"
        self.assertEqual('33.12345', normalize_comma_period(line))

        line = "33,123"
        self.assertEqual('33123', normalize_comma_period(line))

        line = "33,123.45"
        self.assertEqual('33123.45', normalize_comma_period(line))

        line = "33.123,45"
        self.assertEqual('33123.45', normalize_comma_period(line))

        line = '10.000.000'
        self.assertEqual('10000000', normalize_comma_period(line))

        line = '1.000'
        self.assertEqual('1000', normalize_comma_period(line))

        line = '1.000.000'
        self.assertEqual('1000000', normalize_comma_period(line))

        line = '100.123'
        self.assertEqual('100.123', normalize_comma_period(line))


    def test_text2number_comma(self):
        "Test text2number() with comma"

        line = "33,000,000"
        self.assertEqual(text2number(line), 33000000)

        line = "33,000,000.0"
        self.assertEqual(text2number(line), 33000000.0)

        line = "33,3 m"
        self.assertTrue(math.isclose(text2number(line), 33300000))

        line = "33,32 m"
        self.assertEqual(text2number(line), 33320000)

        line = "33.3 m"
        self.assertTrue(math.isclose(text2number(line), 33300000))

        line = "33.32 m"
        self.assertTrue(math.isclose(text2number(line), 33320000))

        line = "33.323 m"
        self.assertEqual(text2number(line), 33323000)

        line = "33,323 m"
        self.assertEqual(text2number(line), 33323000000)

        line = "1,401"
        self.assertEqual(text2number(line), 1401)

        line = "1.401"
        self.assertEqual(text2number(line), 1.401)

        line = "14,01"
        self.assertEqual(text2number(line), 14.01)

        line = "1101.401"
        self.assertEqual(text2number(line), 1101.401)

        line = "1,101.401"
        self.assertEqual(text2number(line), 1101.401)

        line = "1.101,401"
        self.assertEqual(text2number(line), 1101.401)


    def test_text2number_acronym(self):
        "Test text2number() with acronym"

        line = "33 m"
        self.assertEqual(text2number(line), 33000000)

        line = "33 M"
        self.assertEqual(text2number(line), 33000000)

        line = "33.5 M"
        self.assertEqual(text2number(line), 33500000)

        line = "1.2 b"
        self.assertEqual(text2number(line), 1200000000)

        line = "1.2 B"
        self.assertEqual(text2number(line), 1200000000)

        line = "0.2 B"
        self.assertEqual(text2number(line), 200000000)

        line = ".2 B"
        self.assertEqual(text2number(line), 200000000)

        line = "1.2 t"
        self.assertEqual(text2number(line), 1200000000000)

        line = "1.2 T"
        self.assertEqual(text2number(line), 1200000000000)

        line = "33M"
        self.assertEqual(text2number(line), 33000000)

        line = "1.2B"
        self.assertEqual(text2number(line), 1200000000)

        line = "1.2T"
        self.assertEqual(text2number(line), 1200000000000)


    def test_extract_numbers_comma(self):
        "Test extract_numbers() with acronym"

        line = "33,3 m"
        self.assertTrue(math.isclose(text2number(line), 33300000))

        line = "33,32 m"
        self.assertEqual(text2number(line), 33320000)


    def test_float(self):
        "Test text2number(), floating point parsing"

        line = 'one point one'
        self.assertEqual(text2number(line), 1.1)

        line = 'zero point two'
        self.assertEqual(text2number(line), 0.2)

        line = 'zero point two three'
        self.assertEqual(text2number(line), 0.23)

        line = 'five point four'
        self.assertEqual(text2number(line), 5.4)

        line = 'five point four three'
        self.assertTrue(math.isclose(text2number(line), 5.43))

        line = 'five point four three one'
        self.assertTrue(math.isclose(text2number(line), 5.431))

        line = 'five point four three zero'
        self.assertTrue(math.isclose(text2number(line), 5.43))

        line = 'seventeen point four'
        # self.assertTrue(math.isclose(text2number(line), 17.4))
        self.assertEqual(text2number(line), 17.4)

        line = 'twenty-one point four'
        self.assertEqual(text2number(line), 21.4)

        line = 'fifty one point four'
        self.assertEqual(text2number(line), 51.4)

        line = 'one hundred one point four'
        self.assertEqual(text2number(line), 101.4)

        line = 'one thousand one point four'
        self.assertEqual(text2number(line), 1001.4)

        # line = 'five point thirty three'
        # print('[{}] => {}'.format(st, text2number(st)))

        line = 'thirty three'
        self.assertEqual(text2number(line), 33)

        line = 'thirty three point three three'
        self.assertEqual(text2number(line), 33.33)

        line = '.3'
        self.assertEqual(text2number(line), 0.3)


    def test_extract_numbers(self):
        "Test extract_numbers()"

        line = 'I found three billion and thirty five dollars'
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 3000000035)


        # a typo in the input document
        line = '$11,000,000  of General Liability Insurance ($1,000,000 base + $10,000,00 umbrella)  covering:'
        adict_list = extract_numbers(line)
        # self.assertEqual(len(adict_list), 3)
        self.assertEqual(len(adict_list), 0)
        """
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 11000000)
        adict = adict_list[1]
        self.assertEqual(adict['norm']['value'], 1000000)
        adict = adict_list[2]
        self.assertEqual(adict['norm']['value'], 1000000)
        """

        line = "one and half pound and three and half pound, eight and half dollars three and half million dollars"
        adict_list = extract_numbers(line)
        self.assertEqual(len(adict_list), 4)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 1.5)
        adict = adict_list[1]
        self.assertEqual(adict['norm']['value'], 3.5)
        adict = adict_list[2]
        self.assertEqual(adict['norm']['value'], 8.5)
        adict = adict_list[3]
        self.assertEqual(adict['norm']['value'], 3500000)


    def test_extract_number(self):
        "Test extract_numbers()"

        line = 'three billion and thirty five'
        adict = extract_number(line)
        self.assertEqual(adict['norm']['value'], 3000000035)


    def test_extract_nunmber_in_words(self):
        "Test extract_numbers_in_words()"

        line = ''
        adict_list = extract_numbers_in_words(line)
        print('adict_list: {}'.format(adict_list))
        self.assertEqual(len(adict_list), 0)


        line = 'one hundred thirty five'
        adict_list = extract_numbers_in_words(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 135)

        line = 'ten apples'
        adict_list = extract_numbers_in_words(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 10)

        line = 'three billion and thirty five'
        adict_list = extract_numbers_in_words(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 3000000035)


        line = 'twelve million one hundred forty four thousand and sixty-six'
        adict_list = extract_numbers_in_words(line)
        print("adict_list:")
        print(adict_list)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 12144066)

        line = 'twelve million one hundred forty four thousand and'
        adict_list = extract_numbers_in_words(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 12144000)


    """
    def test_extract_nunmber_list_in_word(self):
        "Test extract_number_list_in_words()"
        "Many are basically years.?"

        line = 'ten ten'
        adict_list = extract_numbers_in_words(line)
        print('adict_list: {}'.format(adict_list))
        self.assertEqual(len(adict_list), 1)
        self.assertEqual(adict['norm']['value'], 1010)

        line = 'twenty fourteen'
        adict_list = extract_numbers_in_words(line)
        print('adict_list: {}'.format(adict_list))
        self.assertEqual(len(adict_list), 1)
        self.assertEqual(adict['norm']['value'], 2014)

        line = 'twenty o three'
        adict_list = extract_numbers_in_words(line)
        print('adict_list: {}'.format(adict_list))
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 2003)

        line = 'nineteen eighty four'
        adict_list = extract_numbers_in_words(line)
        print('adict_list: {}'.format(adict_list))
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 1984)

        line = 'one two three'
        adict_list = extract_numbers_in_words(line)
        print('adict_list: {}'.format(adict_list))
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 123)

        line = 'one hundred thirty five'
        adict_list = extract_numbers_in_words(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 135)


        line = 'ten apples'
        adict_list = extract_numbers_in_words(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 10)

        line = 'three billion and thirty five'
        adict_list = extract_numbers_in_words(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 3000000035)


        line = 'twelve million one hundred forty four thousand and sixty-six'
        adict_list = extract_numbers_in_words(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        self.assertEqual(adict['norm']['value'], 12144066)
    """
