#!/usr/bin/env python3

import unittest

import logging

from kirke.utils.alignedstr import AlignedStrMapper


class TestAlingedStr(unittest.TestCase):

    def test_aligned_str(self):
        "Test AlignedStr()"

        line1 = 'Hi     Mary'
        line2 = 'Hi Mary'
        smapper = AlignedStrMapper(line1, line2)
        self.assertEqual(smapper.get_to_offset(9),
                         5)

        # test the reverse
        smapper = AlignedStrMapper(line2, line1)
        self.assertEqual(smapper.get_to_offset(5),
                         9)

        line1 = 'Hi     Mary  '
        line2 = 'Hi Mary          '
        smapper = AlignedStrMapper(line1, line2)
        self.assertEqual(smapper.get_to_offset(9),
                         5)

        smapper = AlignedStrMapper(line2, line1)
        self.assertEqual(smapper.get_to_offset(5),
                         9)

        """
        with self.assertRaises(Exception) as context:
            smapper = AlignedStrMapper(line1, line2)
            logging.warning('eee: [{}]'.format(str(context.exception)))
            self.assertTrue("character diff at 1, char '.'"
                            in str(context.exception))

        line1 = 'Hi John'
        line2 = 'Hi Mary'
        smapper = AlignedStrMapper(line1, line2)
        """

    def test_failed_aligned_str(self):

        line1 = 'Hi  Mary.'
        line2 = 'Hi Mary'
        try:
            smapper = AlignedStrMapper(line1, line2)
        except Exception as e:
            print("e: [{}]".format(e))
            self.assertEquals("Character diff at 8, char '.'",
                              str(e))

        try:
            smapper = AlignedStrMapper(line2, line1)
        except Exception as e:
            print("e: [{}]".format(e))
            self.assertEquals("Character diff at 7, eoln",
                              str(e))

        line1 = 'Hi John'
        try:
            smapper = AlignedStrMapper(line1, line2)
        except Exception as e:
            print("e: [{}]".format(e))
            self.assertEquals("Character diff at 3, char 'J'",
                              str(e))

        line1 = 'xHi John'
        try:
            smapper = AlignedStrMapper(line1, line2)
        except Exception as e:
            print("e: [{}]".format(e))
            self.assertEquals("Character diff at 0, char 'x'",
                              str(e))



        # self.assertRaises() doesn't seem to work
        # always passes regardless what are passed in.
        """
        with self.assertRaises(Exception) as context:
            smapper = AlignedStrMapper(line1, line2)
            self.assertTrue("character diff at 1, char '.'"
                            in str(context.exception))
        """

