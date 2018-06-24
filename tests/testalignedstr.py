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
        self.assertTrue(smapper.is_fully_synced)

        # test the reverse
        smapper = AlignedStrMapper(line2, line1)
        self.assertEqual(smapper.get_to_offset(5),
                         9)
        self.assertTrue(smapper.is_fully_synced)

        line1 = 'Hi     Mary  '
        line2 = 'Hi Mary          '
        smapper = AlignedStrMapper(line1, line2)
        self.assertEqual(smapper.get_to_offset(9),
                         5)
        self.assertTrue(smapper.is_fully_synced)

        smapper = AlignedStrMapper(line2, line1)
        self.assertEqual(smapper.get_to_offset(5),
                         9)
        self.assertTrue(smapper.is_fully_synced)

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

        line1 = 'LAND REGISTRY PRESCRIBED LEASE CLAUSES ________________________________I'
        line2 = 'LAND REGISTRY PRESCRIBED LEASE CLAUSES _I'
        smapper = AlignedStrMapper(line1, line2)
        self.assertEquals(smapper.from_se_list,
                          [(0, 40), (71, 72)])
        self.assertEquals(smapper.to_se_list,
                          [(0, 40), (40, 41)])

        line1 = '2.    THE LETTING TERMS_2'
        line2 = '2. THE LETTING TERMS__________________________________________________'
        smapper = AlignedStrMapper(line1, line2)
        self.assertEquals(smapper.from_se_list,
                          [(0, 3), (6, 24)])
        self.assertEquals(smapper.to_se_list,
                          [(0, 3), (3, 21)])

        # now check the reverse
        smapper = AlignedStrMapper(line2, line1)
        self.assertEquals(smapper.from_se_list,
                          [(0, 3), (3, 21)])
        self.assertEquals(smapper.to_se_list,
                          [(0, 3), (6, 24)])


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


    def test_extra_fse_tse(self):

        line1 = '2.    THE LETTING TERMS_2'
        line2 = '2. THE LETTING TERMS__________________________________________________'
        smapper = AlignedStrMapper(line1, line2)
        self.assertEqual(smapper.extra_fse,
                         (24, 25))
        self.assertIsNone(smapper.extra_tse,
                         None)

        # now check the reverse
        smapper = AlignedStrMapper(line2, line1)
        self.assertIsNone(smapper.extra_fse)
        self.assertEqual(smapper.extra_tse,
                          (24, 25))

