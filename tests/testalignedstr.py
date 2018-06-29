#!/usr/bin/env python3

import unittest

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

        line1 = 'LAND REGISTRY PRESCRIBED LEASE CLAUSES ________________________________I'
        line2 = 'LAND REGISTRY PRESCRIBED LEASE CLAUSES _I'
        smapper = AlignedStrMapper(line1, line2)
        self.assertEqual(smapper.from_se_list,
                         [(0, 40), (71, 72)])
        self.assertEqual(smapper.to_se_list,
                         [(0, 40), (40, 41)])

        line1 = '2.    THE LETTING TERMS_2'
        line2 = '2. THE LETTING TERMS__________________________________________________'
        smapper = AlignedStrMapper(line1, line2)
        self.assertEqual(smapper.from_se_list,
                         [(0, 3), (6, 24)])
        self.assertEqual(smapper.to_se_list,
                         [(0, 3), (3, 21)])

        # now check the reverse
        smapper = AlignedStrMapper(line2, line1)
        self.assertEqual(smapper.from_se_list,
                         [(0, 3), (3, 21)])
        self.assertEqual(smapper.to_se_list,
                         [(0, 3), (6, 24)])


    def test_failed_aligned_str(self):

        line1 = 'I'
        line2 = 'I166'
        smapper = AlignedStrMapper(line1, line2)
        self.assertEqual(smapper.from_se_list,
                         [(0, 1)])
        self.assertEqual(smapper.to_se_list,
                         [(0, 1)])
        self.assertIsNone(smapper.extra_fse)
        self.assertEqual(smapper.extra_tse,
                         (1, 4))

        smapper = AlignedStrMapper(line2, line1)
        self.assertEqual(smapper.from_se_list,
                         [(0, 1)])
        self.assertEqual(smapper.to_se_list,
                         [(0, 1)])
        self.assertEqual(smapper.extra_fse,
                         (1, 4))
        self.assertIsNone(smapper.extra_tse)

        line1 = 'Hi  Mary.'
        line2 = 'Hi Mary'
        smapper = AlignedStrMapper(line1, line2)
        self.assertEqual(smapper.from_se_list,
                         [(0, 3), (4, 8)])
        self.assertEqual(smapper.to_se_list,
                         [(0, 3), (3, 7)])
        self.assertEqual(smapper.extra_fse,
                         (8, 9))
        self.assertIsNone(smapper.extra_tse)

        smapper = AlignedStrMapper(line2, line1)
        self.assertEqual(smapper.from_se_list,
                         [(0, 3), (3, 7)])
        self.assertEqual(smapper.to_se_list,
                         [(0, 3), (4, 8)])
        self.assertIsNone(smapper.extra_fse)
        self.assertEqual(smapper.extra_tse,
                         (8, 9))

        line1 = 'Hi  Mary_'
        line2 = 'Hi Mary'
        smapper = AlignedStrMapper(line1, line2)
        self.assertEqual(smapper.from_se_list,
                         [(0, 3), (4, 8)])
        self.assertEqual(smapper.to_se_list,
                         [(0, 3), (3, 7)])
        self.assertEqual(smapper.extra_fse,
                         (8, 9))
        self.assertIsNone(smapper.extra_tse)

        line1 = 'Hi  Mary__'
        line2 = 'Hi Mary_'
        smapper = AlignedStrMapper(line1, line2)
        self.assertEqual(smapper.from_se_list,
                         [(0, 3), (4, 9)])
        self.assertEqual(smapper.to_se_list,
                         [(0, 3), (3, 8)])
        self.assertIsNone(smapper.extra_fse)
        self.assertIsNone(smapper.extra_tse)

        line1 = 'Hi John'
        smapper = AlignedStrMapper(line1, line2)
        self.assertFalse(smapper.is_aligned)


        line1 = 'xHi John'
        smapper = AlignedStrMapper(line1, line2)
        self.assertFalse(smapper.is_aligned)

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
