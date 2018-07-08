#!/usr/bin/env python3

import unittest

from kirke.utils.alignedstr import AlignedStrMapper


class TestAlingedStr(unittest.TestCase):

    def test_aligned_str(self):
        "Test AlignedStr()"

        line1 = 'Hi     Mary'
        line2 = 'Hi Mary'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.get_to_offset(9),
                         5)
        self.assertTrue(as_mapper.is_fully_synced)

        # test the reverse
        as_mapper = AlignedStrMapper(line2, line1)
        self.assertEqual(as_mapper.get_to_offset(5),
                         9)
        self.assertTrue(as_mapper.is_fully_synced)

        line1 = 'Hi     Mary  '
        line2 = 'Hi Mary          '
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.get_to_offset(9),
                         5)
        self.assertTrue(as_mapper.is_fully_synced)

        as_mapper = AlignedStrMapper(line2, line1)
        self.assertEqual(as_mapper.get_to_offset(5),
                         9)
        self.assertTrue(as_mapper.is_fully_synced)

        line1 = 'LAND REGISTRY PRESCRIBED LEASE CLAUSES ________________________________I'
        line2 = 'LAND REGISTRY PRESCRIBED LEASE CLAUSES _I'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 40), (71, 72)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 40), (40, 41)])

        line1 = '2.    THE LETTING TERMS_2'
        line2 = '2. THE LETTING TERMS__________________________________________________'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 3), (6, 24)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 3), (3, 21)])


        line1 = '2.    THE LETTING TERMS_234'
        line2 = '2. THE LETTING TERMS__________________________________________________'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 3), (6, 24)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 3), (3, 21)])

        # now check the reverse
        as_mapper = AlignedStrMapper(line2, line1)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 3), (3, 21)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 3), (6, 24)])


    def test_failed_aligned_str(self):

        line1 = 'I'
        line2 = 'I166'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 1)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 1)])
        self.assertIsNone(as_mapper.extra_fse)
        self.assertEqual(as_mapper.extra_tse,
                         (1, 4))

        as_mapper = AlignedStrMapper(line2, line1)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 1)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 1)])
        self.assertEqual(as_mapper.extra_fse,
                         (1, 4))
        self.assertIsNone(as_mapper.extra_tse)

        line1 = 'Hi  Mary.'
        line2 = 'Hi Mary'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 3), (4, 8)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 3), (3, 7)])
        self.assertEqual(as_mapper.extra_fse,
                         (8, 9))
        self.assertIsNone(as_mapper.extra_tse)

        as_mapper = AlignedStrMapper(line2, line1)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 3), (3, 7)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 3), (4, 8)])
        self.assertIsNone(as_mapper.extra_fse)
        self.assertEqual(as_mapper.extra_tse,
                         (8, 9))

        line1 = 'Hi  Mary_'
        line2 = 'Hi Mary'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 3), (4, 8)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 3), (3, 7)])
        self.assertEqual(as_mapper.extra_fse,
                         (8, 9))
        self.assertIsNone(as_mapper.extra_tse)

        line1 = 'Hi  Mary__'
        line2 = 'Hi Mary_'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 3), (4, 9)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 3), (3, 8)])
        self.assertIsNone(as_mapper.extra_fse)
        self.assertIsNone(as_mapper.extra_tse)

        line1 = 'Hi John'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertFalse(as_mapper.is_aligned)

        line1 = 'xHi John'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertFalse(as_mapper.is_aligned)

    def test_hyphen(self):
        line1 = 'Dept. # Dept. Name_Account# Account Description'
        line2 = 'Dept. # Dept. Name Account# Account Description'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 18), (19, 47)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 18), (19, 47)])
        self.assertEqual(as_mapper.extra_fse,
                         None)
        self.assertEqual(as_mapper.extra_tse,
                         None)

        line1 = 'Dept. # Dept. Name__Account# Account Description'
        line2 = 'Dept. # Dept. Name_Account# Account Description'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 19), (20, 48)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 19), (19, 47)])
        self.assertEqual(as_mapper.extra_fse,
                         None)
        self.assertEqual(as_mapper.extra_tse,
                         None)

        line1 = 'Dept. # Dept. Name__Account# Account Description'
        line2 = 'Dept. # Dept. Name_   Account# Account Description'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.from_se_list,
                         [(0, 19), (20, 48)])
        self.assertEqual(as_mapper.to_se_list,
                         [(0, 19), (22, 50)])
        self.assertEqual(as_mapper.extra_fse,
                         None)
        self.assertEqual(as_mapper.extra_tse,
                         None)

    def test_extra_fse_tse(self):
        line1 = '2.    THE LETTING TERMS_2'
        line2 = '2. THE LETTING TERMS__________________________________________________'
        as_mapper = AlignedStrMapper(line1, line2)
        self.assertEqual(as_mapper.extra_fse,
                         (24, 25))
        self.assertIsNone(as_mapper.extra_tse,
                          None)

        # now check the reverse
        as_mapper = AlignedStrMapper(line2, line1)
        self.assertIsNone(as_mapper.extra_fse)
        self.assertEqual(as_mapper.extra_tse,
                         (24, 25))
