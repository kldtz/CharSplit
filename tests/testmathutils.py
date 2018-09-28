#!/usr/bin/env python3

import unittest

from kirke.utils import mathutils

    
class TestMathUtils(unittest.TestCase):

    def test_pairs_to_sets(self):
        self.assertEquals(mathutils.pairs_to_sets([]), list(set()))
        self.assertEquals(mathutils.pairs_to_sets([(1, 2), (2, 1)]), [{1, 2}])
        self.assertEquals(mathutils.pairs_to_sets([(1, 2), (2, 3)]), [{1, 2, 3}])
        self.assertEquals(mathutils.pairs_to_sets([(1, 2), (3, 4)]), [{1, 2}, {3, 4}])
        self.assertEquals(mathutils.pairs_to_sets([(1, 2), (3, 4), (2, 3)]), [{1, 2, 3, 4}])
        self.assertEquals(mathutils.pairs_to_sets([(1, 2), (2, 3), (3, 4)]), [{1, 2, 3, 4}])
        self.assertNotEquals(mathutils.pairs_to_sets([(1, 2), (2, 3), (3, 4)]), [{1, 2, 3}, {4}])
        self.assertNotEquals(mathutils.pairs_to_sets([(1, 2), (2, 3), (3, 4)]), [{1, 2}, {3, 4}])


    def test_is_rect_overlap(self):
        bot_left_1 = (0, 10)
        top_right_1 = (10, 0)
        bot_left_2 = (5, 5)
        top_right_2 = (15, 0)
        self.assertTrue(mathutils.is_rect_overlap(bot_left_1,
                                                  top_right_1,
                                                  bot_left_2,
                                                  top_right_2))


if __name__ == "__main__":
    unittest.main()

