#!/usr/bin/env python

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


if __name__ == "__main__":
    unittest.main()

