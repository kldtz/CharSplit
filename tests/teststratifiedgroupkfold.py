#!/usr/bin/env python3

from typing import Iterator, List, Tuple
import unittest

import numpy as np
from sklearn.model_selection import GroupKFold

from kirke.utils.stratifiedgroupkfold import StratifiedGroupKFold


def split_out_to_list(split_out: Iterator[Tuple[np.ndarray, np.ndarray]]) \
    -> List[Tuple[List[int], List[int]]]:
    out_list = []  # type: List[Tuple[List[int], List[int]]]
    for train_ndarray, test_ndarray in split_out:
        train_list = train_ndarray.tolist()
        test_list = test_ndarray.tolist()
        out_list.append((train_list, test_list))
    return out_list


class TestStratifiedGroupKFoldUtils(unittest.TestCase):

    # pylint: disable=invalid-name
    def test_stratified_group_k_fold(self):
        "Test StratifiedGroupKFold()"

        # pylint: disable=invalid-name
        X = np.array([10, 20, 30, 40, 50, 60])
        # pylint: disable=invalid-name
        y = np.array([1, 0, 1, 0, 1, 0])
        # y = np.array([0, 0, 0, 1, 1, 1])
        # groups = np.array([1, 1, 2, 2, 3, 3])
        # groups = np.array([1, 2, 3, 1, 2, 3])
        groups = np.array([1, 1, 2, 2, 3, 3])

        sgkf = StratifiedGroupKFold(n_splits=3)
        # for train, test in sgkf.split(X, y, groups):
        # print("final stratified_groupkfold k=3: %s %s" % (train, test))

        gold_output = [([2, 3, 4, 5], [0, 1]),
                       ([0, 1, 4, 5], [2, 3]),
                       ([0, 1, 2, 3], [4, 5])]

        self.assertEqual(list(sgkf.split(X, y, groups)),
                         gold_output)

    # pylint: disable=invalid-name
    def test_group_k_fold_not_stratified(self):
        # this is for a list of 9
        # pylint: disable=invalid-name
        X = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
        # pylint: disable=invalid-name
        y = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0])
        groups = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

        gkf = GroupKFold(n_splits=3)

        # doc_index 1 is negative (2nd)
        # doc index 4 is negative (5th)
        # doc index 7 is negative (8th)
        # they, [1, 4, 7] shouldn't be exist in the 2nd row below if
        # the grouping is stratified
        gold_output = [([0, 1, 3, 4, 6, 7], [2, 5, 8]),
                       ([0, 2, 3, 5, 6, 8], [1, 4, 7]),
                       ([1, 2, 4, 5, 7, 8], [0, 3, 6])]

        self.assertEqual(split_out_to_list(gkf.split(X, y, groups)),
                         gold_output)

        # this is for a list of 7
        X = np.array([10, 20, 30, 40, 50, 60, 70])
        y = np.array([0, 0, 0, 1, 0, 1, 1])
        groups = np.array([1, 2, 3, 4, 5, 6, 7])

        gkf = GroupKFold(n_splits=3)

        # doc_index 1 is negative (2nd)
        # doc index 4 is negative (5th)
        # they, [1, 4], shouldn't be exist in the 3rd row below if
        # the grouping is stratified
        gold_output = [([1, 2, 4, 5], [0, 3, 6]),
                       ([0, 1, 3, 4, 6], [2, 5]),
                       ([0, 2, 3, 5, 6], [1, 4])]

        self.assertEqual(split_out_to_list(gkf.split(X, y, groups)),
                         gold_output)
