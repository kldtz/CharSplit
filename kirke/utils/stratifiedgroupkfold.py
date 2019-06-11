from collections import defaultdict
# pylint: disable=unused-import
from typing import DefaultDict, Iterator, List, Optional, Set, Tuple

from sklearn.model_selection import StratifiedKFold


# pylint: disable=too-few-public-methods
class StratifiedGroupKFold:

    def __init__(self, n_splits: int = 3) -> None:
        self.n_splits = n_splits

    # pylint: disable=too-many-locals, invalid-name
    def get_n_splits(self,
                     # pylint: disable=unused-argument
                     X: Optional[List] = None,
                     # pylint: disable=unused-argument
                     y: Optional[List[bool]] = None,
                     # pylint: disable=unused-argument
                     groups: Optional[List[int]] = None) \
                     -> int:
        return self.n_splits

    # pylint: disable=too-many-locals, invalid-name
    def split(self,
              # pylint: disable=invalid-name
              X: List,
              y: List[bool],
              groups: List[int]) \
              -> Iterator[Tuple[List[int], List[int]]]:
        """Return the training and testing index int to X and y."""

        docid_list = []  # type: List[int]
        docid_index_i_map = defaultdict(list)  # type: DefaultDict[int, List[int]]
        docid_ylist_map = defaultdict(list)  # type: DefaultDict[int, List[bool]]
        docid_set = set([])  # type: Set[int]
        docid_is_positive_doc_map = {}
        # docid_attrvevy_map = defaultdict(list)
        for index_i, (unused_x_attrvec, y_bool, docid) in enumerate(zip(X, y, groups)):
            # docid_attrvecy_map[docid].append((x_attrvec, y_bool))
            docid_index_i_map[docid].append(index_i)
            docid_ylist_map[docid].append(y_bool)

            if docid not in docid_set:
                docid_set.add(docid)
                docid_list.append(docid)
                docid_is_positive_doc_map[docid] = False

            if y_bool:
                docid_is_positive_doc_map[docid] = True

        # this is a list of doc_is_positive
        docid_y_list = []  # type: List[bool]
        for docid in docid_list:
            is_positive = docid_is_positive_doc_map[docid]
            docid_y_list.append(is_positive)

        # now we have a list of docid_list and docid_y
        # find the original index associate with each docid and output their indices
        skf = StratifiedKFold(n_splits=self.n_splits)
        for train_index, test_index in skf.split(docid_list, docid_y_list):
            X_train_index_outs = []  # type: List[int]
            X_test_index_outs = []  # type: List[int]
            for docid_index in train_index:
                docid = docid_list[docid_index]
                X_train_index_outs.extend(docid_index_i_map[docid])

            for docid_index in test_index:
                docid = docid_list[docid_index]
                X_test_index_outs.extend(docid_index_i_map[docid])

            yield X_train_index_outs, X_test_index_outs
