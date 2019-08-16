#!/usr/bin/env python3

from abc import ABC, abstractmethod
from collections import Counter
import logging
# pylint: disable=unused-import
from typing import Any, Dict

from kirke.utils import ebantdoc4, osutils

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GLOBAL_THRESHOLD = 0.12


class EbClassifier(ABC):

    def __init__(self, provision: str) -> None:
        super().__init__()
        self.provision = provision
        self.threshold = GLOBAL_THRESHOLD
        self.pred_status = {}  # type: Dict[str, Any]

    def set_threshold(self, val: float):
        self.threshold = val

    def get_pred_status(self):
        return self.pred_status

    def save(self, model_file_name: str):
        # logger.debug("saving model file: %s", model_file_name)
        # joblib.dump(self, model_file_name)
        osutils.joblib_atomic_dump(self, model_file_name)

    def train(self,
              txt_fn_list: str,
              work_dir: str,
              model_file_name: str) -> None:
        ebantdoc_list = ebantdoc4.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)

        # figure out the language of those docs
        lang_counter = Counter()  # type: Counter
        for eb_antdoc in ebantdoc_list:
            lang_counter[eb_antdoc.doc_lang] += 1
        top_lang, unused_freq = lang_counter.most_common(1)[0]

        self.train_antdoc_list(ebantdoc_list,
                               lang=top_lang,
                               model_file_name=model_file_name)

    @abstractmethod
    def train_antdoc_list(self,
                          ebantdoc_list,
                          *,
                          lang: str,
                          model_file_name: str) -> None:
        pass

    @abstractmethod
    def predict_antdoc(self, eb_antdoc, work_dir):
        pass

    @abstractmethod
    def predict_and_evaluate(self, ebantdoc_list, work_dir, diagnose_mode=False):
        pass
