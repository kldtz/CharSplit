#!/usr/bin/env python3

from abc import ABC, abstractmethod
import logging

from kirke.utils import ebantdoc4, osutils

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GLOBAL_THRESHOLD = 0.12


class EbClassifier(ABC):

    def __init__(self, provision):
        self.provision = provision
        self.threshold = GLOBAL_THRESHOLD
        self.pred_status = {}
        super(EbClassifier, self).__init__()

    def set_threshold(self, val):
        self.threshold = val

    def get_pred_status(self):
        return self.pred_status

    def save(self, model_file_name):
        logger.info("saving model file: %s", model_file_name)
        # joblib.dump(self, model_file_name)
        osutils.joblib_atomic_dump(self, model_file_name)

    # this returns an estimator + list of scores for the training docs,
    # (estimator, List[float])
    def train(self, txt_fn_list, work_dir, model_file_name) -> None:
        ebantdoc_list = ebantdoc4.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
        self.train_antdoc_list(ebantdoc_list, work_dir, model_file_name)

    @abstractmethod
    def train_antdoc_list(self, ebantdoc_list, work_dir, model_file_name) -> None:
        pass

    @abstractmethod
    def predict_antdoc(self, eb_antdoc, work_dir):
        pass

    @abstractmethod
    def predict_and_evaluate(self, ebantdoc_list, work_dir, diagnose_mode=False):
        pass
