import logging
from abc import ABC, abstractmethod

from sklearn.externals import joblib

from kirke.utils import ebantdoc3

GLOBAL_THRESHOLD = 0.12


class BaseAnnotator(ABC):

    def __init__(self, name, desc):
        self.name = name
        self.description = desc
        self.threshold = GLOBAL_THRESHOLD
        self.pred_status = {}
        super(BaseAnnoator, self).__init__()

    def set_threshold(self, val):
        self.threshold = val

    def get_pred_status(self):
        return self.pred_status

    def save(self, model_file_name):
        logging.info("saving model file: %s", model_file_name)
        joblib.dump(self, model_file_name)

    def train(self, txt_fn_list, work_dir, model_file_name, name=None):
        ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
        return self.train_antdoc_list(ebantdoc_list, work_dir, model_file_name)

    @abstractmethod
    def train_antdoc_list(self, ebantdoc_list, work_dir, model_file_name):
        pass

    @abstractmethod
    def predict_antdoc(self, eb_antdoc, work_dir):
        pass

    @abstractmethod
    def predict_and_evaluate(self, ebantdoc_list, work_dir, is_debug=False):
        pass