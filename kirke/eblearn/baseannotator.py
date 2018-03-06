import logging
from abc import ABC, abstractmethod
from typing import Dict, List

# pylint: disable=import-error
from sklearn.pipeline import Pipeline
# pylint: disable=import-error
from sklearn.externals import joblib


GLOBAL_THRESHOLD = 0.12


class BaseAnnotator(ABC):

    def __init__(self, label: str, desc: str) -> None:
        self.label = label
        self.description = desc
        self.threshold = GLOBAL_THRESHOLD
        self.pred_status = {}  # type: Dict
        self.file_name = 'unknown'
        super().__init__()

    def set_threshold(self, val: float) -> None:
        self.threshold = val

    def get_pred_status(self) -> Dict:
        return self.pred_status

    def save(self, model_file_name) -> None:
        logging.info("saving model file: %s", model_file_name)
        self.file_name = model_file_name
        joblib.dump(self, model_file_name)

    # pylint: disable=too-many-arguments
    @abstractmethod
    def train_samples(self,
                      samples: List[Dict],
                      label_list: List[bool],
                      group_id_list: List[int],
                      pipeline: Pipeline,
                      parameters: Dict,
                      work_dir: str) -> None:
        pass

    @abstractmethod
    def predict_antdoc(self,
                       eb_antdoc,
                       work_dir: str):
        pass

    @abstractmethod
    def predict_and_evaluate(self,
                             samples: List[Dict],
                             label_list: List[bool],
                             work_dir: str,
                             is_debug=False):
        pass
