from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional

# pylint: disable=import-error
from sklearn.pipeline import Pipeline

from kirke.utils import osutils

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
        logger.info("saving model file: %s", model_file_name)
        self.file_name = model_file_name
        # joblib.dump(self, model_file_name)
        osutils.joblib_atomic_dump(self, model_file_name)

    # pylint: disable=too-many-arguments
    @abstractmethod
    def train_candidates(self,
                         candidates: List[Dict],
                         label_list: List[bool],
                         group_id_list: List[int],
                         pipeline: Pipeline,
                         parameters: Dict,
                         work_dir: str) -> None:
        pass

    @abstractmethod
    def predict_antdoc(self,
                       eb_antdoc,
                       work_dir: str,
                       nbest: Optional[int] = None):
        pass

    @abstractmethod
    def predict_and_evaluate(self,
                             candidates: List[Dict],
                             label_list: List[bool],
                             work_dir: str,
                             is_debug=False):
        pass
