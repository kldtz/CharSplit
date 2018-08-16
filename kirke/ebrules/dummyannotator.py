
from typing import Dict, List, Tuple

class DummyAnnotator:

    def __init__(self) -> None:
        pass

    # simply take everything
    # pylint: disable=no-self-use
    def apply_rules(self, samples: List[Dict]) -> List[Tuple[float, Dict]]:
        return [(1.0, sample) for sample in samples]

    # pylint: disable=no-self-use
    def post_process(self,
                     prob_samples: List[Tuple[float, Dict]],
                     # pylint: disable=unused-argument
                     text: str) -> List[Tuple[float, Dict]]:
        return prob_samples
