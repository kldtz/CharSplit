
from typing import Dict, List, Tuple

class DummyAnnotator:

    def __init__(self) -> None:
        pass

    # simply take everything
    def apply_rules(self, samples: List[Dict]) -> List[Tuple[float, Dict]]:
        return [(1.0, sample) for sample in samples]

    def post_process(self,
                     prob_samples: List[Tuple[float, Dict]],
                     text: str) -> List[Tuple[float, Dict]]:
        return prob_samples
