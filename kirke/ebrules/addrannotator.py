
from typing import Dict, List, Tuple

from kirke.ebrules import addresses

# pylint: disable=too-few-public-methods
class AddressAnnotator:

    def __init__(self) -> None:
        pass

    # simply take everything
    # pylint: disable=no-self-use
    def apply_rules(self, samples: List[Dict]) -> List[Tuple[float, Dict]]:

        prob_samples = []  # type: List[Tuple[float, Dict]]
        for sample in samples:
            prob_samples.append((addresses.classify(sample['text']), sample))

        return prob_samples
