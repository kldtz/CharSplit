
from typing import Dict, List, Tuple

from kirke.ebrules import addresses

class AddressAnnotator:

    def __init__(self) -> None:
        pass

    # simply take everything
    def apply_rules(self, samples) -> List[float]:

        prob_samples = []  # type: List[Tuple[float, Dict]]
        for sample in samples:
            prob_samples.append((addresses.classify(sample['text']), sample))

        return prob_samples
