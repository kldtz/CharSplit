
from typing import Dict, List, Tuple

from kirke.ebrules import addresses

class AddressAnnotator:

    def __init__(self) -> None:
        pass

    # simply take everything
    def apply_rules(self, samples: List[Dict]) -> List[Tuple[float, Dict]]:

        prob_samples = []  # type: List[Tuple[float, Dict]]
        for sample in samples:
            prob_samples.append((addresses.classify(sample['text']), sample))

        return prob_samples

class SampleAddAddrLineProb:

    def __init__(self) -> None:
        pass

    def enrich(self, samples: List[Dict]) -> List[Dict]:
        print("SampleAddrLineProb.enriching() called")
        for i, sample in enumerate(samples):
            sample['addr_line_prob'] = addresses.classify(sample['text'])
            if (i + 1) % 1000 == 0:
                print("processed addr_line_prob {}".format(i+1))
        return samples
