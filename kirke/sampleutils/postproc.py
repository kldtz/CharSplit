
from typing import Dict

class SpanDefaultPostProcessing:

    def __init__(self) -> None:
        self.label = 'span_default'

    def enrich(self, sample: Dict) -> None:
        print("calling SpanDefaultPostProcessing.enrich()")
        del_keys = []
        for key, val in sample.items():
            if key not in set(['start', 'end', 'label', 'prob', 'text', 'span_list']):
                del_keys.append(key)

        for del_key in del_keys:
            del sample[del_key]

