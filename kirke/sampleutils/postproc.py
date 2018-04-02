
from typing import Dict

from kirke.sampleutils.doccandidatesutils import DocCandidatesTransformer

class SpanDefaultPostProcessing(DocCandidatesTransformer):

    def __init__(self) -> None:
        self.label = 'span_default'

    def enrich(self, candidate: Dict) -> None:
        del_keys = []
        for key, val in candidate.items():
            if key not in set(['start', 'end', 'label', 'prob', 'text', 'span_list', 'norm']):
                del_keys.append(key)

        for del_key in del_keys:
            del candidate[del_key]

