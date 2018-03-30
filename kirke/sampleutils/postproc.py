
from typing import List, Dict
from operator import itemgetter
from kirke.sampleutils.doccandidatesutils import DocCandidatesTransformer

class SpanDefaultPostProcessing(DocCandidatesTransformer):

    def __init__(self) -> None:
        self.label = 'span_default'

    def doc_postproc(self, candidates: List[Dict], nbest: int) -> None:
        for candidate in candidates:
            del_keys = []
            for key, val in candidate.items():
                if key not in set(['start', 'end', 'label', 'prob', 'text', 'span_list', 'norm']):
                    del_keys.append(key)

            for del_key in del_keys:
                del candidate[del_key]
        if nbest:
            nbest_candidates = sorted(candidates, key=itemgetter('prob'), reverse=True)[:nbest]
            return nbest_candidates
        else:
            return candidates
