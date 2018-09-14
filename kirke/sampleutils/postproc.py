
from typing import List, Dict
from operator import itemgetter
from kirke.sampleutils.doccandidatesutils import DocCandidatesTransformer


# pylint: disable=abstract-method
class SpanDefaultPostProcessing(DocCandidatesTransformer):

    def __init__(self) -> None:
        super().__init__()
        self.label = 'span_default'

    # pylint: disable=no-self-use
    def doc_postproc(self, candidates: List[Dict], nbest: int) -> List[Dict]:
        for candidate in candidates:
            del_keys = []
            for key, unused_val in candidate.items():
                if key not in set(['start', 'end', 'label', 'prob', 'text',
                                   'span_list', 'norm']):
                    del_keys.append(key)

            for del_key in del_keys:
                del candidate[del_key]
        if nbest > 0:
            nbest_candidates = sorted(candidates, key=itemgetter('prob'), reverse=True)[:nbest]
            return nbest_candidates

        return candidates

    def enrich(self, candidate: Dict) -> None:
        pass


# pylint: disable=abstract-method
class TablePostProcessing(DocCandidatesTransformer):

    def __init__(self) -> None:
        super().__init__()
        self.label = 'span_table'

    # pylint: disable=no-self-use
    def doc_postproc(self, candidates: List[Dict], nbest: int) -> List[Dict]:
        for candidate in candidates:
            del_keys = []
            candidate_json = None
            for key, unused_val in candidate.items():
                if key not in set(['start', 'end', 'label', 'prob', 'text',
                                   'span_list', 'norm', 'json']):
                    del_keys.append(key)

            for del_key in del_keys:
                del candidate[del_key]
        if nbest > 0:
            nbest_candidates = sorted(candidates, key=itemgetter('prob'), reverse=True)[:nbest]
            return nbest_candidates

        return candidates

    def enrich(self, candidate: Dict) -> None:
        pass

