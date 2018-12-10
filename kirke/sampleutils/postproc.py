
# pylint: disable=unused-import
from typing import Dict, List, Tuple
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

class SentDefaultPostProcessing(DocCandidatesTransformer):

    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.label = 'sent_default'
        self.threshold = threshold

    # pylint: disable=no-self-use
    def get_merged_cand(self, cand_list: List[Dict]) -> Dict:
        if len(cand_list) == 1:
            return cand_list[0]
        max_prob = max([cand['prob'] for cand in cand_list])
        min_start = min([cand['start'] for cand in cand_list])
        max_end = max([cand['end'] for cand in cand_list])
        new_text = " ".join([cand['text'] for cand in cand_list]) ####
        new_span = []  # type: List[Tuple[int, int]]
        for cand in cand_list:
            new_span.extend(cand['span_list'])
        new_cand = {'start': min_start,
                    'end': max_end,
                    'label': cand_list[0]['label'],
                    'prob': max_prob,
                    'text': new_text,
                    'span_list': new_span}
        return new_cand

    def merge_adjacent_sents(self, candidates: List[Dict]) -> List[Dict]:
        result = []
        prev_list = []
        for candidate in candidates:
            if candidate['prob'] >= self.threshold:
                prev_list.append(candidate)
            else:
                if prev_list:
                    result.append(self.get_merged_cand(prev_list))
                    prev_list = []
                else:
                    result.append(candidate)
        if prev_list:
            result.append(self.get_merged_cand(prev_list))
        return result


     # pylint: disable=no-self-use
    def doc_postproc(self, candidates: List[Dict], nbest: int) -> List[Dict]:
        merged_cands = self.merge_adjacent_sents(candidates)
        for candidate in merged_cands:
            del_keys = []
            for key, unused_val in candidate.items():
                if key not in set(['start', 'end', 'label', 'prob', 'text',
                                   'span_list', 'norm']):
                    del_keys.append(key)

            for del_key in del_keys:
                del candidate[del_key]
        if nbest > 0:
            nbest_candidates = sorted(merged_cands, key=itemgetter('prob'), reverse=True)[:nbest]
            return nbest_candidates

        return merged_cands

    def enrich(self, candidate: Dict) -> None:
        pass
