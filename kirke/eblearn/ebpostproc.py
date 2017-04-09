import re
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List

from kirke.utils import strutils
from kirke.utils.ebantdoc import EbEntityType
from kirke.eblearn import ebattrvec

AntResult = namedtuple('AntResult', ['label', 'prob', 'start', 'end', 'text'])

class ConciseProbAttrvec:

    def __init__(self, prob, start, end, entities, text):
        self.prob = prob
        self.start = start
        self.end = end
        self.entities = entities
        self.text = text
        

def to_cx_prob_attrvecs(prob_attrvec_list):
    return [ConciseProbAttrvec(prob,
                               attrvec[ebattrvec.EB_ATTR_IDX_MAP['ent_start']],
                               attrvec[ebattrvec.EB_ATTR_IDX_MAP['ent_end']],
                               attrvec[ebattrvec.ENTITIES_INDEX],
                               attrvec[ebattrvec.TOKENS_TEXT_INDEX])
            for prob, attrvec in prob_attrvec_list]


def merge_cx_prob_attrvecs_with_entities(cx_prob_attrvec_list):
    # don't bother with len 1
    if len(cx_prob_attrvec_list) == 1:
        return cx_prob_attrvec_list[0]

    max_prob = cx_prob_attrvec_list[0].prob
    min_start = cx_prob_attrvec_list[0].start
    max_end = cx_prob_attrvec_list[0].end
    merged_entities = list(cx_prob_attrvec_list[0].entities)
    only_first_text = cx_prob_attrvec_list[0].text
    for cx_prob_attrvec in cx_prob_attrvec_list[1:]:
        if cx_prob_attrvec.prob > max_prob:
            max_prob = cx_prob_attrvec.prob
        if cx_prob_attrvec.start < min_start:
            min_start = cx_prob_attrvec.start
        if cx_prob_attrvec.end > max_end:
            max_end = cx_prob_attrvec.end
        merged_entities.extend(cx_prob_attrvec.entities)

    #for i, (prob, start, end) in enumerate(prob_start_end_list):
    #    print("jjj: {}".format((prob, start, end)))
    #print("result jjj: {}".format((max_prob, min_start, max_end)))

    return ConciseProbAttrvec(max_prob, min_start, max_end, merged_entities, only_first_text)


def merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold):
    result = []
    prev_list = []
    for cx_prob_attrvec in cx_prob_attrvec_list:
        if cx_prob_attrvec.prob >= threshold:
            prev_list.append(cx_prob_attrvec)
        else:
            if prev_list:
                result.append(merge_cx_prob_attrvecs_with_entities(prev_list))
                prev_list = []
            result.append(cx_prob_attrvec)
    if prev_list:
        result.append(merge_cx_prob_attrvecs_with_entities(prev_list))
    return result


"""
## TODO, jshaw, should be removed because ebsent is no longer accessible in ebantdoc
# pylint: disable=C0103
def merge_ebsent_probs_with_entities(prob_ebsent_list):
    # don't bother with len 1
    if len(prob_ebsent_list) == 1:
        return prob_ebsent_list[0]

    max_prob, _ = prob_ebsent_list[0]
    for prob, _ in prob_ebsent_list[1:]:
        if prob > max_prob:
            max_prob = prob

    #for i, (prob, start, end) in enumerate(prob_start_end_list):
    #    print("jjj: {}".format((prob, start, end)))
    #print("result jjj: {}".format((max_prob, min_start, max_end)))

    ebsent_list = [ebsent for prob, ebsent in prob_ebsent_list]
    return (max_prob, corenlpsent.merge_ebsents(ebsent_list))


def merge_prob_ebsents(prob_ebsent_list, threshold):
    result = []
    prev_list = []
    for prob, ebsent in prob_ebsent_list:
        if prob >= threshold:
            prev_list.append((prob, ebsent))
        else:
            if prev_list:
                result.append(merge_ebsent_probs_with_entities(prev_list))
                prev_list = []
            result.append((prob, ebsent))
    if prev_list:
        result.append(merge_ebsent_probs_with_entities(prev_list))
    return result
"""

PROVISION_PAT_MAP = {
    'change_control': re.compile(r'change\s+(of|in)\s+control', re.IGNORECASE | re.DOTALL),
    'confidentiality': re.compile(r'(information.*confidential|confidential.*information)',
                                  re.IGNORECASE | re.DOTALL),
    'limliability': re.compile(r'((is|are)\s+not\s+(liable|responsible)|'
                               r'will\s+not\s+be\s+(held\s+)?(liable|responsible)|'
                               r'no\s+(\S+\s+){1,5}(is|will\s+be)\s+responsible\s+for|'
                               r'not\s+(be\s+)?required\s+to\s+make\s+(\S+\s+){1,3}payment|'
                               r'need\s+not\s+make\s(\S+\s+){1,3}payment)',
                               re.IGNORECASE | re.DOTALL),
    'term': re.compile(r'[“"]Termination\s+Date[”"]', re.IGNORECASE | re.DOTALL)}


# override some provisions during testing
def gen_provision_overrides(provision, sent_st_list):
    overrides = [None for _ in range(len(sent_st_list))]

    global_min_length = 6
    min_pattern_override_length = 8
    if provision == 'term':
        min_pattern_override_length = 0

    provision_pattern = PROVISION_PAT_MAP.get(provision)
    for sent_idx, sent_st in enumerate(sent_st_list):
        toks = sent_st.split()   # TODO, a little repetitive, split again
        num_words = len(toks)
        num_numeric = sum(1 for tok in toks if strutils.is_number(tok))
        is_toc = num_words > 60 and num_numeric / num_words > .2
        if (provision_pattern and provision_pattern.search(sent_st) and
                num_words > min_pattern_override_length and not is_toc):
            overrides[sent_idx] = 1
        if num_words < global_min_length:
            overrides[sent_idx] = 0
    return overrides


# pylint: disable=R0903
class EbPostPredictProcessing(ABC):

    @abstractmethod
    def post_process(self, prob_attrvec_list, threshold, provision=None):
        pass

# pylint: disable=R0903
class DefaultPostPredictProcessing(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'default'

    def post_process(self, prob_attrvec_list, threshold, provision=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            if cx_prob_attrvec.prob >= threshold:
                tmp_provision = provision if provision else self.provision
                ant_result.append(AntResult(label=tmp_provision,
                                            prob=cx_prob_attrvec.prob,
                                            start=cx_prob_attrvec.start,
                                            end=cx_prob_attrvec.end,
                                            text=strutils.remove_nltab(cx_prob_attrvec.text[:50]) + '...'))
        return ant_result


# pylint: disable=R0903
class PostPredPartyProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'party'

    def post_process(self, prob_attrvec_list, threshold, provision=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            if cx_prob_attrvec.prob >= threshold:
                for entity in cx_prob_attrvec.entities:
                    if entity.ner in {EbEntityType.PERSON.name, EbEntityType.ORGANIZATION.name}:
                        ant_result.append(AntResult(label=self.provision,
                                                    prob=cx_prob_attrvec.prob,
                                                    start=entity.start,
                                                    end=entity.end,
                                                    text=strutils.remove_nltab(entity.text)))
        return ant_result

# pylint: disable=R0903
class PostPredDateProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'date'

    def post_process(self, cx_prob_attrvec_list, threshold, provision=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(cx_prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            if cx_prob_attrvec.prob >= threshold:
                for entity in cx_prob_attrvec.entities:
                    if entity.ner == EbEntityType.DATE.name:
                        ant_result.append(AntResult(label=self.provision,
                                                    prob=cx_prob_attrvec.prob,
                                                    start=entity.start,
                                                    end=entity.end,
                                                    text=strutils.remove_nltab(entity.text)))
        return ant_result


PROVISION_POSTPROC_MAP = {
    'default': DefaultPostPredictProcessing(),
    'party': PostPredPartyProc(),
    'date': PostPredDateProc()
}

def obtain_postproc(provision):
    postproc = PROVISION_POSTPROC_MAP.get(provision)
    if not postproc:
        postproc = PROVISION_POSTPROC_MAP['default']
    return postproc
