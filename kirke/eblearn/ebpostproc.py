import re
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List

from kirke.utils import strutils
from kirke.utils.ebantdoc import EbEntityType
from kirke.eblearn import ebattrvec

# AntResult = namedtuple('AntResult', ['label', 'prob', 'start', 'end', 'text'])
# xxxx namedtuple('AntResult', ['label', 'prob', 'start', 'end', 'text'])
class AntResult:

    def __init__(self, label, prob, start, end, text):
        self.label = label
        self.prob = prob
        self.start = start
        self.end = end
        self.text = text

    def to_dict(self):
        return {'label': self.label,
                'prob': self.prob,
                'start': self.start,
                'end': self.end,
                'text': self.text}

class ConciseProbAttrvec:

    def __init__(self, prob, start, end, entities, text):
        self.prob = prob
        self.start = start
        self.end = end
        self.entities = entities
        self.text = text
        

def to_cx_prob_attrvecs(prob_attrvec_list) -> List[ConciseProbAttrvec]:
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
    def post_process(self, doc_text, prob_attrvec_list, threshold, provision=None):
        pass

    
# pylint: disable=R0903
class DefaultPostPredictProcessing(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'default'

    def post_process(self, doc_text, prob_attrvec_list, threshold, provision=None) -> List[AntResult]:
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
                                            text=strutils.remove_nltab(cx_prob_attrvec.text[:50]) + '...').to_dict())
        return ant_result

# Note from PythonClassifier.java:
# The NER seems to pick up the bare word LLC, INC, and CORP as parties sometimes.  This RE
# defines strings that should not be considered parties.
NOT_PARTY_PAT = re.compile(r'(inc|llc|corp)\.?', re.IGNORECASE)

# pylint: disable=R0903
class PostPredPartyProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'party'

    def post_process(self, doc_text, prob_attrvec_list, threshold, provision=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            if cx_prob_attrvec.prob >= threshold:
                for entity in cx_prob_attrvec.entities:
                    if entity.ner in {EbEntityType.PERSON.name, EbEntityType.ORGANIZATION.name}:

                        if 'agreement' in entity.text.lower() or NOT_PARTY_PAT.match(entity.text):
                            continue
                        ant_result.append(AntResult(label=self.provision,
                                                    prob=cx_prob_attrvec.prob,
                                                    start=entity.start,
                                                    end=entity.end,
                                                    text=strutils.remove_nltab(entity.text)).to_dict())
        return ant_result

# pylint: disable=R0903
class PostPredDateProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'date'

    # TODO, jshaw, it seems that in the original code PythonClassifier.java
    # the logic is to keep only the first date, not all dates in a doc
    def post_process(self, doc_text, cx_prob_attrvec_list, threshold, provision=None) -> List[AntResult]:
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
                                                    text=strutils.remove_nltab(entity.text)).to_dict())
                        # return only 1 date
                        return ant_result
        return ant_result

#       Pattern titleRE = Pattern.compile(
#          "(?:exhibit \\d+\\.\\d+\\s+|this )?((?:.+? )?agreement)(?: \\(| is)?",
#          Pattern.CASE_INSENSITIVE);

# Note from PythonClassifier.java:
# A title might optionally start with an Exhibit X.X number (for SEC contracts) or optionally
# start with "this XXXX Agreement".  It may end (optionally) with the word agreement, and
# with the word is or an open paren (for the defined term parentetical)
TITLE_PAT = re.compile(r'(?:exhibit \d+\.\d+\s+|this )?((?:.+? )?agreement)(?: \(| is)?', re.IGNORECASE)

class PostPredTitleProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'title'

    def post_process(self, doc_text, cx_prob_attrvec_list, threshold, provision=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(cx_prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            if cx_prob_attrvec.prob >= threshold:
                anttext = doc_text[cx_prob_attrvec.start:cx_prob_attrvec.end]
                mat = TITLE_PAT.match(anttext)
                if mat:
                    tmp_start = cx_prob_attrvec.start + mat.start(1)
                    tmp_title = mat.group(1)
                    ant_result.append(AntResult(label=self.provision,
                                                prob=cx_prob_attrvec.prob,
                                                start=tmp_start,
                                                end=tmp_start + len(tmp_title),
                                                text=tmp_title).to_dict())
                    return ant_result
        return ant_result


class PostPredBestDateProc(EbPostPredictProcessing):

    def __init__(self, prov_name):
        self.provision = prov_name
        self.threshold = 0.5

    def post_process(self, doc_text, cx_prob_attrvec_list, threshold, provision=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(cx_prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        best_effectivedate_sent = self.get_best(merged_prob_attrvec_list)

        ant_result = []
        if best_effectivedate_sent:
            first = None
            first_after_effective = None

            for entity in best_effectivedate_sent.entities:
                if entity.ner == EbEntityType.DATE.name:
                    prior_text = doc_text[best_effectivedate_sent.start:entity.start]
                    has_prior_text_effective = 'ffective' in prior_text

                    ant_rx = AntResult(label=self.provision,
                                       prob=best_effectivedate_sent.prob,
                                       start=entity.start,
                                       end=entity.end,
                                       text=strutils.remove_nltab(doc_text[entity.start:entity.end])).to_dict()
                    if not first:
                        first = ant_rx
                    if has_prior_text_effective and not first_after_effective:
                        first_after_effective = ant_rx
                        
            if first_after_effective:
                ant_result.append(first_after_effective)
            elif first:
                ant_result.append(first)

        return ant_result

    def get_best(self, prob_attrvec_list: List[ConciseProbAttrvec]) -> ConciseProbAttrvec:
        best_prob = 0
        best = None
        for cx_prob_attrvec in prob_attrvec_list:
            if cx_prob_attrvec.prob >= self.threshold:   # this is not threshold from top
                if cx_prob_attrvec.prob > best_prob:
                    best_prob = cx_prob_attrvec.prob
                    best = cx_prob_attrvec
        return best

    
PROVISION_POSTPROC_MAP = {
    'default': DefaultPostPredictProcessing(),
    'party': PostPredPartyProc(),
    'title': PostPredTitleProc(),
    # 'date': PostPredDateProc(),
    'date': PostPredBestDateProc('date'),
    'sigdate': PostPredBestDateProc('sigdate'),
    'effectivedate': PostPredBestDateProc('effectivedate')
}


def obtain_postproc(provision):
    postproc = PROVISION_POSTPROC_MAP.get(provision)
    if not postproc:
        postproc = PROVISION_POSTPROC_MAP['default']
    return postproc
