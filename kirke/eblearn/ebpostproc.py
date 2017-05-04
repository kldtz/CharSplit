import re
from abc import ABC, abstractmethod
from typing import List

from kirke.utils import strutils, entityutils, stopwordutils
from kirke.utils.ebantdoc import EbEntityType
from kirke.eblearn import ebattrvec


PROVISION_PAT_MAP = {
    'change_control': (re.compile(r'change\s+(of|in)\s+control', re.IGNORECASE | re.DOTALL), 1.0),
#    'confidentiality': (re.compile(r'(information.*confidential|confidential.*information)',
#                                   re.IGNORECASE | re.DOTALL), 1.0),
    'limliability': (re.compile(r'((is|are)\s+not\s+(liable|responsible)|'
                               r'will\s+not\s+be\s+(held\s+)?(liable|responsible)|'
                               r'no\s+(\S+\s+){1,5}(is|will\s+be)\s+responsible\s+for|'
                               r'not\s+(be\s+)?required\s+to\s+make\s+(\S+\s+){1,3}payment|'
                               r'need\s+not\s+make\s(\S+\s+){1,3}payment)',
                                re.IGNORECASE | re.DOTALL), 1.0),
    'term': (re.compile(r'[“"]Termination\s+Date[”"]', re.IGNORECASE | re.DOTALL), 1.0)
}


# pylint: disable=too-few-public-methods
class AntResult:

    # pylint: disable=too-many-arguments
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


# pylint: disable=too-few-public-methods
class ConciseProbAttrvec:

    # pylint: disable=too-many-arguments
    def __init__(self, prob, start, end, entities, text):
        self.prob = prob
        self.start = start
        self.end = end
        self.entities = entities
        self.text = text


def to_cx_prob_attrvecs(prob_attrvec_list) -> List[ConciseProbAttrvec]:
    return [ConciseProbAttrvec(prob,
                               attrvec.start,
                               attrvec.end,
                               attrvec.entities,
                               attrvec.bag_of_words)
            for prob, attrvec in prob_attrvec_list]


# pylint: disable=invalid-name
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


SHORT_PROVISIONS = set(['title', 'date', 'effectivedate', 'sigdate'])

# override some provisions during testing
def gen_provision_overrides(provision, sent_st_list):
    overrides = [0.0 for _ in range(len(sent_st_list))]

    global_min_length = 6
    min_pattern_override_length = 8
    if provision == 'term':
        min_pattern_override_length = 0

    provision_pattern = None
    adjust_prob = 0.0
    pat_adjscore = PROVISION_PAT_MAP.get(provision)
    if pat_adjscore:
        provision_pattern = pat_adjscore[0]
        adjust_prob = pat_adjscore[1]

    for sent_idx, sent_st in enumerate(sent_st_list):
        # pylint: disable=fixme
        # toks = sent_st.split()   # TODO, a little repetitive, split again
        toks = stopwordutils.get_nonstopwords_gt_len1(sent_st)
        num_words = len(toks)
        num_numeric = sum(1 for tok in toks if strutils.is_number(tok))
        is_toc = num_words > 60 and num_numeric / num_words > 0.2
        is_table_row = num_words > 5 and num_numeric / num_words > 0.3
        contains_dots = '....' in sent_st
        if (provision_pattern and provision_pattern.search(sent_st) and
            num_words > min_pattern_override_length and not is_toc):
            overrides[sent_idx] = adjust_prob
        if num_words < global_min_length and provision not in SHORT_PROVISIONS:
            overrides[sent_idx] = -10.0
        if is_table_row or contains_dots:
            overrides[sent_idx] = -10.0
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

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list,
                                                          threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            if cx_prob_attrvec.prob >= threshold:
                tmp_provision = provision if provision else self.provision
                ant_result.append(AntResult(label=tmp_provision,
                                            prob=cx_prob_attrvec.prob,
                                            start=cx_prob_attrvec.start,
                                            end=cx_prob_attrvec.end,
                                            # pylint: disable=line-too-long
                                            text=strutils.remove_nltab(cx_prob_attrvec.text[:50]) + '...').to_dict())
        return ant_result

# Note from PythonClassifier.java:
# The NER seems to pick up the bare word LLC, INC, and CORP as parties sometimes.  This RE
# defines strings that should not be considered parties.
NOT_PARTY_PAT = re.compile(r'((inc|llc|corp)\.?|p\.?\s*o\.?\s*box.*)', re.IGNORECASE)

# pylint: disable=R0903
class PostPredPartyProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'party'

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None) -> List[AntResult]:
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
                                                    # pylint: disable=line-too-long
                                                    text=strutils.remove_nltab(entity.text)).to_dict())
        return ant_result


class PostPredChoiceOfLawProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'choiceoflaw'

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            if cx_prob_attrvec.prob >= threshold:
                anttext = doc_text[cx_prob_attrvec.start:cx_prob_attrvec.end]
                state_se_tuple_list = entityutils.extract_unique_states(anttext)
                if state_se_tuple_list:
                    for state_se in state_se_tuple_list:
                        tmp_start = cx_prob_attrvec.start + state_se[0]
                        tmp_end = cx_prob_attrvec.start + state_se[1]
                        tmp_state = state_se[2]
                        ant_result.append(AntResult(label=self.provision,
                                                    prob=cx_prob_attrvec.prob,
                                                    start=tmp_start,
                                                    end=tmp_end,
                                                    text=tmp_state).to_dict())
                else:
                    ant_result.append(AntResult(label=self.provision,
                                                prob=cx_prob_attrvec.prob,
                                                start=cx_prob_attrvec.start,
                                                end=cx_prob_attrvec.end,
                                                text=anttext).to_dict())
        return ant_result


# Note from PythonClassifier.java:
# A title might optionally start with an Exhibit X.X number (for SEC contracts) or optionally
# start with "this XXXX Agreement".  It may end (optionally) with the word agreement, and
# with the word is or an open paren (for the defined term parentetical)
# pylint: disable=line-too-long
TITLE_PAT = re.compile(r'(?:exhibit \d+\.\d+\s+|this |\s*execution copy\s+)?((?:.+? )?agreement)(?: \(| is)?', re.IGNORECASE)

class PostPredTitleProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'title'

    def post_process(self, doc_text, cx_prob_attrvec_list, threshold,
                     provision=None) -> List[AntResult]:
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


# used by both PostPredDateProc, PostPredEffectiveDate
def get_best_date(prob_attrvec_list: List[ConciseProbAttrvec], threshold) -> ConciseProbAttrvec:
    best_prob = 0
    best = None
    for cx_prob_attrvec in prob_attrvec_list:
        if cx_prob_attrvec.prob >= threshold:   # this is not threshold from top
            if cx_prob_attrvec.prob > best_prob:
                best_prob = cx_prob_attrvec.prob
                best = cx_prob_attrvec
    return best


# pylint: disable=R0903
class PostPredBestDateProc(EbPostPredictProcessing):

    def __init__(self, prov):
        self.provision = prov
        self.threshold = 0.1

    # TODO, jshaw, it seems that in the original code PythonClassifier.java
    # the logic is to keep only the first date, not all dates in a doc
    def post_process(self, doc_text, cx_prob_attrvec_list, threshold,
                     provision=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(cx_prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list,
                                                          threshold)

        best_date_sent = get_best_date(merged_prob_attrvec_list,
                                       self.threshold)

        ant_result = []
        if best_date_sent:
            for entity in best_date_sent.entities:
                if entity.ner == EbEntityType.DATE.name:
                    ant_rx = AntResult(label=self.provision,
                                       prob=best_date_sent.prob,
                                       start=entity.start,
                                       end=entity.end,
                                       # pylint: disable=line-too-long
                                       text=strutils.remove_nltab(doc_text[entity.start:entity.end])).to_dict()
                    ant_result.append(ant_rx)

                    # print("post_process, bestDate({}) = {}".format(self.provision, ant_result))
                    return ant_result

        # print("post_process, bestDate2({}) = {}".format(self.provision, ant_result))
        return ant_result


class PostPredEffectiveDateProc(EbPostPredictProcessing):

    def __init__(self, prov_name):
        self.provision = prov_name
        self.threshold = 0.5

    def post_process(self, doc_text, cx_prob_attrvec_list, threshold,
                     provision=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(cx_prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list,
                                                          threshold)

        best_effectivedate_sent = get_best_date(merged_prob_attrvec_list,
                                                self.threshold)

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
                                       # pylint: disable=line-too-long
                                       text=strutils.remove_nltab(doc_text[entity.start:entity.end])).to_dict()
                    if not first:
                        first = ant_rx
                    if has_prior_text_effective and not first_after_effective:
                        first_after_effective = ant_rx

            if first_after_effective:
                ant_result.append(first_after_effective)
            elif first:
                ant_result.append(first)

        # print("post_process, effectivedate({}) = {}".format(self.provision, ant_result))

        return ant_result


PROVISION_POSTPROC_MAP = {
    'default': DefaultPostPredictProcessing(),
    'choiceoflaw': PostPredChoiceOfLawProc(),
    'date': PostPredBestDateProc('date'),
    'effectivedate': PostPredEffectiveDateProc('effectivedate'),
    'party': PostPredPartyProc(),
    'sigdate': PostPredBestDateProc('sigdate'),
    'title': PostPredTitleProc(),
}


def obtain_postproc(provision):
    postproc = PROVISION_POSTPROC_MAP.get(provision)
    if not postproc:
        postproc = PROVISION_POSTPROC_MAP['default']
    return postproc
