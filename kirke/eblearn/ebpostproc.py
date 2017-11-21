import re
from abc import ABC, abstractmethod
from typing import List

from kirke.utils import evalutils, strutils, entityutils, stopwordutils, mathutils
from kirke.utils.ebantdoc import EbEntityType
from kirke.eblearn import ebattrvec
from kirke.ebrules import dates


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


SHORT_PROVISIONS = set(['title', 'date', 'effectivedate', 'sigdate', 'choiceoflaw'])

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
    def post_process(self, doc_text, prob_attrvec_list, threshold, provision=None, prov_human_ant_list=None):
        pass


# pylint: disable=R0903
class DefaultPostPredictProcessing(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'default'

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list,
                                                          threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
            if cx_prob_attrvec.prob >= threshold or sent_overlap:
                tmp_provision = provision if provision else self.provision
                ant_result.append(AntResult(label=tmp_provision,
                                            prob=cx_prob_attrvec.prob,
                                            start=cx_prob_attrvec.start,
                                            end=cx_prob_attrvec.end,
                                            # pylint: disable=line-too-long
                                            text=strutils.remove_nltab(cx_prob_attrvec.text[:50]) + '...'))
        return ant_result, threshold

# Note from PythonClassifier.java:
# The NER seems to pick up the bare word LLC, INC, and CORP as parties sometimes.  This RE
# defines strings that should not be considered parties.
NOT_PARTY_PAT = re.compile(r'((inc|llc|corp)\.?|p\.?\s*o\.?\s*box.*)', re.IGNORECASE)

# pylint: disable=R0903
class PostPredPartyProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'party'
        self.threshold = 0.5

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
            if cx_prob_attrvec.prob >= threshold or sent_overlap:
                for entity in cx_prob_attrvec.entities:
                    if entity.ner in {EbEntityType.PERSON.name, EbEntityType.ORGANIZATION.name}:

                        if 'agreement' in entity.text.lower() or NOT_PARTY_PAT.match(entity.text):
                            continue
                        ant_result.append(AntResult(label=self.provision,
                                                    prob=cx_prob_attrvec.prob,
                                                    start=entity.start,
                                                    end=entity.end,
                                                    text=strutils.remove_nltab(entity.text)))
        return ant_result, self.threshold

EMPLOYEE_PAT = re.compile(r'.*(Executive|Employee|employee|Officer|Chairman|you)[“"”]?\)?')

EMPLOYER_PAT = re.compile(r'.*[“"”](Company|COMPANY|Corporation|Association)[“"”]?\)?')
EMPLOYER_PAT2 = re.compile(r'.*(Employer)[“"”]?\)?', re.IGNORECASE)

LICENSEE_PAT = re.compile(r'.*(Licensee|LICENSEE|licensee|Purchaser|PURCHASER|Buyer|BUYER|Customer|CUSTOMER)[“"”]?\)?')

LICENSOR_PAT = re.compile(r'.*(Licensor|LICENSOR|licensor|Seller|SELLER|Manufacturer|MANUFACTURER|Supplier|SUPPLIER|Vendor|VENDOR)[“"”]?\)?')

BORROWER_PAT = re.compile(r'[^“"”]+[“"”](Borrower|BORROWER|borrower)[“"”]?\)?')

LENDER_PAT = re.compile(r'[^“"”]+[“"”]?(Lender|LENDER|Noteholder|Issuer|Provider|Purchase|SUBSCRIBER|Subscriber|Bank)s?[“"”]?\)?')
LENDER_PAT2 = re.compile(r'[“"”]Banks?[“"”]', re.IGNORECASE)

AGENT_PAT = re.compile(r'[^“"”]+[“"”]?(Agent|AGENT|Arranger)s?[“"”]?\)?')

DEBUG_MODE = False

def pick_best_provision(found_provision_list, has_x3=False):

    for found_provision in found_provision_list:
        prov_st, prov_start, prov_end, match_type = found_provision

        if DEBUG_MODE:
            print("   --check prov {} ({}, {}): {}".format(match_type,
                                                           prov_start,
                                                           prov_end,
                                                           prov_st))
    if not found_provision_list:
        return found_provision_list
    final_list = []
    rest_list = []
    for found_provision in found_provision_list:
        prov_st, prov_start, prov_end, match_type = found_provision
        if match_type == 'x1':
            final_list.append(found_provision)
        else:
            rest_list.append(found_provision)
    # take the first one in final_list
    if final_list:
        return final_list[0]

    rest_list2 = []
    for found_provision in rest_list:
        prov_st, prov_start, prov_end, match_type = found_provision
        if match_type == 'x2':
            final_list.append(found_provision)
        else:
            rest_list2.append(found_provision)

    # take the first one in final_list
    if final_list:
        return final_list[0]

    if has_x3:
        return rest_list2[0]
    return []


def extract_ea_employer(sent_start, sent_end, attrvec_entities, doc_text):
    is_provision_found = False
    prov_end_start_map = {}
    found_provision_list = []
    person_after_list = []

    for entity in attrvec_entities:
        if (entity.ner == 'ORGANIZATION' and
            mathutils.start_end_overlap((entity.start, entity.end),
                                        (sent_start, sent_end))):
            # print("  entities: {}".format(entity))
            if DEBUG_MODE:
                print("  entities: {}".format(entity))

            entity_doc_st = doc_text[entity.start:entity.end]
            if ("Company" in entity_doc_st or
                "COMPANY" in entity_doc_st or
                "Corporation" in entity_doc_st or
                "CORPORATION" in entity_doc_st or
                "Employer" in entity_doc_st or
                "EMPLOYER" in entity_doc_st):
                found_provision_list.append((entity_doc_st,
                                             entity.start,
                                             entity.end, 'x1'))
                is_provision_found = True
            person_after_list.append(entity.end)
            prov_end_start_map[entity.end] = entity.start

    if not is_provision_found:
        person_after_list.append(sent_end)
        for i, person_end in enumerate(person_after_list[:-1]):
            person_after_sent = doc_text[person_end:person_after_list[i+1]].replace(r'[\n\r]', ' ')
            if DEBUG_MODE:
                print("  person_xafter_st: [{}]".format(person_after_sent))

            mat = EMPLOYER_PAT.search(person_after_sent)
            if mat:
                prov_start = prov_end_start_map[person_end]
                prov_end = person_end + mat.end()
                found_doc_st = doc_text[prov_start:prov_end]
                found_provision_list.append((found_doc_st,
                                             prov_start,
                                             prov_end, 'x2'))
                is_provision_found = True

            if not is_provision_found:
                mat2 = EMPLOYER_PAT2.search(person_after_sent)
                if mat2:
                    prov_start = prov_end_start_map[person_end]
                    prov_end = person_end + mat2.end()
                    found_doc_st = doc_text[prov_start:prov_end]
                    found_provision_list.append((found_doc_st,
                                                 prov_start,
                                                 prov_end, 'x3'))
                    is_provision_found = True

    best_provision = pick_best_provision(found_provision_list, has_x3=True)

    if best_provision:
        prov_st, prov_start, prov_end, match_type = best_provision
        if DEBUG_MODE:
            print("*** found prov {} ({}, {}): {}".format(match_type,
                                                          prov_start,
                                                          prov_end,
                                                          prov_st))
        return best_provision

    return None


def extract_ea_employee(sent_start, sent_end, attrvec_entities, doc_text):
    is_provision_found = False
    prov_end_start_map = {}
    found_provision_list = []
    person_after_list = []

    for entity in attrvec_entities:
        if (entity.ner == 'PERSON' and
            mathutils.start_end_overlap((entity.start, entity.end),
                                        (sent_start, sent_end))):
            # print("  entities: {}".format(entity))
            if DEBUG_MODE:
                print("  entities: {}".format(entity))

            entity_doc_st = doc_text[entity.start:entity.end]
            if ("Executive" in entity_doc_st or
                "Employee" in entity_doc_st):
                found_provision_list.append((entity_doc_st,
                                             entity.start,
                                             entity.end, 'x1'))
                is_provision_found = True
            person_after_list.append(entity.end)
            prov_end_start_map[entity.end] = entity.start

    if not is_provision_found:
        person_after_list.append(sent_end)
        for i, person_end in enumerate(person_after_list[:-1]):
            person_after_sent = doc_text[person_end:person_after_list[i+1]].replace(r'[\n\r]', ' ')
            if DEBUG_MODE:
                print("  person_xafter_st: [{}]".format(person_after_sent))

            mat = EMPLOYEE_PAT.search(person_after_sent)
            if mat:
                prov_start = prov_end_start_map[person_end]
                prov_end = person_end + mat.end()
                found_doc_st = doc_text[prov_start:prov_end]
                found_provision_list.append((found_doc_st,
                                             prov_start,
                                             prov_end, 'x2'))
                is_provision_found = True

    best_provision = pick_best_provision(found_provision_list)

    if best_provision:
        prov_st, prov_start, prov_end, match_type = best_provision
        if DEBUG_MODE:
            print("*** found prov {} ({}, {}): {}".format(match_type,
                                                          prov_start,
                                                          prov_end,
                                                          prov_st))
        return best_provision

    return None


def extract_lic_licensee(sent_start, sent_end, attrvec_entities, doc_text):
    is_provision_found = False
    prov_end_start_map = {}
    found_provision_list = []
    person_after_list = []
    person_before_list = []

    for entity in attrvec_entities:
        if ((entity.ner == 'ORGANIZATION' or entity.ner == 'PERSON') and
            mathutils.start_end_overlap((entity.start, entity.end),
                                        (sent_start, sent_end))):
            # print("  entities: {}".format(entity))
            if DEBUG_MODE:
                print("  entities: {}".format(entity))

            entity_doc_st = doc_text[entity.start:entity.end]
            if ("Licensee" in entity_doc_st or
                "LICENSEE" in entity_doc_st):
                found_provision_list.append((entity_doc_st,
                                             entity.start,
                                             entity.end, 'x1'))
                is_provision_found = True
            person_after_list.append(entity.end)
            prov_end_start_map[entity.end] = entity.start
            person_before_list.append((entity.start, entity.end))

    if not is_provision_found:
        person_after_list.append(sent_end)
        for i, person_end in enumerate(person_after_list[:-1]):
            person_after_sent = doc_text[person_end:person_after_list[i+1]].replace(r'[\n\r]', ' ')
            if DEBUG_MODE:
                print("  person_xafter_st: [{}]".format(person_after_sent))

            mat = LICENSEE_PAT.search(person_after_sent)
            if mat:
                prov_start = prov_end_start_map[person_end]
                prov_end = person_end + mat.end()
                found_doc_st = doc_text[prov_start:prov_end]
                found_provision_list.append((found_doc_st,
                                             prov_start,
                                             prov_end, 'x2'))
                is_provision_found = True

            if not is_provision_found:
                for person_before, person_end in person_before_list:
                    if person_before != sent_start:
                        person_before_sent = doc_text[sent_start:person_before].replace(r'[\n\r]', ' ')
                        last_word_mat = re.search(r'\S+\s*$', person_before_sent)
                        if last_word_mat:
                            if 'licensee' in last_word_mat.group().lower():
                                prov_start = sent_start + last_word_mat.start()
                                prov_end = person_end
                                found_doc_st = doc_text[prov_start:prov_end]
                                found_provision_list.append((found_doc_st,
                                                             prov_start,
                                                             prov_end, 'x3'))
                                is_provision_found = True

    best_provision = pick_best_provision(found_provision_list, has_x3=True)

    if best_provision:
        prov_st, prov_start, prov_end, match_type = best_provision
        if DEBUG_MODE:
            print("*** found prov {} ({}, {}): {}".format(match_type,
                                                          prov_start,
                                                          prov_end,
                                                          prov_st))
        return best_provision

    return None


def extract_lic_licensor(sent_start, sent_end, attrvec_entities, doc_text):
    is_provision_found = False
    prov_end_start_map = {}
    found_provision_list = []
    person_after_list = []
    person_before_list = []

    for entity in attrvec_entities:
        if ((entity.ner == 'ORGANIZATION' or entity.ner == 'PERSON') and
            mathutils.start_end_overlap((entity.start, entity.end),
                                        (sent_start, sent_end))):
            # print("  entities: {}".format(entity))
            if DEBUG_MODE:
                print("  entities: {}".format(entity))

            entity_doc_st = doc_text[entity.start:entity.end]
            if ("Licensor" in entity_doc_st or
                "LICENSOR" in entity_doc_st or
                "Manufacturer" in entity_doc_st or
                "MANUFACTURER" in entity_doc_st or
                "Supplier" in entity_doc_st or
                "SUPPLIER" in entity_doc_st or
                "Vendor" in entity_doc_st or
                "VENDOR" in entity_doc_st or
                "Seller" in entity_doc_st or
                "SELLER" in entity_doc_st):
                found_provision_list.append((entity_doc_st,
                                             entity.start,
                                             entity.end, 'x1'))
                is_provision_found = True
            person_after_list.append(entity.end)
            prov_end_start_map[entity.end] = entity.start
            person_before_list.append((entity.start, entity.end))

    if not is_provision_found:
        person_after_list.append(sent_end)
        for i, person_end in enumerate(person_after_list[:-1]):
            person_after_sent = doc_text[person_end:person_after_list[i+1]].replace(r'[\n\r]', ' ')
            if DEBUG_MODE:
                print("  person_xafter_st: [{}]".format(person_after_sent))

            mat = LICENSOR_PAT.search(person_after_sent)
            if mat:
                prov_start = prov_end_start_map[person_end]
                prov_end = person_end + mat.end()
                found_doc_st = doc_text[prov_start:prov_end]
                found_provision_list.append((found_doc_st,
                                             prov_start,
                                             prov_end, 'x2'))
                is_provision_found = True

            if not is_provision_found:
                for person_before, person_end in person_before_list:
                    if person_before != sent_start:
                        person_before_sent = doc_text[sent_start:person_before].replace(r'[\n\r]', ' ')
                        last_word_mat = re.search(r'\S+\s*$', person_before_sent)
                        if last_word_mat:
                            if 'licensor' in last_word_mat.group().lower():
                                prov_start = sent_start + last_word_mat.start()
                                prov_end = person_end
                                found_doc_st = doc_text[prov_start:prov_end]
                                found_provision_list.append((found_doc_st,
                                                             prov_start,
                                                             prov_end, 'x3'))
                                is_provision_found = True

    best_provision = pick_best_provision(found_provision_list, has_x3=True)

    if best_provision:
        prov_st, prov_start, prov_end, match_type = best_provision
        if DEBUG_MODE:
            print("*** found prov {} ({}, {}): {}".format(match_type,
                                                          prov_start,
                                                          prov_end,
                                                          prov_st))
        return best_provision

    return None


def extract_la_borrower(sent_start, sent_end, attrvec_entities, doc_text):
    is_provision_found = False
    prov_end_start_map = {}
    found_provision_list = []
    person_after_list = []
    person_before_list = []

    for entity in attrvec_entities:
        if ((entity.ner == 'ORGANIZATION' or entity.ner == 'PERSON') and
            mathutils.start_end_overlap((entity.start, entity.end),
                                        (sent_start, sent_end))):
            # print("  entities: {}".format(entity))
            if DEBUG_MODE:
                print("  entities: {}".format(entity))

            entity_doc_st = doc_text[entity.start:entity.end]
            if ("Borrower" in entity_doc_st or
                "BORROWER" in entity_doc_st) and re.search(r'[“"”]', entity_doc_st):
                found_provision_list.append((entity_doc_st,
                                             entity.start,
                                             entity.end, 'x1'))
                is_provision_found = True
            person_after_list.append(entity.end)
            prov_end_start_map[entity.end] = entity.start
            person_before_list.append((entity.start, entity.end))

    if not is_provision_found:
        person_after_list.append(sent_end)
        for i, person_end in enumerate(person_after_list[:-1]):
            person_after_sent = doc_text[person_end:person_after_list[i+1]].replace(r'[\n\r]', ' ')
            if DEBUG_MODE:
                print("  person_xafter_st: [{}]".format(person_after_sent))

            mat = BORROWER_PAT.search(person_after_sent)
            if mat:
                prov_start = prov_end_start_map[person_end]
                prov_end = person_end + mat.end()
                found_doc_st = doc_text[prov_start:prov_end]
                found_provision_list.append((found_doc_st,
                                             prov_start,
                                             prov_end, 'x2'))
                is_provision_found = True

    best_provision = pick_best_provision(found_provision_list)

    if best_provision:
        prov_st, prov_start, prov_end, match_type = best_provision
        if DEBUG_MODE:
            print("*** found prov {} ({}, {}): {}".format(match_type,
                                                          prov_start,
                                                          prov_end,
                                                          prov_st))
        return best_provision

    return None


def extract_la_lender(sent_start, sent_end, attrvec_entities, doc_text):
    is_provision_found = False
    prov_end_start_map = {}
    found_provision_list = []
    person_after_list = []
    person_before_list = []

    for entity in attrvec_entities:
        if ((entity.ner == 'ORGANIZATION' or entity.ner == 'PERSON') and
            mathutils.start_end_overlap((entity.start, entity.end),
                                        (sent_start, sent_end))):
            # print("  entities: {}".format(entity))
            if DEBUG_MODE:
                print("  entities: {}".format(entity))

            entity_doc_st = doc_text[entity.start:entity.end]
            if ("Lender" in entity_doc_st or
                "Noteholder" in entity_doc_st or
                "Issuer" in entity_doc_st or
                "Bank" in entity_doc_st or
                "Subscriber" in entity_doc_st or
                "SUBSCRIBER" in entity_doc_st or
                "Provider" in entity_doc_st or
                "Purchaser" in entity_doc_st or
                "LENDER" in entity_doc_st):
                found_provision_list.append((entity_doc_st,
                                             entity.start,
                                             entity.end, 'x1'))
                is_provision_found = True
            person_after_list.append(entity.end)
            prov_end_start_map[entity.end] = entity.start
            person_before_list.append((entity.start, entity.end))

    if not is_provision_found:
        person_after_list.append(sent_end)
        for i, person_end in enumerate(person_after_list[:-1]):
            person_after_sent = doc_text[person_end:person_after_list[i+1]].replace(r'[\n\r]', ' ')
            if DEBUG_MODE:
                print("  person_xafter_st: [{}]".format(person_after_sent))

            mat = LENDER_PAT.search(person_after_sent)
            if mat:
                prov_start = prov_end_start_map[person_end]
                prov_end = person_end + mat.end()
                found_doc_st = doc_text[prov_start:prov_end]
                found_provision_list.append((found_doc_st,
                                             prov_start,
                                             prov_end, 'x2'))
                is_provision_found = True

            if not is_provision_found:
                mat = LENDER_PAT2.search(person_after_sent)
                if mat:
                    prov_start = prov_end_start_map[person_end]
                    prov_end = person_end + mat.end()
                    found_doc_st = doc_text[prov_start:prov_end]
                    found_provision_list.append((found_doc_st,
                                                 prov_start,
                                                 prov_end, 'x3'))
                    is_provision_found = True

    best_provision = pick_best_provision(found_provision_list, has_x3=True)

    if best_provision:
        prov_st, prov_start, prov_end, match_type = best_provision
        if DEBUG_MODE:
            print("*** found prov {} ({}, {}): {}".format(match_type,
                                                          prov_start,
                                                          prov_end,
                                                          prov_st))
        return best_provision

    return None


def extract_la_agent_trustee(sent_start, sent_end, attrvec_entities, doc_text):
    is_provision_found = False
    prov_end_start_map = {}
    found_provision_list = []
    person_after_list = []
    person_before_list = []

    for entity in attrvec_entities:
        if ((entity.ner == 'ORGANIZATION' or entity.ner == 'PERSON') and
            mathutils.start_end_overlap((entity.start, entity.end),
                                        (sent_start, sent_end))):
            # print("  entities: {}".format(entity))
            if DEBUG_MODE:
                print("  entities: {}".format(entity))

            entity_doc_st = doc_text[entity.start:entity.end]
            if ("Agent" in entity_doc_st or
                "Noteholder" in entity_doc_st or
                "Issuer" in entity_doc_st or
                # "Bank" in entity_doc_st or
                "Subscriber" in entity_doc_st or
                "SUBSCRIBER" in entity_doc_st or
                "Provider" in entity_doc_st or
                "Purchaser" in entity_doc_st or
                "AGENT" in entity_doc_st) and len(entity_doc_st.split()) > 1:
                # the last number of token check is for "KeyBank"  
                found_provision_list.append((entity_doc_st,
                                             entity.start,
                                             entity.end, 'x1'))
                is_provision_found = True
            person_after_list.append(entity.end)
            prov_end_start_map[entity.end] = entity.start
            person_before_list.append((entity.start, entity.end))

    if not is_provision_found:
        person_after_list.append(sent_end)
        for i, person_end in enumerate(person_after_list[:-1]):
            person_after_sent = doc_text[person_end:person_after_list[i+1]].replace(r'[\n\r]', ' ')
            if DEBUG_MODE:
                print("  person_xafter_st: [{}]".format(person_after_sent))

            mat = AGENT_PAT.search(person_after_sent)
            if mat:
                prov_start = prov_end_start_map[person_end]
                prov_end = person_end + mat.end()
                found_doc_st = doc_text[prov_start:prov_end]
                found_provision_list.append((found_doc_st,
                                             prov_start,
                                             prov_end, 'x2'))
                is_provision_found = True

    best_provision = pick_best_provision(found_provision_list, has_x3=True)

    if best_provision:
        prov_st, prov_start, prov_end, match_type = best_provision
        if DEBUG_MODE:
            print("*** found prov {} ({}, {}): {}".format(match_type,
                                                          prov_start,
                                                          prov_end,
                                                          prov_st))
        return best_provision

    return None


# pylint: disable=R0903
class PostPredEaEmployerProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'ea_employer'

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
            if cx_prob_attrvec.prob >= threshold or sent_overlap:
                employer_matched_span = extract_ea_employer(cx_prob_attrvec.start,
                                                            cx_prob_attrvec.end,
                                                            cx_prob_attrvec.entities,
                                                            doc_text)
                if employer_matched_span:
                    prov_st, prov_start, prov_end, match_type = employer_matched_span
                    ant_result.append(AntResult(label=self.provision,
                                                prob=cx_prob_attrvec.prob,
                                                start=prov_start,
                                                end=prov_end,
                                                # pylint: disable=line-too-long
                                                text=strutils.remove_nltab(prov_st)))
                    break
        return ant_result, threshold


# pylint: disable=R0903
class PostPredEaEmployeeProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'ea_employee'

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
            if cx_prob_attrvec.prob >= threshold or sent_overlap:
                employee_matched_span = extract_ea_employee(cx_prob_attrvec.start,
                                                            cx_prob_attrvec.end,
                                                            cx_prob_attrvec.entities,
                                                            doc_text)
                if employee_matched_span:
                    prov_st, prov_start, prov_end, match_type = employee_matched_span
                    ant_result.append(AntResult(label=self.provision,
                                                prob=cx_prob_attrvec.prob,
                                                start=prov_start,
                                                end=prov_end,
                                                # pylint: disable=line-too-long
                                                text=strutils.remove_nltab(prov_st)))
                    break
        return ant_result, threshold

# pylint: disable=R0903
class PostPredLicLicenseeProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'lic_licensee'

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
            if cx_prob_attrvec.prob >= threshold or sent_overlap:
                licensee_matched_span = extract_lic_licensee(cx_prob_attrvec.start,
                                                             cx_prob_attrvec.end,
                                                             cx_prob_attrvec.entities,
                                                             doc_text)
                if licensee_matched_span:
                    prov_st, prov_start, prov_end, match_type = licensee_matched_span
                    ant_result.append(AntResult(label=self.provision,
                                                prob=cx_prob_attrvec.prob,
                                                start=prov_start,
                                                end=prov_end,
                                                # pylint: disable=line-too-long
                                                text=strutils.remove_nltab(prov_st)))
                    break
        return ant_result, threshold


# pylint: disable=R0903
class PostPredLicLicensorProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'lic_licensor'

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
            if cx_prob_attrvec.prob >= threshold or sent_overlap:
                licensor_matched_span = extract_lic_licensor(cx_prob_attrvec.start,
                                                             cx_prob_attrvec.end,
                                                             cx_prob_attrvec.entities,
                                                             doc_text)
                if licensor_matched_span:
                    prov_st, prov_start, prov_end, match_type = licensor_matched_span
                    ant_result.append(AntResult(label=self.provision,
                                                prob=cx_prob_attrvec.prob,
                                                start=prov_start,
                                                end=prov_end,
                                                # pylint: disable=line-too-long
                                                text=strutils.remove_nltab(prov_st)))
                    break
        return ant_result, threshold


# pylint: disable=R0903
class PostPredLaBorrowerProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'la_borrower'

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
            if cx_prob_attrvec.prob >= threshold or sent_overlap:
                borrower_matched_span = extract_la_borrower(cx_prob_attrvec.start,
                                                             cx_prob_attrvec.end,
                                                             cx_prob_attrvec.entities,
                                                             doc_text)
                if borrower_matched_span:
                    prov_st, prov_start, prov_end, match_type = borrower_matched_span
                    ant_result.append(AntResult(label=self.provision,
                                                prob=cx_prob_attrvec.prob,
                                                start=prov_start,
                                                end=prov_end,
                                                # pylint: disable=line-too-long
                                                text=strutils.remove_nltab(prov_st)))
                    break
        return ant_result, threshold


# pylint: disable=R0903
class PostPredLaLenderProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'la_lender'

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
            if cx_prob_attrvec.prob >= threshold or sent_overlap:
                lender_matched_span = extract_la_lender(cx_prob_attrvec.start,
                                                             cx_prob_attrvec.end,
                                                             cx_prob_attrvec.entities,
                                                             doc_text)
                if lender_matched_span:
                    prov_st, prov_start, prov_end, match_type = lender_matched_span
                    ant_result.append(AntResult(label=self.provision,
                                                prob=cx_prob_attrvec.prob,
                                                start=prov_start,
                                                end=prov_end,
                                                # pylint: disable=line-too-long
                                                text=strutils.remove_nltab(prov_st)))
                    break
        return ant_result, threshold


# pylint: disable=R0903
class PostPredLaAgentTrusteeProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'la_agent_trustee'

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
            if cx_prob_attrvec.prob >= threshold:
                agent_trustee_matched_span = extract_la_agent_trustee(cx_prob_attrvec.start,
                                                                      cx_prob_attrvec.end,
                                                                      cx_prob_attrvec.entities,
                                                                      doc_text)
                if agent_trustee_matched_span:
                    prov_st, prov_start, prov_end, match_type = agent_trustee_matched_span
                    ant_result.append(AntResult(label=self.provision,
                                                prob=cx_prob_attrvec.prob,
                                                start=prov_start,
                                                end=prov_end,
                                                # pylint: disable=line-too-long
                                                text=strutils.remove_nltab(prov_st)))
                    break
        return ant_result, threshold


class PostPredChoiceOfLawProc(EbPostPredictProcessing):

    def __init__(self):
        self.provision = 'choiceoflaw'

    def post_process(self, doc_text, prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
            if cx_prob_attrvec.prob >= threshold or sent_overlap:
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
                                                    text=tmp_state))
                else:
                    ant_result.append(AntResult(label=self.provision,
                                                prob=cx_prob_attrvec.prob,
                                                start=cx_prob_attrvec.start,
                                                end=cx_prob_attrvec.end,
                                                text=anttext))
        return ant_result, threshold


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
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(cx_prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list, threshold)

        ant_result = []
        for cx_prob_attrvec in merged_prob_attrvec_list:
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
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
                                                text=tmp_title))
                    return ant_result, threshold
        return ant_result, threshold


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
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(cx_prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list,
                                                          threshold)

        best_date_sent = get_best_date(merged_prob_attrvec_list, threshold)
        ant_result = []
        if best_date_sent:
            for entity in best_date_sent.entities:
                if entity.ner == EbEntityType.DATE.name:
                    ant_result.append(AntResult(label=self.provision,
                                      prob=best_date_sent.prob,
                                      start=entity.start,
                                      end=entity.end,
                                      # pylint: disable=line-too-long
                                      text=strutils.remove_nltab(doc_text[entity.start:entity.end])))
                    return ant_result, self.threshold
        return ant_result, self.threshold


class PostPredEffectiveDateProc(EbPostPredictProcessing):

    def __init__(self, prov_name):
        self.provision = prov_name
        self.threshold = 0.5

    def post_process(self, doc_text, cx_prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(cx_prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list,
                                                          threshold)
        best_effectivedate_sent = get_best_date(merged_prob_attrvec_list, threshold)
        ant_result = []
        if best_effectivedate_sent:
            first = None
            first_after_effective = None
            for entity in best_effectivedate_sent.entities:
                if entity.ner == EbEntityType.DATE.name:
                    prior_text = doc_text[best_effectivedate_sent.start:entity.start]
                    has_prior_text_effective = 'effective' in prior_text.lower()
                    ant_rx = AntResult(label=self.provision,
                                       prob=best_effectivedate_sent.prob,
                                       start=entity.start,
                                       end=entity.end,
                                       # pylint: disable=line-too-long
                                       text=strutils.remove_nltab(doc_text[entity.start:entity.end]))
                    if not first:
                        first = ant_rx
                    if has_prior_text_effective and not first_after_effective:
                        first_after_effective = ant_rx
 
            if first_after_effective:
                ant_result.append(first_after_effective)
            elif first:
                ant_result.append(first)
        return ant_result, self.threshold

class PostPredLeaseDateProc(EbPostPredictProcessing):

    # Class (static) variables for keywords; Rent Commencement Date != C. D.
    lc = {
        'verbs': '|'.join(['commence', 'commences', 'commencing', 'commenced',
                           'begin', 'begins', 'beginning', 'begun']),
        'nouns': '|'.join(['commencement']),
        'noun_revokers': '|'.join(['rent'])
    }
    le = {
        'verbs': '|'.join(['expire', 'expires', 'expiring', 'expired',
                           'terminate', 'terminates', 'terminating',
                           'terminated', 'cancel', 'canceled', 'cancelled',
                           'end', 'ends', 'ending', 'ended']),
        'nouns': '|'.join(['expiration', 'termination']),
        'noun_revokers': '|'.join(['rent'])
    }
    revokers = '|'.join(['earlier', 'earliest', 'later', 'first', 'last',
                         'former', 'latter', 'previous', 'prior', 'sooner',
                         '\(a\)', '\(i\)', '\(1\)'])

    # Compile regular expressions once
    revokers_regex = re.compile(r'\b{}\b'.format(revokers), re.I)
    lc_regexes = {
        'verbs': re.compile(r'\b{}\b'.format(lc['verbs']), re.I),
        'terms': re.compile(r'\([^)]*?\b{}\s+date\b'.format(lc['nouns']), re.I),
        'nouns': re.compile(r'(?<!{})\s*{}\s+date\b'
                            .format(lc['noun_revokers'], lc['nouns']), re.I),
        'end': re.compile(r'^\s*\S*\s*(?<!{})\s*{}\s+date\s*:?\s*$'
                          .format(lc['noun_revokers'], lc['nouns']), re.I),
        'revokers': revokers_regex
    }
    le_regexes = {
        'verbs': re.compile(r'\b{}\b'.format(le['verbs']), re.I),
        'terms': re.compile(r'\([^)]*?\b{}\s+date\b'.format(le['nouns']), re.I),
        'nouns': re.compile(r'(?<!{})\s*{}\s+date\b'
                            .format(le['noun_revokers'], le['nouns']), re.I),
        'end': re.compile(r'^\s*\S*\s*(?<!{})\s*{}\s+date\s*:?\s*$'
                          .format(le['noun_revokers'], le['nouns']), re.I),
        'revokers': revokers_regex
    }

    def __init__(self, prov):
        """Currently supports both commencement and expiration dates."""
        self.provision = prov
        self.regexes = (PostPredLeaseDateProc.lc_regexes
                        if prov == 'l_commencement_date'
                        else PostPredLeaseDateProc.le_regexes)
        self.stop_at_one_date = False
        self.threshold = 0.24

    def ant(self, line, cx_prob_attrvec, date):
        """Compiles an ant_result."""
        text = strutils.remove_nltab(line[date[0]:date[1]][:50]) + '...'
        return AntResult(label=self.provision, prob=cx_prob_attrvec.prob,
                         start=cx_prob_attrvec.start + date[0],
                         end=cx_prob_attrvec.start + date[1],
                         text=text)

    def post_process(self, doc_text, cx_prob_attrvec_list, threshold,
                     provision=None, prov_human_ant_list=None) -> List[AntResult]:
        cx_prob_attrvec_list = to_cx_prob_attrvecs(cx_prob_attrvec_list)
        merged_prob_attrvec_list = merge_cx_prob_attrvecs(cx_prob_attrvec_list,
                                                          threshold)
        ant_result = []
        for i, cx_prob_attrvec in enumerate(merged_prob_attrvec_list):
            sent_overlap = evalutils.find_annotation_overlap(cx_prob_attrvec.start, cx_prob_attrvec.end, prov_human_ant_list)
            line = doc_text[cx_prob_attrvec.start:cx_prob_attrvec.end]
            if not (cx_prob_attrvec.prob >= threshold
                    or self.regexes['end'].search(line)):
                continue

            # Get date start, end offsets relative to current line
            date_list = sorted(dates.extract_std_dates(line))
            date_found = False

            # If an l_commencement verb is in the line, take next date
            for mat in self.regexes['verbs'].finditer(line):
                date = next((d for d in date_list if d[0] > mat.end()), None)
                if date:
                    between_verb_date = line[mat.end():date[0]]
                    if not self.regexes['revokers'].search(between_verb_date):
                        date_found = True
                        ant_result.append(self.ant(line, cx_prob_attrvec, date))
                        break
            if date_found:
                # If stopping when we find a date, return only this date
                if self.stop_at_one_date:
                    return [ant_result[-1]], self.threshold
                continue
 
            # If an l_commencement term is in the line, take previous date
            for mat in self.regexes['terms'].finditer(line):
                rv_dates = reversed(date_list)
                date = next((d for d in rv_dates if d[1] < mat.start()), None)
                if date:
                    between_term_date = line[mat.end():date[0]]
                    if not self.regexes['revokers'].search(between_term_date):
                        date_found = True
                        ant_result.append(self.ant(line, cx_prob_attrvec, date))
                        break
            if date_found:
                if self.stop_at_one_date:
                    return [ant_result[-1]], self.threshold
                continue
 
            # If there is an l_commencement non-term noun, take next date
            for mat in self.regexes['nouns'].finditer(line):
                date = next((d for d in date_list if d[0] > mat.end()), None)
                if date:
                    between_noun_date = line[mat.end():date[0]]
                    if not self.regexes['revokers'].search(between_noun_date):
                        date_found = True
                        ant_result.append(self.ant(line, cx_prob_attrvec, date))
                        break
            if date_found:
                if self.stop_at_one_date:
                    return [ant_result[-1]], self.threshold
                continue
 
            # If no date found and next line starts with a date, return that
            if i + 1 < len(merged_prob_attrvec_list):
                next_attrvec = merged_prob_attrvec_list[i + 1]
                # Don't want to repeat a date, but does not matter if stopping
                if next_attrvec.prob < threshold or self.stop_at_one_date:
                    next_line = doc_text[next_attrvec.start:next_attrvec.end]
                    next_date_list = dates.extract_std_dates(next_line)
                    if next_date_list:
                        # Ensure no alphanumeric chars left or right (lr)
                        next_date = sorted(next_date_list)[0]
                        lr = next_line[:next_date[0]] + next_line[next_date[1]:]
                        if not any(c.isalnum() for c in lr):
                            next_prob = next_attrvec.prob
                            next_attrvec.prob = 1.0
                            ant_result.append(self.ant(next_line, next_attrvec,
                                                       next_date))
                            next_attrvec.prob = next_prob
                            if self.stop_at_one_date:
                                return [ant_result[-1]], self.threshold
                            continue
            if cx_prob_attrvec.prob >= threshold:
                ant_result.append(self.ant(line, cx_prob_attrvec,
                                           (0, len(line))))
        return ant_result, self.threshold
    

PROVISION_POSTPROC_MAP = {
    'default': DefaultPostPredictProcessing(),
    'choiceoflaw': PostPredChoiceOfLawProc(),
    'date': PostPredBestDateProc('date'),
    'ea_employee': PostPredEaEmployeeProc(),
    'ea_employer': PostPredEaEmployerProc(),
    # The classifier label is "effectivedate", but 'extractor' is expecting
    # 'effectivedate_auto'
    'effectivedate': PostPredEffectiveDateProc('effectivedate_auto'),
    'la_borrower': PostPredLaBorrowerProc(),
    'la_lender': PostPredLaLenderProc(),
    'la_agent_trustee': PostPredLaAgentTrusteeProc(),
    'lic_licensee': PostPredLicLicenseeProc(),
    'lic_licensor': PostPredLicLicensorProc(),
    'l_commencement_date': PostPredLeaseDateProc('l_commencement_date'),
    'l_expiration_date': PostPredLeaseDateProc('l_expiration_date'),
    'party': PostPredPartyProc(),
    'sigdate': PostPredBestDateProc('sigdate'),
    'title': PostPredTitleProc(),
}


def obtain_postproc(provision):
    postproc = PROVISION_POSTPROC_MAP.get(provision)
    if not postproc:
        postproc = PROVISION_POSTPROC_MAP['default']
    return postproc
