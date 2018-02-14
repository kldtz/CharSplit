from collections import namedtuple
from enum import Enum
import json
import os
from pathlib import Path
import re

from kirke.utils import stopwordutils, mathutils, entityutils
from kirke.docstruct import docutils

class EbEntityType(Enum):
    PERSON = 1
    ORGANIZATION = 2
    LOCATION = 3
    DATE = 4
    DEFINE_TERM = 5


EB_PROVISION_ST_LIST = ['party', 'date', 'title', 'change_control', 'assign',
                        'indemnify', 'sublicense', 'securities_transfer',
                        'assign_lender', 'assign_landlord_owner',
                        'agent_trustee_assign', 'loan_prepay',
                        'loan_term_ex', 'renewal', 'term', 'termination',
                        'limliability', 'choiceoflaw', 'arbitration',
                        'jurisdiction', 'events_default', 'nonsolicit',
                        'amending_agreement', 'closing_date', 'confidentiality',
                        'equitable_relief', 'escrow', 'exclusivity',
                        'force_majeure', 'guaranty', 'insurance',
                        'jury_trial', 'note_preamble', 'preamble', 'survival',
                        'third_party_bene', 'l_estoppel_cert', 'l_term',
                        'l_mutual_waiver_subrogation', 'l_no_lien',
                        'l_no_recordation_lease', 'l_address',
                        'l_quiet_enjoy', 'l_time_essence', 'l_address_only',
                        'l_premises', 'l_square_footage', 'l_execution_date',
                        'l_commencement_date', 'l_expiration_date', 'l_rent',
                        'l_operating_escalation', 'l_re_tax',
                        'l_electric_charges', 'l_security', 'l_default_rate',
                        'l_renewal_opt', 'l_expansion_opt', 'l_early_term',
                        'l_building_services', 'l_holdover', 'l_brokers',
                        'l_permitted_use', 'l_parking', 'l_tenant_assignment',
                        'l_estoppel_cert_only', 'l_snda', 'l_tenant_notice',
                        'l_condemnation_term', 'l_damage_term',
                        'l_landlord_repair', 'l_tenant_repair',
                        'l_lessee_eod', 'l_lessor_eod']
EB_PROVISION_ST_SET = set(EB_PROVISION_ST_LIST)


EbEntityTuple = namedtuple('EbEntityTuple', ['start', 'end', 'ner', 'text'])


class EbEntity:
    __slots__ = ['start', 'end', 'ner', 'text']

    def __init__(self, start, end, ner, text):
        self.start = start
        self.end = end
        self.ner = ner
        self.text = text

    def to_tuple(self):
        return EbEntityTuple(self.start, self.end, self.ner, self.text)

    def __str__(self):
        return str((self.ner, self.start, self.end, self.text))

    def to_dict(self):
        return {'ner': self.ner,
                'start': self.start,
                'end': self.end,
                'text': self.text}


def entities_to_dict_list(entities):
    if entities:
        return [entity.to_dict() for entity in entities]
    return []


# from kirke.utils import corenlputils, ebantdoc, mathutils, strutils, osutils, entityutils, txtreader

_WANTED_ENTITY_NAMES = {EbEntityType.PERSON.name,
                        EbEntityType.ORGANIZATION.name,
                        EbEntityType.LOCATION.name,
                        EbEntityType.DATE.name,
                        EbEntityType.DEFINE_TERM.name}

INCORRECT_CORENLP_ENTITIES = {
    'Service', 'Confidential Information',
    'Confidential Information Agreement',
    'Employment Agreement', 'Employment Agreement',
    'Intellectual Property',
    'Notice of Termination', 'Territory', 'Base Salary',
    'Company for Cause', 'Term of Employment', 'Treasury Regulations',
    'Treas. Reg', 'Confidential Treatment Requested',
    'Termination of Employment of Executive', 'Field of Use',
    'Change in Control', 'Research Project', 'Peer Executives',
    'Executive of Executive', 'Research Plans',
    'Event of Force Majeure'}

INCORRECT_PERSON_ENTITIES = {
    'Employee', 'General Counsel', 'Developer', 'Executive',
    'Former Employee', 'Chief Financial Officer'}

INCORRECT_ORG_ENTITIES = {'Licensee', 'Landlord', 'Plaintiff', 'Named Users'}

INCORRECT_DATE_ENTITIES = {
    'Employment Period', 'Severance Period', 'Performance Period', 'Warranty Period',
    'Covenant Period', 'Effective Date', 'Remaining Unexpired Employment Period',
    'Grant Date', 'Notice Period', 'Continued Coverage Period',
    'Royalty Period', 'Non-Competition Period', 'Revocation Period',
    'Post-termination Period', 'Transition Support Period', 'Restricted Period',
    'Employer Notice Period', 'Control Period', 'Continuation Period',
    'Tool Delivery Period', 'Non-Compete Period', 'Mandatory Retirement Date',
    'Lease Commencement Date', 'Date of Termination', 'Grant Date',
    'Commencement Date', 'Termination Date', 'Disability Effective Date',
    'Lease Expiration Date', 'Good Reason Termination Date',
    'Dismissal Effective Date', 'Transition Date', 'Supply Start Date',
    'Date of Change in Control', 'Performance Share Award Grant Date',
    'Partial License Termination'}

INCORRECT_LOC_ENTITIES = {'State of New York', 'Princeton'} # ??

INCORRECT_DOMAIN_ENTITIES = {
    'Annual Performance Share Award', 'Restated Employment Agreement',
    'Employment Term', 'Equity Documents', 'Disability of Executive',
    'Security Deposit', 'Termination Without Cause',
    'Notice of Intent', 'Annual Base Salary', 'Change of Control Transaction',
    'Fair Market Value', 'Incentive Compensation', 'Reimbursement Amount',
    'Continueation Coverage Reimbursement Payments', 'Exhibit A.', 'Exhibit B.',
    'Exhibit C.', 'Excess Rent', 'Release of Claims', 'Reason of Death',
    'Notice', 'Intellectual Property Rights', 'Financial Interest',
    'Change of Control', 'Change of Control and Executive',
    'Limited Warranty', 'U.S.A.', 'El Camino Real', 'Borrower and Borrower',
    'Issuing Bank',
    'Delaware Limited Liability Company'}

_LOC_OR_ORG = {EbEntityType.ORGANIZATION.name,
               EbEntityType.LOCATION.name}

_PERSON_DFTERM_SET = set([EbEntityType.DEFINE_TERM.name,
                          EbEntityType.PERSON.name])
_ORG_DFTERM_SET = set([EbEntityType.DEFINE_TERM.name,
                       EbEntityType.ORGANIZATION.name])


def _fix_incorrect_tokens(xst, orig_label, token_list, entity_st_set, new_ner):
    if xst in entity_st_set:
        # reset the ner in those tokens
        for token in token_list:
            token.ner = new_ner
        if new_ner == 'O':  # for everyone else, return itself
            return None
        return new_ner
    return orig_label


def _tokens_to_entity(token_list):
    start = token_list[0].start
    end = token_list[-1].end
    label = token_list[0].ner
    xst = ' '.join([token.word for token in token_list])

    label = _fix_incorrect_tokens(xst, label, token_list, INCORRECT_CORENLP_ENTITIES, 'O')
    if label is None:
        return None
    label = _fix_incorrect_tokens(xst, label, token_list, INCORRECT_DOMAIN_ENTITIES, 'DOMAIN-X')
    if label == 'DOMAIN-X':
        return None

    entity_ner_set = set([token.ner for token in token_list])
    if len(entity_ner_set) > 1:
        entity_ner_set.remove(EbEntityType.DEFINE_TERM.name)
        label = entity_ner_set.pop()
    elif len(entity_ner_set) == 1 and entity_ner_set.pop() == EbEntityType.DEFINE_TERM.name:
        return None

    label = _fix_incorrect_tokens(xst, label, token_list, INCORRECT_PERSON_ENTITIES,
                                  EbEntityType.PERSON.name)
    label = _fix_incorrect_tokens(xst, label, token_list, INCORRECT_ORG_ENTITIES,
                                  EbEntityType.ORGANIZATION.name)
    label = _fix_incorrect_tokens(xst, label, token_list, INCORRECT_DATE_ENTITIES,
                                  EbEntityType.DATE.name)
    label = _fix_incorrect_tokens(xst, label, token_list, INCORRECT_LOC_ENTITIES,
                                  EbEntityType.LOCATION.name)

    return EbEntity(start, end, label, xst)


def is_distinct_ner_type(ner1, ner2):
    if ner1 == ner2:
        return False
    if (ner1 in _PERSON_DFTERM_SET and ner2 in _PERSON_DFTERM_SET):
        return False
    if (ner1 in _ORG_DFTERM_SET and ner2 in _ORG_DFTERM_SET):
        return False
    return True


def _extract_entities(tokens, wanted_ner_names):
    entity_list = []
    prev_entity_tokens = []
    prev_ner = None

    for token in tokens:
        curr_ner = token.ner
        if curr_ner in wanted_ner_names:
            if is_distinct_ner_type(curr_ner, prev_ner) and prev_entity_tokens:
                eb_entity = _tokens_to_entity(prev_entity_tokens)
                if eb_entity:
                    entity_list.append(eb_entity)
                prev_entity_tokens = []
            prev_entity_tokens.append(token)
        else:
            if prev_entity_tokens:
                eb_entity = _tokens_to_entity(prev_entity_tokens)
                if eb_entity:
                    entity_list.append(eb_entity)
                prev_entity_tokens = []
        prev_ner = curr_ner
    # for the last token, if it has desired entity
    if prev_entity_tokens:
        eb_entity = _tokens_to_entity(prev_entity_tokens)
        if eb_entity:
            entity_list.append(eb_entity)
    return entity_list


# 'POS' == "'s"
NAME_POS_SET = set(['NNS', 'CD', 'NNP', 'NN', 'POS'])

# this is destructive/in-place
def _extract_entities_v2(tokens, raw_sent_text, start_offset=0):
    ptr = -1
    max_token_ptr = len(tokens)
    # fix incorrect pos
    '''
    for token in tokens:
        if token.word == 'CORPORATE':
            token.pos = 'NNP'
    '''
    for i, token in enumerate(tokens):
        # print('{}\t{}'.format(i, token))
        if (token.word[0].isupper() and
            token.word.lower() in set(['llc.', 'llc', 'inc.', 'inc',
                                       'l.p.', 'n.a.', 'corp',
                                       'corporation', 'corp.', 'ltd.',
                                       'ltd', 'co.', 'co', 'l.l.p.',
                                       'lp', 's.a.', 'sa',
                                       'n.v.', 'plc', 'plc.', 'l.l.c.'])):
            # reset all previous tokens to ORG
            # print("I am in here")
            ptr = i
            while ptr >= 0:
                if ptr == i - 1 and tokens[ptr].word == ',':
                    tokens[ptr].ner = EbEntityType.ORGANIZATION.name
                    ptr -= 1
                    '''
                    elif tokens[ptr].pos in NAME_POS_SET:
                        # print("tokens[{}].pos = {}, {}".format(ptr, tokens[ptr].pos, tokens[ptr]))
                        tokens[ptr].ner = EbEntityType.ORGANIZATION.name
                        ptr -= 1
                    '''
                else:
                    break
        # separate "the Company and xxx"
        if (token.word in 'Company' and token.ner == EbEntityType.ORGANIZATION.name and
            (i + 1) < max_token_ptr and tokens[i+1].word == 'and' and
            tokens[i+1].ner == EbEntityType.ORGANIZATION.name):
            tokens[i+1].ner = 'O'

    pat_list = entityutils.extract_define_party(raw_sent_text, start_offset=start_offset)
    if pat_list:
        for i, token in enumerate(tokens):
            for pat in pat_list:
                if mathutils.start_end_overlap((pat[1], pat[2]), (token.start, token.end)):
                    token.ner = EbEntityType.DEFINE_TERM.name

    #print()
    #for i, token in enumerate(tokens, 1):
    #    print('x234 {}\t{}'.format(i, token))


def get_sechead_attr(attrs):
    for attr in attrs:
        # print("is_attr_section_head: {} || {}".format(attr, attr[2]))
        if (len(attr) > 3 and attr[0] == 'sechead'):
            return attr
    return ''


# this is in-place
def update_ebsents_with_sechead(ebsent_list, paras_with_attrs):
    if not ebsent_list:  # if there is no data
        return

    para_i, len_paras = 0, len(paras_with_attrs)
    ebsent_i, len_ebsents = 0, len(ebsent_list)
    ebsent = ebsent_list[ebsent_i]
    ebsent_start, ebsent_end = ebsent.start, ebsent.end

    while para_i < len_paras and ebsent_i < len_ebsents:
        span_se_list, line, attrs = paras_with_attrs[para_i]
        (para_from_start, para_from_end), (para_to_start, para_to_end) = docutils.span_frto_list_to_fromto(span_se_list)

        if para_to_start == para_to_end:  # empty line, move on
            para_i += 1
            continue
        sechead_attr = get_sechead_attr(attrs)
        if sechead_attr:
            # print("attrs: {}".format(attrs[0]))
            sechead_type, sh_prefix_num, sh_header, sh_idx = sechead_attr
        else:
            sh_header = ''
        # print("para #{}: {}".format(para_i, paras_with_attrs[para_i]))
        while ebsent_start <= para_to_end:
            if mathutils.start_end_overlap((ebsent_start, ebsent_end), (para_to_start, para_to_end)):
                # print("\tebsent set sechead ({}, {}): {}".format(ebsent_start, ebsent_end, sh_header))
                if sh_header:
                    ebsent.set_sechead(' '.join(stopwordutils.tokens_remove_stopwords([word.lower() for word in re.findall(r'\w+', sh_header)],
                                                                                      is_lower=True)))
                # else, don't even set it
            ebsent_i += 1
            if ebsent_i < len_ebsents:
                ebsent = ebsent_list[ebsent_i]
                ebsent_start, ebsent_end = ebsent.start, ebsent.end
            else:
                ebsent_start = para_to_end + 1  # end the loop
        para_i += 1
    #ebsent_i = 0
    #while ebsent_i < len_ebsents:
    #    print("sent #{}: {}".format(ebsent_i, ebsent_list[ebsent_i]))
    #    ebsent_i += 1

def populate_ebsent_entities(ebsent, raw_sent_text):
    tokens = ebsent.get_tokens()
    _extract_entities_v2(tokens, raw_sent_text, ebsent.start)
    entity_list = _extract_entities(tokens, _WANTED_ENTITY_NAMES)
    if entity_list:
        ebsent.set_entities(entity_list)


def fix_ner_tags(ebsent):
    tokens = ebsent.get_tokens()
    for token in tokens:
        if token.word == 'Lessee' and token.ner in _LOC_OR_ORG:
            token.ner = 'O'

def get_labels_if_start_end_overlap(sent_start, sent_end, ant_start_end_list):
    result_label_list = []
    for ant in ant_start_end_list:
        if mathutils.start_end_overlap((sent_start, sent_end), (ant.start, ant.end)):
            result_label_list.append(ant.label)
    return result_label_list


# cannot use this because in line 600 prov_annotation.start = xxx in ebtext2antdoc.py
# maybe fix in future.
# ProvisionAnnotation = namedtuple('ProvisionAnnotation', ['start', 'end', 'label'])
# pylint: disable=R0903
class ProvisionAnnotation:
    __slots__ = ['start', 'end', 'label']

    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label

    def __repr__(self):
        return "ProvisionAnnotation({}, {}, '{}')".format(self.start, self.end, self.label)

    def __lt__(self, other):
        return (self.start, self.end) < (other.start, other.end)

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def to_tuple(self):
        return (self.start, self.end, self.label)

              
#    def to_tuple(self):
#        return (self.lable, self.start, self.end)


# pylint: disable=R0902
class EbProvisionAnnotation:
    __slots__ = ['confidence', 'correctness', 'start', 'end',
                 'ptype', 'text', 'pid', 'custom_text']

    def __init__(self, ajson):
        self.confidence = ajson['confidence']
        self.correctness = ajson.get('correctness')
        self.start = ajson.get('start')
        self.end = ajson.get('end')
        self.ptype = ajson.get('type')  # not 'type' but 'ptype'
        self.text = ajson.get('text')
        self.pid = ajson.get('id')    # string, not 'id' but 'pid'
        self.custom_text = ajson.get('customText')  # boolean

    def to_dict(self):
        return {'confidence': self.confidence,
                'correctness': self.correctness,
                'customText': self.custom_text,
                'start': self.start,
                'end': self.end,
                'id': self.pid,
                'text': self.text,
                'type': self.ptype}

    def __str__(self):
        return str(self.to_dict())

    def to_tuple(self):
        return ProvisionAnnotation(self.start, self.end, self.ptype)


# the result is a list of
# (start, end, ant_name)
def load_prov_annotation_list(txt_file_name, cpoint_cunit_mapper, provision=None):
    prov_ant_fn = txt_file_name.replace('.txt', '.ant')
    prov_ant_file = Path(prov_ant_fn)
    prov_ebdata_fn = txt_file_name.replace('.txt', '.ebdata')
    prov_ebdata_file = Path(prov_ebdata_fn)

    prov_annotation_list = []
    is_test = False
    if os.path.exists(prov_ant_fn):
        # in is_bespoke_mode, only the annotation for a particular provision
        # is returned.
        prov_annotation_list = (load_prov_ant(prov_ant_fn, provision)
                                if prov_ant_file.is_file() else [])

    elif os.path.exists(prov_ebdata_fn):
        prov_annotation_list, is_test = (load_prov_ebdata(prov_ebdata_fn, provision)
                                         if prov_ebdata_file.is_file() else ([], False))

    # in-place update offsets
    result = []
    for eb_prov_ant in prov_annotation_list:
        eb_prov_ant.start, eb_prov_ant.end = \
            cpoint_cunit_mapper.to_codepoint_offsets(eb_prov_ant.start,
                                                     eb_prov_ant.end)
        result.append(eb_prov_ant.to_tuple())

    return result, is_test


def load_prov_ant(filename, provision_name=None):

    # logging.info('load provision %s annotation: [%s]', provision_name, filename)
    with open(filename, 'rt') as handle:
        parsed = json.load(handle)

    result = [EbProvisionAnnotation(ajson) for ajson in parsed]

    # if provision_name is specified, only return that specific provision
    if provision_name:
        return [provision_se for provision_se in result if provision_se.label == provision_name]

    return result


def load_prov_ebdata(filename, provision_name=None):
    result = []
    is_test_set = False
    with open(filename, 'rt') as handle:
        parsed = json.load(handle)

    for _, ajson_list in parsed['ants'].items():
        # print("ajson_map: {}".format(ajson_map))
        for ajson in ajson_list:
            result.append(EbProvisionAnnotation(ajson))

    is_test_set = parsed.get('isTestSet', False)

    # if provision_name is specified, only return that specific provision
    if provision_name:
        return [provision_se for provision_se in result
                if provision_se.label == provision_name], is_test_set

    return result, is_test_set
