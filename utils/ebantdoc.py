from collections import namedtuple
from enum import Enum
import json


class EbEntityType(Enum):
    PERSON = 1
    ORGANIZATION = 2
    LOCATION = 3
    DATE = 4

eb_provision_st_list = ['party', 'date', 'title', 'change_control', 'assign',
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
eb_provision_st_set = set(eb_provision_st_list)


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

    
# ProvisionAnnotationTuple = namedtuple('ProvisionAnnotation', ['label', 'start', 'end'])
class ProvisionAnnotation:
    __slots__ = ['label', 'start', 'end']
    
    def __init__(self, label, start, end):
        self.label = label
        self.start = start
        self.end = end

    def __repr__(self):
        return "ProvisionAnnotation('{}', {}, {})".format(self.label, self.start, self.end)    
#    def to_tuple(self):
#        return (self.lable, self.start, self.end)

    
class EbProvisionAnnotation:
    __slots__ = ['confidence', 'correctness', 'start', 'end',
                 'type', 'text', 'id', 'custom_text']
    def __init__(self, ajson):
        self.confidence = ajson['confidence']
        self.correctness = ajson.get('correctness')
        self.start = ajson.get('start')
        self.end = ajson.get('end')
        self.type = ajson.get('type')
        self.text = ajson.get('text')
        self.id = ajson.get('id')    # string
        self.custom_text = ajson.get('customText')  # boolean

    def to_dict(self):
        return {"confidence" : self.confidence,
                "correctness" : self.correctness,
                "customText" : self.custom_text,
                "start" : self.start,
                "end" : self.end,
                "id" : self.id,
                "text" : self.text,
                "type" : self.type}

    def __str__(self):
        return str(self.to_dict())

    def to_tuple(self):
        return ProvisionAnnotation(self.type, self.start, self.end)


def load_provision_annotations(filename, provision_name=None):
    result = []
    with open(filename, "rt") as handle:
        parsed = json.load(handle)
        for ajson in parsed:
            eb_ant = EbProvisionAnnotation(ajson)

            result.append(eb_ant.to_tuple())

    # if provision_name is specified, only return that specific provision            
    if provision_name:  
        return [provision_se for provision_se in result if provision_se.label == provision_name]
        
    return result


class EbAnnotatedDoc:

    def __init__(self, file_name, eb_sents, prov_ant_list, attrvec_list, text):
        self.file_id = file_name
        self.ebsents = eb_sents
        self.prov_annotation_list = prov_ant_list
        self.provision_set = [prov_ant.label for prov_ant in prov_ant_list]
        self.attrvec_list = attrvec_list
        self.text = text

    def get_file_id(self):
        return self.file_id
    
    def get_ebsent_list(self):
        return self.ebsents

    def set_provision_annotations(self, ant_list):
        self.provision_annnotation_list = ant_list

    def get_provision_annotations(self):
        return self.provision_annotation_list

    def get_provision_set(self):
        return self.provision_set

    def get_attrvec_list(self):
        return self.attrvec_list

    def get_text(self):
        return self.text

