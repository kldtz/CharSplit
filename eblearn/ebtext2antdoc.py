import os
import json
from pathlib import Path
import logging
from sklearn.externals import joblib

from annotator.corenlp_sent_annotator import CoreNlpSentenceAnnotator
from eblearn import sent2ebattrvec

from utils import strutils, corenlputils, osutils, ebantdoc, mathutils

from utils.ebantdoc import EbEntityType

import time

DEFAULT_IS_CACHE_ENABLED = True

incorrect_corenlp_entities = {'Service', 'Confidential Information', 'Confidential Information Agreement',
                              'Employment Agreement', 'Employment Agreement',
                              'Notice of Termination', 'Territory', 'Base Salary',
                              'Company for Cause', 'Term of Employment', 'Treasury Regulations',
                              'Treas. Reg', 'Confidential Treatment Requested',
                              'Termination of Employment of Executive', 'Field of Use',
                              'Change in Control', 'Research Project', 'Peer Executives',
                              'Executive of Executive', 'Research Plans',
                              'Event of Force Majeure'}

incorrect_person_entities = {'Employee', 'General Counsel', 'Developer', 'Executive', 'Former Employee',
                             'Chief Financial Officer'}

incorrect_org_entities = {'Licensee', 'Landlord', 'Plaintiff', 'Named Users'}

incorrect_date_entities = {'Employment Period', 'Severance Period', 'Performance Period', 'Warranty Period',
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

incorrect_loc_entities = {'State of New York'} # ??

incorrect_domain_entities = {'Annual Performance Share Award', 'Restated Employment Agreement',
                             'Employment Term', 'Equity Documents', 'Disability of Executive',
                             'Security Deposit', 'Termination Without Cause',
                             'Notice of Intent', 'Annual Base Salary', 'Change of Control Transaction',
                             'Fair Market Value', 'Incentive Compensation', 'Reimbursement Amount',
                             'Continueation Coverage Reimbursement Payments', 'Exhibit A.', 'Exhibit B.',
                             'Exhibit C.', 'Excess Rent', 'Release of Claims', 'Reason of Death',
                             'Notice', 'Intellectual Property Rights', 'Financial Interest',
                             'Change of Control', 'Change of Control and Executive',
                             'Limited Warranty'}

def _fix_incorrect_tokens(st, orig_label, token_list, entity_st_set, new_ner):
    if st in entity_st_set:
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
    st = ' '.join([token.word for token in token_list])

    label = _fix_incorrect_tokens(st, label, token_list, incorrect_corenlp_entities, 'O')
    if label is None:
        return None
    label = _fix_incorrect_tokens(st, label, token_list, incorrect_domain_entities, 'DOMAIN-X')
    if label == 'DOMAIN-X':
        return None
    
    label = _fix_incorrect_tokens(st, label, token_list, incorrect_person_entities,
                                  EbEntityType.PERSON.name)
    label = _fix_incorrect_tokens(st, label, token_list, incorrect_org_entities,
                                  EbEntityType.ORGANIZATION.name)
    label = _fix_incorrect_tokens(st, label, token_list, incorrect_date_entities,
                                  EbEntityType.DATE.name)
    label = _fix_incorrect_tokens(st, label, token_list, incorrect_loc_entities,
                                  EbEntityType.LOCATION.name)

    return ebantdoc.EbEntity(start, end, label, st)

_wanted_entity_names = {EbEntityType.PERSON.name,
                        EbEntityType.ORGANIZATION.name,
                        EbEntityType.LOCATION.name,
                        EbEntityType.DATE.name}

_loc_or_org = {EbEntityType.ORGANIZATION.name, EbEntityType.LOCATION.name}


def _extract_entities(tokens, wanted_ner_names):
    entity_list = []
    prev_entity_tokens = []
    prev_ner = None
    for token in tokens:
        curr_ner = token.ner
        if curr_ner in _wanted_entity_names:
            if curr_ner != prev_ner and prev_entity_tokens:
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


def populate_ebsent_entities(ebsent):
    tokens = ebsent.get_tokens()
    entity_list = _extract_entities(tokens, _wanted_entity_names)
    if entity_list:
        ebsent.set_entities(entity_list)


def fix_ner_tags(ebsent):
    tokens = ebsent.get_tokens()
    for token in tokens:
        if token.word == 'Lessee' and token.ner in _loc_or_org:
            token.ner = 'O'

            
# def filter_feature_start_end(feat_json_list, feature_name_set):
#    result_list = []
#    for feat_json in feat_json_list:
#        if feat_json['type'] in feature_name_set:
#            result_list.append((feat_json['type'], feat_json['start'], feat_json['end']))
#    return result_list


def get_labels_if_start_end_overlap(sent_start, sent_end, ant_start_end_list):
    result_label_list = []
    for ant in ant_start_end_list:
        if mathutils.start_end_overlap((sent_start, sent_end), (ant.start, ant.end)):
            result_label_list.append(ant.label)
    return result_label_list



sent_annotator = CoreNlpSentenceAnnotator()

# output_json is not None for debugging purpose
def parse_to_eb_antdoc(atext, txt_file_name, work_dir=None):
    # load/save the corenlp file if output_dir is specified
    is_cache_enabled = DEFAULT_IS_CACHE_ENABLED
    if work_dir is None:
        is_cache_enabled = False
    #else:  # if work_dir is not None:
    #    # print("txt_file_name= [{}]".format(txt_file_name))
    if txt_file_name:
        txt_basename = os.path.basename(txt_file_name)
        # if cache version exists, load that and return
        if is_cache_enabled:
            eb_antdoc_fn = work_dir + "/" + txt_basename.replace('.txt', '.ebantdoc.pkl')
            if os.path.exists(eb_antdoc_fn):
                eb_antdoc = joblib.load(eb_antdoc_fn)
                logging.info("loading cached version: {}".format(eb_antdoc_fn))
                return eb_antdoc

            json_fn = work_dir + "/" + txt_basename.replace('.txt', '.corenlp.json')
            if os.path.exists(json_fn):
                corenlp_json = json.loads(strutils.loads(json_fn))
            else:
                corenlp_json = sent_annotator.annotate(atext)
                strutils.dumps(json.dumps(corenlp_json), json_fn)
        else:
            corenlp_json = sent_annotator.annotate(atext)                
    else:
        corenlp_json = sent_annotator.annotate(atext)

    prov_ant_fn = txt_file_name.replace('.txt', '.ant')
    prov_ant_file = Path(prov_ant_fn)
    if prov_ant_file.is_file():        
        prov_annotation_list = ebantdoc.load_provision_annotations(prov_ant_fn)
    else:
        prov_annotation_list = []

    ebsent_list = corenlputils.corenlp_json_to_ebsent_list(txt_file_name, corenlp_json, atext)
    # print('number of sentences: {}'.format(len(ebsent_list)))

    # fix any domain specific entity extraction, such as 'Lessee' as a location
    # this is a in-place replacement
    for ebsent in ebsent_list:
        fix_ner_tags(ebsent)
        populate_ebsent_entities(ebsent)
        # for i, entity in enumerate(ebsent.get_entities()):
        #    print("{}, entity #{}: {}".format(txt_file_name, i, entity.to_tuple()))

        overlap_provisions = []
        if prov_annotation_list:
            overlap_provisions = get_labels_if_start_end_overlap(ebsent.get_start(),
                                                                 ebsent.get_end(),
                                                                 prov_annotation_list)
        ebsent.set_labels(overlap_provisions)

    attrvec_list = []
    num_sent = len(ebsent_list)
    # we need prev and next sentences because such information are used in the
    # feature extraction
    prev_ebsent = None
    next_ebsent = None    
    for sent_idx, ebsent in enumerate(ebsent_list):
        sent_st = ebsent.get_text()
        if sent_idx != num_sent-1:
            next_ebsent = ebsent_list[sent_idx + 1]
        else:
            next_ebsent = None

        fv = sent2ebattrvec.sent2ebattrvec(txt_file_name, ebsent, sent_idx + 1,
                                           prev_ebsent, next_ebsent, atext)

        # print('{}\t{}'.format(sent_idx+1, sent_st))
        # print("fv= " + str(fv))
        # print("num none = " + str(count_none(fv.to_list())))
        attrvec_list.append(fv.to_list())
        prev_ebsent = ebsent

    eb_antdoc = ebantdoc.EbAnnotatedDoc(txt_file_name, ebsent_list,
                                        prov_annotation_list, attrvec_list, atext)

    if txt_file_name and is_cache_enabled:
        txt_basename = os.path.basename(txt_file_name)
        # if cache version exists, load that and return
        eb_antdoc_fn = work_dir + "/" + txt_basename.replace('.txt', '.ebantdoc.pkl')
        joblib.dump(eb_antdoc, eb_antdoc_fn)
        logging.info("save in cached: {}".format(eb_antdoc_fn))            
    return eb_antdoc

def doc_to_ebantdoc(txt_file_name, work_dir):
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.info("mkdir '{}'".format(work_dir))
        osutils.mkpath(work_dir)

    start_time = time.time()
    doc_text = strutils.loads(txt_file_name)
    eb_antdoc = parse_to_eb_antdoc(doc_text, txt_file_name, work_dir=work_dir)
    now_time = time.time()
    logging.info('feature extraction: "{}, took {:.2f} seconds"'.format(txt_file_name, now_time - start_time))

    return eb_antdoc

def doclist_to_ebantdoc_list(doclist_file, work_dir):
    logging.info('doclist_to_ebantdoc_list({}, {})'.format(doclist_file, work_dir))
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.info("mkdir '{}'".format(work_dir))
        osutils.mkpath(work_dir)
    eb_antdoc_list = []
    with open(doclist_file, 'rt') as fin:
        for txt_file_name in fin:
            txt_file_name = txt_file_name.strip()
            eb_antdoc = doc_to_ebantdoc(txt_file_name, work_dir)
            eb_antdoc_list.append(eb_antdoc)
    logging.info('Finished run_feature_extraction()')

    return eb_antdoc_list

