import json
import logging
import os
import time
from pathlib import Path

from sklearn.externals import joblib

from kirke.eblearn import sent2ebattrvec

# TODO, remove.  this is mainly for printing out sentence text for debug
# at the end of parsexxx
from kirke.eblearn import ebattrvec

from kirke.utils import corenlputils, ebantdoc, mathutils, strutils, osutils, entityutils


DEFAULT_IS_CACHE_ENABLED = True

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
    'Limited Warranty'}

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

    label = _fix_incorrect_tokens(xst, label, token_list, INCORRECT_PERSON_ENTITIES,
                                  ebantdoc.EbEntityType.PERSON.name)
    label = _fix_incorrect_tokens(xst, label, token_list, INCORRECT_ORG_ENTITIES,
                                  ebantdoc.EbEntityType.ORGANIZATION.name)
    label = _fix_incorrect_tokens(xst, label, token_list, INCORRECT_DATE_ENTITIES,
                                  ebantdoc.EbEntityType.DATE.name)
    label = _fix_incorrect_tokens(xst, label, token_list, INCORRECT_LOC_ENTITIES,
                                  ebantdoc.EbEntityType.LOCATION.name)

    return ebantdoc.EbEntity(start, end, label, xst)

_WANTED_ENTITY_NAMES = {ebantdoc.EbEntityType.PERSON.name,
                        ebantdoc.EbEntityType.ORGANIZATION.name,
                        ebantdoc.EbEntityType.LOCATION.name,
                        ebantdoc.EbEntityType.DATE.name}

_LOC_OR_ORG = {ebantdoc.EbEntityType.ORGANIZATION.name,
               ebantdoc.EbEntityType.LOCATION.name}


def _extract_entities(tokens, wanted_ner_names):
    entity_list = []
    prev_entity_tokens = []
    prev_ner = None

    for token in tokens:
        curr_ner = token.ner
        if curr_ner in wanted_ner_names:
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

NAME_POS_SET = set(['NNS', 'CD', 'NNP', 'NN'])

# this is destructive/in-place
def _extract_entities_v2(tokens, raw_sent_text, start_offset=0):
    ptr = -1
    max_token_ptr = len(tokens)
    # fix incorrect pos
    for token in tokens:
        if token.word == 'CORPORATE':
            token.pos = 'NNP'

    for i, token in enumerate(tokens):
        # print('{}\t{}'.format(i, token))
        if token.word[0].isupper() and token.word.lower() in set(['llc.', 'llc', 'inc.', 'inc', 'l.p.', 'n.a.',
                                                                  'corp', 'corporation', 'corp.', 'ltd.', 'ltd',
                                                                  'co.', 'co', 'l.l.p.', 'lp', 's.a.', 'sa',
                                                                  'n.v.', 'plc', 'plc.', 'l.l.c.']):
            # reset all previous tokens to ORG
            # print("I am in here")
            ptr = i
            while ptr >= 0:
                if ptr == i - 1 and tokens[ptr].word == ',':
                    tokens[ptr].ner = ebantdoc.EbEntityType.ORGANIZATION.name
                    ptr -= 1
                elif tokens[ptr].pos in NAME_POS_SET:
                    # print("tokens[{}].pos = {}, {}".format(ptr, tokens[ptr].pos, tokens[ptr]))
                    tokens[ptr].ner = ebantdoc.EbEntityType.ORGANIZATION.name
                    ptr -= 1
                else:
                    break
        # separate "the Company and xxx"
        if (token.word in 'Company' and token.ner == ebantdoc.EbEntityType.ORGANIZATION.name and
            (i + 1) < max_token_ptr and tokens[i+1].word == 'and' and tokens[i+1].ner == ebantdoc.EbEntityType.ORGANIZATION.name):
            tokens[i+1].ner = 'O'

    pat_list = entityutils.extract_define_party(raw_sent_text, start_offset=start_offset)
    if pat_list:
        for i, token in enumerate(tokens):
            for pat in pat_list:
                if mathutils.start_end_overlap((pat[1], pat[2]), (token.start, token.end)):
                    if token.word in ['-LRB-', '-RRB-']:
                        token.ner = 'O'
                    else:  # if token.word in [pat[0],]:  # there are some offsets issue, so need to do this
                        token.ner = ebantdoc.EbEntityType.ORGANIZATION.name

    #print()
    #for i, token in enumerate(tokens, 1):
    #    print('x2 {}\t{}'.format(i, token))


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

def save_ebantdoc_sents(eb_antdoc, txt_file_name):
    txt_basename = os.path.basename(txt_file_name)
    doc_sents_dir = 'dir-doc-sents'
    doc_sents_fn = doc_sents_dir + "/" + txt_basename.replace('.txt', '.sent')
    doc_text = eb_antdoc.text
    ts_col = 'TRAIN'
    if eb_antdoc.is_test_set:
        ts_col = 'TEST'
    # print("doc_sents_fn = {}".format(doc_sents_fn))
    with open(doc_sents_fn, 'wt') as fout3:
        for i, attrvec in enumerate(eb_antdoc.attrvec_list, 1):
            # print("attrvec = {}".format(attrvec))
            tmp_start = attrvec[ebattrvec.START_INDEX]
            tmp_end = attrvec[ebattrvec.END_INDEX]
            sent_text = doc_text[tmp_start:tmp_end].replace(r'[\n\t]', ' ')
            labels_st = ""
            if attrvec[ebattrvec.LABELS_INDEX]:
                labels_st = ','.join(sorted(attrvec[ebattrvec.LABELS_INDEX]))
            cols = [str(i), ts_col, labels_st, sent_text]
            print('\t'.join(cols), file=fout3)    


# output_json is not None for debugging purpose
# pylint: disable=R0914
# If is_bespoke mode, the annotation can change across different bespoke runs.
# As a result, never cache .ebantdoc.pkl, but can reuse corenlp.json
def parse_to_eb_antdoc(atext, txt_file_name, work_dir=None, is_bespoke_mode=False):
    # load/save the corenlp file if output_dir is specified
    is_cache_enabled = False if work_dir is None else DEFAULT_IS_CACHE_ENABLED

    # print("txt_file_name= [{}]".format(txt_file_name))
    if txt_file_name:
        txt_basename = os.path.basename(txt_file_name)
        # if cache version exists, load that and return
        if is_cache_enabled:
            eb_antdoc_fn = work_dir + "/" + txt_basename.replace('.txt', '.ebantdoc.pkl')
            if not is_bespoke_mode and os.path.exists(eb_antdoc_fn):
                eb_antdoc = joblib.load(eb_antdoc_fn)
                logging.info("loading from cache: %s", eb_antdoc_fn)

                # TODO, jshaw, remove after debugging
                # save_ebantdoc_sents(eb_antdoc, txt_file_name)
                return eb_antdoc

            json_fn = work_dir + "/" + txt_basename.replace('.txt', '.corenlp.json')
            if os.path.exists(json_fn):
                corenlp_json = json.loads(strutils.loads(json_fn))
                logging.info("loading from cache: %s", json_fn)
            else:
                corenlp_json = corenlputils.annotate_for_enhanced_ner(atext)
                strutils.dumps(json.dumps(corenlp_json), json_fn)
        else:
            corenlp_json = corenlputils.annotate_for_enhanced_ner(atext)
    else:
        corenlp_json = corenlputils.annotate_for_enhanced_ner(atext)

    prov_ant_fn = txt_file_name.replace('.txt', '.ant')
    prov_ant_file = Path(prov_ant_fn)
    prov_ebdata_fn = txt_file_name.replace('.txt', '.ebdata')
    prov_ebdata_file = Path(prov_ebdata_fn)

    prov_annotation_list = []
    is_test = False
    if os.path.exists(prov_ant_fn):
        prov_annotation_list = (ebantdoc.load_provision_annotations(prov_ant_fn)
                                if prov_ant_file.is_file() else [])
    elif os.path.exists(prov_ebdata_fn):
        prov_annotation_list, is_test = (ebantdoc.load_prov_ebdata(prov_ebdata_fn)
                                         if prov_ebdata_file.is_file() else [])


    ebsent_list = corenlputils.corenlp_json_to_ebsent_list(txt_file_name, corenlp_json, atext)
    # print('number of sentences: {}'.format(len(ebsent_list)))

    # fix any domain specific entity extraction, such as 'Lessee' as a location
    # this is a in-place replacement
    # We only handle up to "exhibit_appendix,exhibit_appendix_complete"
    ebsents_without_exhibit = []
    exhibit_appendix_start = -1
    for ebsent in ebsent_list:
        fix_ner_tags(ebsent)
        populate_ebsent_entities(ebsent, atext[ebsent.start:ebsent.end])

        overlap_provisions = (get_labels_if_start_end_overlap(ebsent.get_start(),
                                                              ebsent.get_end(),
                                                              prov_annotation_list)
                              if prov_annotation_list else [])
        # logging.info("overlap_provisions: {}".format(overlap_provisions))
        
        ebsent.set_labels(overlap_provisions)
        if ('exhibit_appendix' in overlap_provisions or
            'exhibit_appendix_complete' in overlap_provisions):
            exhibit_appendix_start = ebsent.get_start()
            # logging.info('exhibit_appendix_start: {}'.format(exhibit_appendix_start))
            break
        ebsents_without_exhibit.append(ebsent)

    # we need to chop provisions after exhibit_appendix_start also
    if exhibit_appendix_start != -1:
        tmp_prov_annotation_list = []
        for prov_annotation in prov_annotation_list:
            if (exhibit_appendix_start <= prov_annotation.start or
                mathutils.start_end_overlap((exhibit_appendix_start, exhibit_appendix_start+1),
                                            (prov_annotation.start, prov_annotation.end))):
                #logging.info("skipping prov '{}' {}, after appendix offset {}".format(prov_annotation.label,
                #                                                                      prov_annotation.start,
                #                                                                      exhibit_appendix_start))
                pass
            else:
                tmp_prov_annotation_list.append(prov_annotation)
        prov_annotation_list = tmp_prov_annotation_list

    # we reset ebsent_list to ebsents_withotu_exhibit
    ebsent_list = ebsents_without_exhibit

    attrvec_list = []
    num_sent = len(ebsent_list)
    # we need prev and next sentences because such information are used in the
    # feature extraction
    prev_ebsent, next_ebsent = None, None
    for sent_idx, ebsent in enumerate(ebsent_list):
        # sent_st = ebsent.get_text()
        if sent_idx != num_sent-1:
            next_ebsent = ebsent_list[sent_idx + 1]
        else:
            next_ebsent = None
        fvec = sent2ebattrvec.sent2ebattrvec(txt_file_name, ebsent, sent_idx + 1,
                                             prev_ebsent, next_ebsent, atext)

        attrvec_list.append(fvec.to_list())
        prev_ebsent = ebsent

    eb_antdoc = ebantdoc.EbAnnotatedDoc(txt_file_name, prov_annotation_list, attrvec_list, atext, is_test)

    if txt_file_name and is_cache_enabled:
        txt_basename = os.path.basename(txt_file_name)
        # if cache version exists, load that and return
        eb_antdoc_fn = work_dir + "/" + txt_basename.replace('.txt', '.ebantdoc.pkl')
        joblib.dump(eb_antdoc, eb_antdoc_fn)
        logging.debug("save in cached: %s", eb_antdoc_fn)

    # TODO, jshaw, remove, this saves the sentence text version
    # if txt_file_name:
    #    save_ebantdoc_sents(eb_antdoc, txt_file_name)
    
    return eb_antdoc


def doc_to_ebantdoc(txt_file_name, work_dir, is_bespoke_mode=False):
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    start_time = time.time()
    doc_text = strutils.loads(txt_file_name)
    eb_antdoc = parse_to_eb_antdoc(doc_text, txt_file_name, work_dir=work_dir, is_bespoke_mode=is_bespoke_mode)
    now_time = time.time()
    logging.debug('feature extraction: %s, took %.2f seconds',
                  txt_file_name, now_time - start_time)

    return eb_antdoc


def doclist_to_ebantdoc_list(doclist_file, work_dir, is_bespoke_mode=False):
    logging.debug('doclist_to_ebantdoc_list(%s, %s)', doclist_file, work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)
    eb_antdoc_list = []
    with open(doclist_file, 'rt') as fin:
        for i, txt_file_name in enumerate(fin, 1):
            txt_file_name = txt_file_name.strip()
            eb_antdoc = doc_to_ebantdoc(txt_file_name, work_dir, is_bespoke_mode)
            # print("get_size(eb_antdoc) = {} bytes".format(osutils.get_size(eb_antdoc)))
            eb_antdoc_list.append(eb_antdoc)
            if i % 10 == 0:
                print("doclist_to_ebantdoc_list #{}".format(i))
    logging.debug('Finished run_feature_extraction()')

    return eb_antdoc_list


#    fn_ebantdoc_map = ebtext2antdoc.fnlist_to_fn_ebantdoc_map(list(txt_file_set), work_dir=work_dir)
def fnlist_to_fn_ebantdoc_map(fn_list, work_dir):
    logging.debug('fnlist_to_fn_ebantdoc_map(len(list)=%d, work_dir=%s)', len(fn_list), work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    fn_ebantdoc_map = {}
    """
    fn_list_extra = ['36074.clean.txt',
                     '60558.clean.txt',
                     '37351.clean.txt',
                     '40792.clean.txt',
                     '44168.clean.txt',
                     '37404.clean.txt',
                     '37005.clean.txt',
                     '38894.clean.txt',
                     '37851.clean.txt',
                     '55899.clean.txt',
                     '41200.clean.txt']
    fn_list_extra2 = [ "dir-data/{}".format(fn) for fn in fn_list_extra]

    print("fn_list[0] = [{}]".format(fn_list[0]))
    print("fn_list_extra2[0] = [{}]".format(fn_list_extra2[0]))    
"""
    
    for i, txt_file_name in enumerate(fn_list, 1):
        eb_antdoc = doc_to_ebantdoc(txt_file_name, work_dir)
        fn_ebantdoc_map[txt_file_name] = eb_antdoc
        if i % 10 == 0:
            print("loaded #{} ebantdoc".format(i))
    logging.debug('Finished run_feature_extraction()')

    return fn_ebantdoc_map


class EbAntdocProvSet:

    def __init__(self, ebantdoc):
        self.file_id = ebantdoc.get_file_id()
        self.provset = ebantdoc.get_provision_set()
        self.is_test_set = ebantdoc.is_test_set

    def get_file_id(self):
        return self.file_id
    
    def get_provision_set(self):
        return self.provset
    

#    fn_ebantdoc_map = ebtext2antdoc.fnlist_to_fn_ebantdoc_map(list(txt_file_set), work_dir=work_dir)
def fnlist_to_fn_ebantdoc_provset_map(fn_list, work_dir):
    logging.debug('fnlist_to_fn_ebantdoc_map(len(list)=%d, work_dir=%s)', len(fn_list), work_dir)
    if work_dir is not None and not os.path.isdir(work_dir):
        logging.debug("mkdir %s", work_dir)
        osutils.mkpath(work_dir)

    fn_ebantdoc_map = {}
    for i, txt_file_name in enumerate(fn_list, 1):
        # if i % 10 == 0:
        logging.info("loaded #{} ebantdoc: {}".format(i, txt_file_name))

        eb_antdoc = doc_to_ebantdoc(txt_file_name, work_dir)
        
        fn_ebantdoc_map[txt_file_name] = EbAntdocProvSet(eb_antdoc)
    logging.debug('Finished run_feature_extraction()')

    return fn_ebantdoc_map