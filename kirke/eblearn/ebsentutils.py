import re
from kirke.utils import stopwordutils, mathutils, ebantdoc, entityutils

# from kirke.utils import corenlputils, ebantdoc, mathutils, strutils, osutils, entityutils, txtreader

_WANTED_ENTITY_NAMES = {ebantdoc.EbEntityType.PERSON.name,
                        ebantdoc.EbEntityType.ORGANIZATION.name,
                        ebantdoc.EbEntityType.LOCATION.name,
                        ebantdoc.EbEntityType.DATE.name,
                        ebantdoc.EbEntityType.DEFINE_TERM.name}

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

_LOC_OR_ORG = {ebantdoc.EbEntityType.ORGANIZATION.name,
               ebantdoc.EbEntityType.LOCATION.name}

_PERSON_DFTERM_SET = set([ebantdoc.EbEntityType.DEFINE_TERM.name,
                          ebantdoc.EbEntityType.PERSON.name])
_ORG_DFTERM_SET = set([ebantdoc.EbEntityType.DEFINE_TERM.name,
                       ebantdoc.EbEntityType.ORGANIZATION.name])


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
        entity_ner_set.remove(ebantdoc.EbEntityType.DEFINE_TERM.name)
        label = entity_ner_set.pop()
    elif len(entity_ner_set) == 1 and entity_ner_set.pop() == ebantdoc.EbEntityType.DEFINE_TERM.name:
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
    for token in tokens:
        if token.word == 'CORPORATE':
            token.pos = 'NNP'

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
            (i + 1) < max_token_ptr and tokens[i+1].word == 'and' and
            tokens[i+1].ner == ebantdoc.EbEntityType.ORGANIZATION.name):
            tokens[i+1].ner = 'O'

    pat_list = entityutils.extract_define_party(raw_sent_text, start_offset=start_offset)
    if pat_list:
        for i, token in enumerate(tokens):
            for pat in pat_list:
                if mathutils.start_end_overlap((pat[1], pat[2]), (token.start, token.end)):
                    token.ner = ebantdoc.EbEntityType.DEFINE_TERM.name

    #print()
    #for i, token in enumerate(tokens, 1):
    #    print('x234 {}\t{}'.format(i, token))


def update_ebsents_with_sechead(ebsent_list, paras_with_attrs):
    para_i, len_paras = 0, len(paras_with_attrs)
    ebsent_i, len_ebsents = 0, len(ebsent_list)
    ebsent = ebsent_list[ebsent_i]
    ebsent_start, ebsent_end = ebsent.start, ebsent.end

    while para_i < len_paras and ebsent_i < len_ebsents:
        (para_from_start, para_from_end), (para_to_start, para_to_end), line, secheadx = paras_with_attrs[para_i]
        if para_to_start == para_to_end:  # empty line, move on
            para_i += 1
            continue
        if secheadx:
            # print("secheadx: {}".format(secheadx[0]))
            sechead_type, sh_prefix_num, sh_header, sh_idx = secheadx[0]
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


