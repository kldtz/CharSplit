import re
import json
import logging

from stanfordcorenlp import StanfordCoreNLP

from kirke.utils.corenlpsent import EbSentence, eb_tokens_to_st

from kirke.utils.strutils import corenlp_normalize_text
from kirke.utils.textoffset import TextCpointCunitMapper


NLP_SERVER = StanfordCoreNLP('http://localhost', port=9500)


# http://stanfordnlp.github.io/CoreNLP/ner.html#sutime
# Using default NER models,
# By default, the models used will be the 3class, 7class,
# and MISCclass models, in that order.
# We should use 3class first because of the reason stated in
# http://stackoverflow.com/questions/33905412/why-does-stanford-corenlp-ner-annotator-load-3-models-by-default


# WARNING: all the spaces before the first non-space character will be removed in the output.
# In other words, the offsets will be incorrect if there are prefix spaces in the text.
# We will fix those issues in the later modules, not here.
def annotate(text_as_string, doc_lang):
    no_ctrl_chars_text = corenlp_normalize_text(text_as_string)
    # "ssplit.isOneSentence": "true"
    # 'ner.model': 'edu/stanford/nlp/models/ner/english.muc.7class.distsim.crf.ser.gz',
    doc_lang = doc_lang[:2]
    supported_langs = ["fr", "es", "zh"] #ar and de also supported, can add later
    if doc_lang in supported_langs:
        logging.info("corenlp running on {}".format(doc_lang))
        output = NLP_SERVER.annotate(no_ctrl_chars_text,
                                   properties={'annotators': 'tokenize,ssplit,ner',
                                               'outputFormat': 'json',
                                               'enforceRequirements': 'false',
                                               'ssplit.newlineIsSentenceBreak': 'two',
				               'pipelineLanguage': doc_lang})
    
    elif doc_lang == "pt":
        logging.info("corenlp running on {}".format(doc_lang))
        output = NLP_SERVER.annotate(no_ctrl_chars_text,
                                   properties={'annotators': 'tokenize,ssplit,ner',
                                               'outputFormat': 'json',
                                               'enforceRequirements': 'false',
                                               'ssplit.newlineIsSentenceBreak': 'two',
                                               'ner.model':'portuguese-ner.ser.gz'})
    else:
        logging.info("corenlp running on en")
        output = NLP_SERVER.annotate(no_ctrl_chars_text,
                                   properties={'annotators': 'tokenize,ssplit,pos,ner',
                                               'outputFormat': 'json',
                                               'ssplit.newlineIsSentenceBreak': 'two',
                                               'pipelineLanguage': 'en'})
    return json.loads(output)


def annotate_for_enhanced_ner(text_as_string, doc_lang="en"):
    acopy_text = transform_corp_in_text(text_as_string)

    cpoint_cunit_mapper = TextCpointCunitMapper(acopy_text)
    out_json = annotate(acopy_text, doc_lang)

    # this is in-place update
    corenlp_offset_cunit_to_cpoint(out_json, cpoint_cunit_mapper)

    return out_json

def corenlp_offset_cunit_to_cpoint(ajson, cpoint_cunit_mapper):
    """This is in-place modification of ajson, translating from
       code unit to code point offsets"""
    for sent_json in ajson['sentences']:
        for token_json in sent_json['tokens']:
            token_json["characterOffsetBegin"], token_json["characterOffsetEnd"] = \
                cpoint_cunit_mapper.to_codepoint_offsets(token_json["characterOffsetBegin"],
                                                         token_json["characterOffsetEnd"])


CORP_EXPR = r"(,\s*|\b)(inc|corp|llc|ltd)\b"
NOSTRIP_SET = set(["ltd"])
CORP_PAT = re.compile(CORP_EXPR, re.IGNORECASE)

def transform_corp_in_text(raw_text):

    def inplace_str_sub(match):
        if match.group(2).lower() in NOSTRIP_SET:
            return match.group(1) + match.group(2).capitalize()
        return match.group(1).replace(',', ' ') + match.group(2).capitalize()

    return CORP_PAT.sub(inplace_str_sub, raw_text)



def is_sent_starts_with_lower(ebsent_list, sent_idx):
    num_sent = len(ebsent_list)
    if sent_idx < num_sent:
        # get first character of the sentence
        tokens = ebsent_list[sent_idx].get_tokens()
        if tokens:
            # first character of the word must be lower letter
            return tokens[0].word[0].islower()
    return False


PAGE_NUMBER_PAT = re.compile(r'^(\d+|(page\s+)?\-?\s*\d+\s*\-?)$', re.IGNORECASE)


def is_sent_page_number(ebsent_list, sent_idx, doc_text):
    num_sent = len(ebsent_list)
    if sent_idx < num_sent:
        # n
        # -n-
        # page n
        # page -n-
        ebsent = ebsent_list[sent_idx]
        sent_txt = doc_text[ebsent.start:ebsent.end]
        return PAGE_NUMBER_PAT.match(sent_txt)
    return False


def filter_out_empty_lines(ebsent_list, atext):
    result = []
    for ebsent in ebsent_list:
        sent_st = atext[ebsent.start:ebsent.end]
        if sent_st.strip():
            result.append(ebsent)
    return result


def _pre_merge_broken_ebsents(ebsent_list, atext):
    result = []

    sent_idx = 0
    num_sent = len(ebsent_list)
    while sent_idx < num_sent:
        ebsent = ebsent_list[sent_idx]
        # print("ebsent #{}: {}".format(sent_idx, ebsent))
        # sent_st = ebsent.get_text()
        sent_st = atext[ebsent.start:ebsent.end]
        # pylint: disable=fixme
        if sent_st:  # TODO: jshaw, a bug, not sure how this is possible
                     # 36973.clean.txt
            last_char = sent_st[-1]
            if last_char not in ['.', '!', '?']:
                # if next sent starts with a lowercase letter
                if is_sent_starts_with_lower(ebsent_list, sent_idx+1):
                    # add all the tokens
                    ebsent.extend_tokens(ebsent_list[sent_idx+1].get_tokens(),
                                         atext)
                    sent_idx += 1
                elif is_sent_page_number(ebsent_list, sent_idx+1, atext) and sent_idx + 2 < num_sent:
                #elif (is_sent_page_number(ebsent_list, sent_idx+1, atext) and
                #      is_sent_starts_with_lower(ebsent_list, sent_idx+1)):
                    # throw away the page number tokens
                    ebsent.extend_tokens(ebsent_list[sent_idx+2].get_tokens(),
                                         atext)
                    sent_idx += 2
            result.append(ebsent)
        sent_idx += 1
    return result


# CoreNLP treats non-breaking space as a character, so things are kind of messed up with offsets.
# We align the first word instead.
def align_first_word_offset(json_sent_list, atext):
    # get first word
    if not json_sent_list:
        return 0
    first_word_json = json_sent_list[0]['tokens'][0]
    first_word = first_word_json['word']
    first_word_start = first_word_json['characterOffsetBegin']
    return atext.find(first_word) - first_word_start


# ajson is result from corenlp
# returns a list of EbSentence
def corenlp_json_to_ebsent_list(file_id, ajson, atext, is_doc_structure=False):
    result = []

    if isinstance(ajson, str):
        logging.error('failed to corenlp file_id_xxx: [{}]'.format(file_id))
        logging.error('ajson= {}...'.format(str(ajson)[:200]))

    # num_prefix_space = _strutils.get_num_prefix_space(atext)
    num_prefix_space = align_first_word_offset(ajson['sentences'], atext)

    for json_sent in ajson['sentences']:
        ebsent = EbSentence(file_id, json_sent, atext, num_prefix_space)
        result.append(ebsent)

    # if is_doc_structure, merge has already being done
    if not is_doc_structure:
        result = _pre_merge_broken_ebsents(result, atext)
    else:
        result = filter_out_empty_lines(result, atext)
    return result

"""
# ajson is result from corenlp
# paras_with_attrs has the the corenlp offsets
# returns a list of EbSentence
def corenlp_json_to_ebsent_list_v2(file_id, ajson, atext, paras_with_attrs):
    result = []

    if isinstance(ajson, str):
        logging.error('failed to corenlp file_id_xxx: [{}]'.format(file_id))
        logging.error('ajson= {}...'.format(str(ajson)[:200]))

    # num_prefix_space = _strutils.get_num_prefix_space(atext)
    num_prefix_space = align_first_word_offset(ajson['sentences'], atext)

    para_i = 0
    (stage1_start, stage1_end), (para_start, para_end), para_line, attr_list = paras_with_attrs[para_i]
    for json_sent in ajson['sentences']:
        ebsent = EbSentence(file_id, json_sent, atext, num_prefix_space)
        stage2_start = ebsent.start - num_prefix_space
        stage2_end = ebsent.end - num_prefix_space

        # xxx
        xxxx
        when there is a page number deletion, how do you keep of the offsets correct, especially
        the end of the sentence is in the middle of the para?
        The offsets in token is of course even more messed up.

        if mathutils.is_overlap((para_start, para_end), (stage1_start, stage1_eng)):
            ebsent.set_sechead(secheadutils.attrs2sechead)

        result.append(ebsent)

    # result = _pre_merge_broken_ebsents(result, atext)
    return result
"""
