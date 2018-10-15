import json
import logging
import re
from typing import Any, List, Optional

from stanfordcorenlp import StanfordCoreNLP

from kirke.utils.corenlpsent import EbSentence
from kirke.utils.strutils import corenlp_normalize_text
from kirke.utils.textoffset import TextCpointCunitMapper

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# disable logging from 'requests'
logging.getLogger("requests").setLevel(logging.WARNING)


# loading it here causes nosetests to be stuck
# NLP_SERVER = StanfordCoreNLP('http://localhost', port=9500)
NLP_SERVER = None

def init_corenlp_server():
    # pylint: disable=global-statement
    global NLP_SERVER
    if NLP_SERVER is None:
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
def annotate(text_as_string: str, doc_lang: Optional[str]) -> Any:
    # to get rid of mypy error: NLP_SERVER has no attribute 'annotate'
    # init_corenlp_server()
    # pylint: disable=global-statement
    global NLP_SERVER
    if NLP_SERVER is None:
        NLP_SERVER = StanfordCoreNLP('http://localhost', port=9500)

    if not doc_lang: # no language detected, text is probably too short or empty
        return {'sentences': []}

    no_ctrl_chars_text = corenlp_normalize_text(text_as_string)

    # "ssplit.isOneSentence": "true"
    # 'ner.model': 'edu/stanford/nlp/models/ner/english.muc.7class.distsim.crf.ser.gz',
    doc_lang = doc_lang[:2]
    supported_langs = ["fr", "es", "zh"] #ar and de also supported, can add later
    if doc_lang in supported_langs:
        logger.debug("corenlp running on %s, len=%d", doc_lang, len(no_ctrl_chars_text))
        output = NLP_SERVER.annotate(no_ctrl_chars_text,
                                     properties={'annotators': 'tokenize,ssplit,ner',
                                                 'outputFormat': 'json',
                                                 'enforceRequirements': 'false',
                                                 'ssplit.newlineIsSentenceBreak': 'two',
                                                 'useKnownLCWords': 'false',
                                                 'pipelineLanguage': doc_lang})
    elif doc_lang == "pt":
        logger.debug("corenlp running on %s, len=%d", doc_lang, len(no_ctrl_chars_text))
        output = NLP_SERVER.annotate(no_ctrl_chars_text,
                                     properties={'annotators': 'tokenize,ssplit,ner',
                                                 'outputFormat': 'json',
                                                 'enforceRequirements': 'false',
                                                 'ssplit.newlineIsSentenceBreak': 'two',
                                                 'useKnownLCWords': 'false',
                                                 'ner.model':'portuguese-ner.ser.gz'})
    else:
        logger.debug("corenlp running on en, len=%d", len(no_ctrl_chars_text))
        output = NLP_SERVER.annotate(no_ctrl_chars_text,
                                     properties={'annotators': 'tokenize,ssplit,pos,ner',
                                                 'outputFormat': 'json',
                                                 'ssplit.newlineIsSentenceBreak': 'two',
                                                 'useKnownLCWords': 'false',
                                                 'pipelineLanguage': 'en'})
    return json.loads(output)


def check_pipeline_lang(doc_lang: str, filename: str) -> str:
    with open(filename, 'r') as doc:
        doc_text = doc.read()
        return annotate(doc_text, doc_lang)

def annotate_for_enhanced_ner(text_as_string: str, doc_lang: str = 'en'):
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


# CoreNLP removes spaces at the beginning of a doc using trim(),
# so the offsets for docs with prefix spaces have wrong offsets.
def align_first_word_offset(json_sent_list, atext):
    if not json_sent_list:
        return 0
    # ' &nbsp a' corenlp matches to start=2, end=3
    # we want start = 3, end =4
    # verified nbsp.isspace() == True, so cannot use that.
    adjust = 0
    for i, unused_char in enumerate(atext):
        if ord(atext[i]) <= 32:  # ord(SPACE) = 32
            adjust += 1
        else:
            break
    return adjust


# ajson is result from corenlp
# returns a list of EbSentence
def corenlp_json_to_ebsent_list(file_id, ajson, atext) -> List[EbSentence]:
    result = []  # type: List[EbSentence]

    if isinstance(ajson, str):
        logger.error('failed to corenlp file_id_xxx: [%s]', file_id)
        logger.error('ajson= %s...', str(ajson)[:200])

    # num_prefix_space = _strutils.get_num_prefix_space(atext)
    num_prefix_space = align_first_word_offset(ajson['sentences'], atext)

    for json_sent in ajson['sentences']:
        ebsent = EbSentence(file_id, json_sent, atext, num_prefix_space)
        result.append(ebsent)

    result = filter_out_empty_lines(result, atext)
    return result
