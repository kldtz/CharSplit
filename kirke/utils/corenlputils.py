import re

from pycorenlp import StanfordCoreNLP

from kirke.utils.corenlpsent import EbSentence, eb_tokens_to_st

NLP_SERVER = StanfordCoreNLP('http://localhost:9500')


# http://stanfordnlp.github.io/CoreNLP/ner.html#sutime        
# Using default NER models, 
# By default, the models used will be the 3class, 7class,
# and MISCclass models, in that order.
# We should use 3class first because of the reason stated in
# http://stackoverflow.com/questions/33905412/why-does-stanford-corenlp-ner-annotator-load-3-models-by-default


# WARNING: all the spaces before the first non-space character will be removed in the output.
# In other words, the offsets will be incorrect if there are prefix spaces in the text.
# We will fix those issues in the later modules, not here.
def annotate(text_as_string):
    # "ssplit.isOneSentence": "true"
    # 'ner.model': 'edu/stanford/nlp/models/ner/english.muc.7class.distsim.crf.ser.gz',
    output = NLP_SERVER.annotate(text_as_string, properties={
        'annotators': 'tokenize,ssplit,pos,lemma,ner',
        'outputFormat': 'json',
        'ssplit.newlineIsSentenceBreak': 'two'
    })
    return output


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


def is_page_number_st(xst):
    return PAGE_NUMBER_PAT.match(xst)


def is_sent_page_number(ebsent_list, sent_idx):
    num_sent = len(ebsent_list)
    if sent_idx < num_sent:
        # get first character of the sentence
        tokens = ebsent_list[sent_idx].get_tokens()
        # n
        # -n-
        # page n
        # page -n-
        if len(tokens) < 5:
            page_num_st = eb_tokens_to_st(tokens)
            if is_page_number_st(page_num_st):
                return True
    return False


def _pre_merge_broken_ebsents(ebsent_list, atext):
    result = []

    sent_idx = 0
    num_sent = len(ebsent_list)
    while sent_idx < num_sent:
        ebsent = ebsent_list[sent_idx]
        # print("ebsent #{}: {}".format(sent_idx, ebsent))
        sent_st = ebsent.get_text()
        last_char = sent_st[-1]
        if last_char not in ['.', '!', '?']:
            # if next sent starts with a lowercase letter
            if is_sent_starts_with_lower(ebsent_list, sent_idx+1):
                # add all the tokens
                ebsent.extend_tokens(ebsent_list[sent_idx+1].get_tokens(),
                                     atext)
                sent_idx += 1
            elif (is_sent_page_number(ebsent_list, sent_idx+1) and
                  is_sent_starts_with_lower(ebsent_list, sent_idx+1)):
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
def corenlp_json_to_ebsent_list(file_id, ajson, atext):
    result = []

    # num_prefix_space = _strutils.get_num_prefix_space(atext)
    num_prefix_space = align_first_word_offset(ajson['sentences'], atext)

    for json_sent in ajson['sentences']:
        ebsent = EbSentence(file_id, json_sent, atext, num_prefix_space)
        result.append(ebsent)

    result = _pre_merge_broken_ebsents(result, atext)
    return result
