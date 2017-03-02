import re

from utils.corenlpsent import EbSentence, eb_tokens_to_st


def is_sent_starts_with_lower(ebsent_list, sent_idx):
    num_sent = len(ebsent_list)
    if sent_idx < num_sent:
        # get first character of the sentence
        tokens = ebsent_list[sent_idx].get_tokens()
        if tokens:
            # first character of the word must be lower letter
            return tokens[0].word[0].islower()
    return False


page_number_pat = re.compile(r'^(\d+|(page\s+)?\-?\s*\d+\s*\-?)$', re.IGNORECASE)

def is_page_number_st(st):
    return page_number_pat.match(st)


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

