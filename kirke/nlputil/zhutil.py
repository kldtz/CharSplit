import json
import pprint
import re
# pylint: disable=unused-import
from typing import Dict, List

def print_sent(tok_list):
    print('tok send: ', end='')
    for tok in tok_list:
        print(tok['word'], end=' ')
    print()


def print_sent_toks_verbatim(json_st: str) -> None:
    ajson = json.loads(json_st)
    # print('ajson:')
    # print(ajson)
    sents_attr = ajson['sentences']

    for sent_i, sentx in enumerate(sents_attr):
        print('\nsent #{}'.format(sent_i))
        pprint.pprint(sentx)

# Googled on 'chinese sentence segmentation', but
# pretty much no good results.  That implies that the topic
# is too trivial to write a paper on.

# https://en.wikipedia.org/wiki/Sentence_boundary_disambiguation
# Languages like Japanese and Chinese have unambiguous sentence-ending markers.

# https://stanfordnlp.github.io/CoreNLP/ssplit.html
# We specifically remove '.' character in this regex because
# sometime a float is spelled with '.' separated by spaces.
ZH_EOS_PAT = re.compile(r'^([。]|[!?！？]+)$')

def fix_zh_corenlp_sent_seg(json_st: str) -> str:
    ajson = json.loads(json_st)
    sents_attr = ajson['sentences']

    sent_toks_list = []  # type: List[List[Dict]]
    for sentx in sents_attr:
        tokens = sentx['tokens']
        num_tokens = len(tokens)
        sent_toks = []  # type: List[Dict]
        for token_i, token_dict in enumerate(tokens):
            sent_toks.append(token_dict)
            if token_dict['word'] == '\u3002' or \
               bool(ZH_EOS_PAT.search(token_dict['word'])):
                next_word = ''
                if token_i + 1 < num_tokens:
                    next_word = tokens[token_i + 1]['word']
                # '。）' is not end of a sentence in Japanese
                if next_word == ')' or \
                   next_word == '）':
                    pass
                else:
                    sent_toks_list.append(sent_toks)
                    sent_toks = []
        # for the last sent in sentx
        if sent_toks:
            sent_toks_list.append(sent_toks)

    out_sent_list = []  # type: List[Dict]
    for sentx_i, sent_toks in enumerate(sent_toks_list):
        out_sent_list.append({'index': sentx_i,
                              'tokens': sent_toks})
    out_dict = {'sentences': out_sent_list}
    return json.dumps(out_dict)
