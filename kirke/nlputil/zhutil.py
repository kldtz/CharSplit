import json
import pprint
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


def fix_zh_corenlp_sent_seg(json_st: str) -> str:
    ajson = json.loads(json_st)
    sents_attr = ajson['sentences']

    sent_toks_list = []  # type: List[List[Dict]]
    for sentx in sents_attr:
        tokens = sentx['tokens']
        sent_toks = []  # type: List[Dict]
        for token_dict in tokens:
            sent_toks.append(token_dict)
            if token_dict['word'] == '\u3002' or \
               token_dict['word'] == '。':
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
