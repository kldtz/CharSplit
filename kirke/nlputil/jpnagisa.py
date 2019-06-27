import json
# pylint: disable=unused-import
from typing import Dict, List, Tuple

# kytea is installed, not nagisa
# pylint: disable=import-error
import nagisa

JP_POS_MAP = {
    '名詞' : 'N', # Noun
    '代名詞' : 'PRP', # Pronoun
    '連体詞' : 'DT', # Adjectival determiner
    '動詞' : 'V', # Verb
    '形容詞' : 'ADJ', # Adjective
    '形状詞' : 'ADJV', # Adjectival verb
    '副詞' : 'ADV', # Adverb
    '助詞' : 'PRT', # Particle
    '助動詞' : 'AUXV', # Auxiliary verb
    '補助記号' : '.', # Punctuation
    '記号' : 'SYM', # Symbol
    '接尾辞' : 'SUF', # Suffix
    '接頭辞' : 'PRE', # Prefix
    '語尾' : 'TAIL', # Word tail (conjugation)
    '接続詞' : 'CC', # Conjunction
    'URL' : 'URL', # URL
    '英単語' : 'ENG', # English word
    '言いよどみ' : 'FIL', # Filler
    'web誤脱' : 'MSP', # Misspelling
    '感動詞' : 'INT', # Interjection
    '新規未知語' : 'UNK', # Unclassified unknown word
    '空白' : 'SPC',  # Space
    # '' : '',
}


class NagisaWordSegmenter:

    def __init__(self) -> None:
        pass

    # pylint: disable=no-self-use
    def to_word_pos_list(self, text: str) -> List[Tuple[str, str]]:
        text = text.replace('　', ' ')
        nagisa_tokens = nagisa.tagging(text)
        words = []  # type: List[Tuple[str, str]]
        for word, pos in zip(nagisa_tokens.words, nagisa_tokens.postags):
            if word.isspace():
                continue
            word = word.replace('　', ' ')
            pos = JP_POS_MAP[pos]
            words.append((word, pos))
        return words

    # pylint: disable=no-self-use
    def to_sent_word_pos_list(self, text: str) \
        -> List[List[Tuple[str, str]]]:
        text = text.replace('　', ' ')
        nagisa_tokens = nagisa.tagging(text)
        sent_list = []  # type: List[List[Tuple[str, str]]]
        words = []  # type: List[Tuple[str, str]]
        for word, pos in zip(nagisa_tokens.words, nagisa_tokens.postags):
            if word.isspace():
                continue
            word = word.replace('　', ' ')
            pos = JP_POS_MAP[pos]
            words.append((word, pos))
            if word == '。':
                sent_list.append(words)
                words = []
        # add last sentence
        if words:
            sent_list.append(words)
        return sent_list

    # pylint: disable=no-self-use
    def to_sent_list(self, text: str) \
        -> List[List[Tuple[str, str, int, int]]]:
        text = text.replace('　', ' ')
        nagisa_tokens = nagisa.tagging(text)
        sent_list = []  # type: List[List[Tuple[str, str, int, int]]]
        words = []  # type: List[Tuple[str, str, int, int]]
        start, end = 0, 0
        for word, pos in zip(nagisa_tokens.words, nagisa_tokens.postags):
            if word.isspace():
                continue
            word = word.replace('　', ' ')
            pos = JP_POS_MAP[pos]
            start = text.find(word, end)
            # if start != tmp_start:
            #     print('wooooooow: [{}] ({}) in [{}]'.format(word, end, text[end:]))
            end = start + len(word)
            words.append((word, pos, start, end))
            if word == '。':
                sent_list.append(words)
                words = []
        # add last sentence
        if words:
            sent_list.append(words)
        return sent_list

    def to_corenlp_json(self, text: str) -> str:
        sentx_list = self.to_sent_list(text)
        sent_json_list = []  # type: List[Dict]
        for senti, sentx in enumerate(sentx_list):
            token_list = []  # type: List[Dict]
            for wordi, (word, pos, start, end) in enumerate(sentx, 1):
                token_list.append({'characterOffsetBegin': start,
                                   'characterOffsetEnd': end,
                                   'index': wordi,
                                   'pos': pos,
                                   'word': word,
                                   'lemma': word,
                                   'ner': 'O'})
            sent_json = {'index': senti,
                         'tokens': token_list}
            sent_json_list.append(sent_json)
        out_sents_json = {'sentences': sent_json_list}
        return json.dumps(out_sents_json)
