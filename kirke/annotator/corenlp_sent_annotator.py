#!/usr/bin/env python

from pycorenlp import StanfordCoreNLP

from kirke.annotator.annotation import SentenceAnnotation


class CoreNlpSentenceAnnotator(object):

    def __init__(self):
        self.nlp = StanfordCoreNLP('http://localhost:9500')

    def span_tokenize(self, text_as_string):
        # "ssplit.isOneSentence": "true"
        output = self.nlp.annotate(text_as_string, properties={
            'annotators': 'tokenize,ssplit,pos,lemma,ner',
            'outputFormat': 'json',
            'ner.model': 'edu/stanford/nlp/models/ner/english.muc.7class.distsim.crf.ser.gz',
            'ssplit.newlineIsSentenceBreak': 'two'
        })
        result = []
        for tagged_span in output['sentences']:
            tokens = tagged_span['tokens']
            sent_start_idx = tokens[0]['characterOffsetBegin']
            sent_end_idx = tokens[-1]['characterOffsetEnd']
            sent_st = text_as_string[sent_start_idx:sent_end_idx]
            result.append(SentenceAnnotation(sent_start_idx, sent_end_idx, sent_st))
        return result

    # WARNING: all the spaces before the first non-space character will be removed in the output.
    # In other words, the offsets will be incorrect if there are prefix spaces in the text.
    # We will fix those issues in the late modules, not here.
    def annotate(self, text_as_string):
        # "ssplit.isOneSentence": "true"
        output = self.nlp.annotate(text_as_string, properties={
            'annotators': 'tokenize,ssplit,pos,lemma,ner',
            'outputFormat': 'json',
            'ner.model': 'edu/stanford/nlp/models/ner/english.muc.7class.distsim.crf.ser.gz',
            'ssplit.newlineIsSentenceBreak': 'two'
        })

        return output
