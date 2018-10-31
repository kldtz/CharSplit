#!/usr/bin/env python3

import configparser
import json
import unittest

from kirke.client import postfileutils
from kirke.utils import corenlputils

# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'

class TestLangs(unittest.TestCase):

    def test_lang_portuguese(self):
        # ------- PORTUGUESE -------
        pt_file = 'dir-test-doc/df6bbe33a74d9d968d37e88d98418dc0-967.txt'
        #check corenlp
        corenlp_result = corenlputils.check_pipeline_lang('pt', pt_file)

        sents = corenlp_result['sentences']
        ner_string = " ".join([tok['ner'] for tok in sents[3]['tokens']])
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'O O O O O O O O LOCAL LOCAL LOCAL O O O O O ABSTRACCAO ABSTRACCAO ABSTRACCAO O ORGANIZACAO O O O O ORGANIZACAO O O O O O O O O O O O O O O')
        ner_string = " ".join([tok['ner'] for tok in sents[42]['tokens']])
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'O ORGANIZACAO O O PESSOA O O O PESSOA O O O O ORGANIZACAO ORGANIZACAO ORGANIZACAO ORGANIZACAO ORGANIZACAO ORGANIZACAO ORGANIZACAO ORGANIZACAO O')
        ner_string = " ".join([tok['ner'] for tok in sents[241]['tokens']])
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'O OBRA OBRA OBRA OBRA OBRA OBRA OBRA OBRA OBRA OBRA OBRA OBRA O O O O LOCAL LOCAL O O O O O O O VALOR O O O O O O')

        # upload file
        result_text = \
                postfileutils.post_annotate_document(pt_file,
                                                     ['choiceoflaw'], # placeholder provision, could be anything
                                                     is_detect_lang=True)
        out_lang = json.loads(result_text)['lang']
        self.assertEqual(out_lang, 'pt')

    def test_lang_french(self):
        # ------- FRENCH -------
        fr_file = 'dir-test-doc/07b078e73cf1fcb953a2206c16b186f0-960.txt'

        corenlp_result = corenlputils.check_pipeline_lang('fr', fr_file)
        sents = corenlp_result['sentences']
        ner_string = " ".join([tok['ner'] for tok in sents[26]['tokens']])
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'O O O O O ORGANIZATION ORGANIZATION ORGANIZATION ORGANIZATION ORGANIZATION ORGANIZATION ORGANIZATION ORGANIZATION O O O O O O O O O O O O O DATE O')
        ner_string = " ".join([tok['ner'] for tok in sents[217]['tokens']])
        print(" ".join([tok['word'] + '/' + tok['ner'] for tok in sents[217]['tokens']]))
        self.assertEqual(ner_string, 'O O O O O O O O O O O O O O')
        ner_string = " ".join([tok['ner'] for tok in sents[243]['tokens']])
        print(" ".join([tok['word'] + '/' + tok['ner'] for tok in sents[243]['tokens']]))
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'LOCATION LOCATION LOCATION LOCATION LOCATION LOCATION LOCATION LOCATION O ORGANIZATION ORGANIZATION O LOCATION')

        result_text = \
                postfileutils.post_annotate_document(fr_file,
                                                     ['choiceoflaw'],
                                                     is_detect_lang=True)
        out_lang = json.loads(result_text)['lang']
        self.assertEqual(out_lang, 'fr')

    def test_lang_chinese(self):
        # ------- CHINESE -------
        zh_file = 'dir-test-doc/d3fb1a4753b1fa03b2ed2dff61475006-935.txt'

        corenlp_result = corenlputils.check_pipeline_lang('zh', zh_file)
        sents = corenlp_result['sentences']
        ner_string = " ".join([tok['ner'] for tok in sents[8]['tokens']])
        print(" ".join([tok['word'] + '/' + tok['ner'] for tok in sents[8]['tokens']]))
        self.assertEqual(ner_string, 'O O O O O O O O O O O O O O MISC O')
        ner_string = " ".join([tok['ner'] for tok in sents[13]['tokens']])
        print(" ".join([tok['word'] + '/' + tok['ner'] for tok in sents[13]['tokens']]))
        self.assertEqual(ner_string, 'O O O O MISC MISC MISC O O O O O O O O O O O O O MISC O')
        ner_string = " ".join([tok['ner'] for tok in sents[16]['tokens']])
        self.assertEqual(ner_string, 'O O O O O O O O O O O O O O O O GPE O O O O O O O O MISC O')

        result_text = \
                postfileutils.post_annotate_document(zh_file,
                                                     ['choiceoflaw'],
                                                     is_detect_lang=True)
        out_lang = json.loads(result_text)['lang']
        self.assertEqual(out_lang, 'zh-cn')

    def test_lang_spanish(self):
        # ------- SPANISH -------
        es_file = 'dir-test-doc/2fd54ff76e2d48f364fdf42f6210d9c0-933.txt'

        corenlp_result = corenlputils.check_pipeline_lang('es', es_file)
        sents = corenlp_result['sentences']
        ner_string = " ".join([tok['ner'] for tok in sents[6]['tokens']])
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'PERS PERS PERS O O O O ORG O O LUG LUG O O O O LUG LUG O O O O O O O O O O O LUG LUG O O O O LUG LUG O O OTROS O')
        ner_string = " ".join([tok['ner'] for tok in sents[18]['tokens']])
        self.assertEqual(ner_string, 'O O O O O O O O O O O O ORG ORG PERS PERS PERS O O ORG ORG ORG O')
        ner_string = " ".join([tok['ner'] for tok in sents[138]['tokens']])
        self.assertEqual(ner_string, 'PERS PERS PERS O O ORG ORG ORG ORG ORG O')

        result_text = \
                postfileutils.post_annotate_document(es_file,
                                                     ['choiceoflaw'],
                                                     is_detect_lang=True)
        out_lang = json.loads(result_text)['lang']
        self.assertEqual(out_lang, 'es')


        # de_file = 'dir-test-doc/7e5262689e831e875af142202f67faec-937.txt'

    def test_lang_english(self):
        # ------- ENGLISH -------
        en_file = 'dir-test-doc/60543.txt'

        corenlp_result = corenlputils.check_pipeline_lang('en', en_file)
        sents = corenlp_result['sentences']
        ner_string = " ".join([tok['ner'] for tok in sents[257]['tokens']])
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O LOCATION LOCATION O O O ORGANIZATION ORGANIZATION ORGANIZATION O O O O O O O O O O O O O')
        ner_string = " ".join([tok['ner'] for tok in sents[336]['tokens']])
        self.assertEqual(ner_string, 'ORGANIZATION ORGANIZATION ORGANIZATION O O')
        ner_string = " ".join([tok['ner'] for tok in sents[362]['tokens']])
        self.assertEqual(ner_string, 'NUMBER O ORDINAL O O LOCATION LOCATION O LOCATION LOCATION')

        result_text = \
                postfileutils.post_annotate_document(en_file,
                                                     ['choiceoflaw'],
                                                     is_detect_lang=True)
        out_lang = json.loads(result_text)['lang']
        self.assertEqual(out_lang, 'en')
