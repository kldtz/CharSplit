#!/usr/bin/env python3

import configparser
import json
import unittest
from typing import Dict, List

from kirke.client import postfileutils
from kirke.utils import corenlputils, ebantdoc4

# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'

def ja_sent_to_tokens(asent: Dict) -> List[str]:
    tokens = [adict['word'] for adict in asent['tokens']]
    return tokens


class TestLangs(unittest.TestCase):

    def test_lang_portuguese(self):
        # ------- PORTUGUESE -------
        pt_file = 'dir-test-doc/df6bbe33a74d9d968d37e88d98418dc0-967.txt'
        #check corenlp
        corenlp_result = corenlputils.check_pipeline_lang('pt', pt_file)

        sents = corenlp_result['sentences']
        ner_string = ' '.join([tok['ner'] for tok in sents[3]['tokens']])
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'O O O O O O O O LOCAL LOCAL LOCAL O O O O O ABSTRACCAO ABSTRACCAO ABSTRACCAO O ORGANIZACAO O O O O ORGANIZACAO O O O O O O O O O O O O O O')
        ner_string = ' '.join([tok['ner'] for tok in sents[42]['tokens']])
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'O ORGANIZACAO O O PESSOA O O O PESSOA O O O O ORGANIZACAO ORGANIZACAO ORGANIZACAO ORGANIZACAO ORGANIZACAO ORGANIZACAO ORGANIZACAO ORGANIZACAO O')
        ner_string = ' '.join([tok['ner'] for tok in sents[241]['tokens']])
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
        ner_string = ' '.join([tok['ner'] for tok in sents[26]['tokens']])
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'O O O O O ORGANIZATION ORGANIZATION ORGANIZATION ORGANIZATION ORGANIZATION ORGANIZATION ORGANIZATION ORGANIZATION O O O O O O O O O O O O O DATE O')
        ner_string = ' '.join([tok['ner'] for tok in sents[217]['tokens']])
        print(' '.join([tok['word'] + '/' + tok['ner'] for tok in sents[217]['tokens']]))
        self.assertEqual(ner_string, 'O O O O O O O O O O O O O O')
        ner_string = ' '.join([tok['ner'] for tok in sents[243]['tokens']])
        print(' '.join([tok['word'] + '/' + tok['ner'] for tok in sents[243]['tokens']]))
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
        ner_string = ' '.join([tok['ner'] for tok in sents[21]['tokens']])
        print(' '.join([tok['word'] + '/' + tok['ner'] for tok in sents[21]['tokens']]))
        self.assertEqual(ner_string, 'O O O O O O O O O O O O O O MISC O')
        ner_string = ' '.join([tok['ner'] for tok in sents[26]['tokens']])
        print(' '.join([tok['word'] + '/' + tok['ner'] for tok in sents[26]['tokens']]))
        self.assertEqual(ner_string, 'O O O O MISC MISC MISC O O O O O O O O O O O O O MISC O')
        ner_string = ' '.join([tok['ner'] for tok in sents[29]['tokens']])
        print(' '.join([tok['word'] + '/' + tok['ner'] for tok in sents[29]['tokens']]))
        self.assertEqual(ner_string, 'O O O O O O O O O O O O O O O O GPE O O O O O O O O')

        result_text = \
                postfileutils.post_annotate_document(zh_file,
                                                     ['choiceoflaw'],
                                                     is_detect_lang=True)
        out_lang = json.loads(result_text)['lang']
        self.assertEqual(out_lang, 'zh')  # zh-cn, but normalized to 'zh'

    def test_lang_spanish(self):
        # ------- SPANISH -------
        es_file = 'dir-test-doc/2fd54ff76e2d48f364fdf42f6210d9c0-933.txt'

        corenlp_result = corenlputils.check_pipeline_lang('es', es_file)
        sents = corenlp_result['sentences']
        ner_string = ' '.join([tok['ner'] for tok in sents[6]['tokens']])
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'PERS PERS PERS O O O O ORG O O LUG LUG O O O O LUG LUG O O O O O O O O O O O LUG LUG O O O O LUG LUG O O OTROS O')
        ner_string = ' '.join([tok['ner'] for tok in sents[18]['tokens']])
        self.assertEqual(ner_string, 'O O O O O O O O O O O O ORG ORG PERS PERS PERS O O ORG ORG ORG O')
        ner_string = ' '.join([tok['ner'] for tok in sents[138]['tokens']])
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
        ner_string = ' '.join([tok['ner'] for tok in sents[257]['tokens']])
        # pylint: disable=line-too-long
        self.assertEqual(ner_string, 'O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O LOCATION LOCATION O O O ORGANIZATION ORGANIZATION ORGANIZATION O O O O O O O O O O O O O')
        ner_string = ' '.join([tok['ner'] for tok in sents[336]['tokens']])
        self.assertEqual(ner_string, 'ORGANIZATION ORGANIZATION ORGANIZATION O O')
        ner_string = ' '.join([tok['ner'] for tok in sents[362]['tokens']])
        self.assertEqual(ner_string, 'NUMBER O ORDINAL O O LOCATION LOCATION O LOCATION LOCATION')

        result_text = \
                postfileutils.post_annotate_document(en_file,
                                                     ['choiceoflaw'],
                                                     is_detect_lang=True)
        out_lang = json.loads(result_text)['lang']
        self.assertEqual(out_lang, 'en')


    def test_lang_japanese(self):
        # ------- JAPANESE -------
        ja_file = 'data/japanese/txt/1010.txt'

        corenlp_result = corenlputils.check_pipeline_lang('ja', ja_file)
        sents = corenlp_result['sentences']
        # for sent_i, sent in enumerate(sents):
        #     print('sent_i [{}] {}'.format(sent_i, ja_sent_to_tokens(sents[sent_i])))

        # maintly to test tokenized words
        # if the tokens are reasonable, then bespoke training can work
        # ok.  No NER.  All 'O' in kytea
        # changed some characters to single-byte characters because
        # internally, Kirke uses only single-byte characters.
        gold_sent_0 = ['請負', '型', '/', '予定', '契約', '業務', '委託', '契約', '書']

        gold_sent_12 = ['2', '前項', 'の', '規定', 'に', 'かかわ', 'ら', 'ず', '、', '仕様', '書',
                       'に', 'お', 'い', 'て', '本', '契約', 'の', '内容', 'と', '矛盾',
                       '・', '抵触', 'する', '内容', 'が', '定め', 'られ', 'た', '場合', '、',
                       '仕様', '書', 'に', 'お', 'け', 'る', '当該', '定め', 'は', 'その', '効力',
                       'を', '有', 'し', 'な', 'い', 'もの', 'と', 'する', '。']

        gold_sent_19 = ['当該', '確定', '数量', 'に', '前項', 'に', '定め', 'る', '単価',
                        'を', '乗じ', 'る', 'こと', 'を', 'も', 'っ', 'て', '、', '本件',
                        '業務', 'の', '実施', 'の', '対価', 'の', '全部', '又',
                        'は', '一部', 'は', '確定', 'する', '。']

        self.assertEqual(gold_sent_0,
                         ja_sent_to_tokens(sents[0]))
        self.assertEqual(gold_sent_12,
                         ja_sent_to_tokens(sents[12]))
        self.assertEqual(gold_sent_19,
                         ja_sent_to_tokens(sents[19]))

        result_text = \
                postfileutils.post_annotate_document(ja_file,
                                                     ['choiceoflaw'],
                                                     is_detect_lang=True)
        out_lang = json.loads(result_text)['lang']
        self.assertEqual(out_lang, 'ja')


    def test_lang_japanese_nlptxt(self):
        """This test is based on .nlp.txt, which has better line break
        information than the above text.  As a result, the line numbers differ.
        """

        ja_file = 'data/japanese/txt/1010.txt'

        eb_antdoc = ebantdoc4.text_to_ebantdoc(ja_file,
                                               work_dir=WORK_DIR,
                                               doc_lang='ja')
        sent_words_list = []
        for attrvec_i, attrvec in enumerate(eb_antdoc.attrvec_list):
            out_st = attrvec.bag_of_words
            out_st = out_st.replace('\n', ' ')
            sent_words_list.append(out_st.split())
            # print('attrvec_i = {}, {}'.format(attrvec_i, out_st.split()))

        # maintly to test tokenized words
        # if the tokens are reasonable, then bespoke training can work
        # ok.  No NER.  All 'O' in kytea
        # changed some characters to single-byte characters because
        # internally, Kirke uses only single-byte characters.
        gold_sent_6 = ['3', '乙', 'は', '、', '本件', '業務', 'の', '実施', 'に', '当た', 'り',
                       '、', '以下', 'の', '(', '1', ')', '及び', '(', '2', ')', 'を', '遵守',
                       'する', 'もの', 'と', 'する', '。']
        gold_sent_15 = ['2', '前項', 'の', '規定', 'に', 'かかわ', 'ら', 'ず', '、', '仕様', '書',
                       'に', 'お', 'い', 'て', '本', '契約', 'の', '内容', 'と', '矛盾',
                       '・', '抵触', 'する', '内容', 'が', '定め', 'られ', 'た', '場合', '、',
                       '仕様', '書', 'に', 'お', 'け', 'る', '当該', '定め', 'は', 'その', '効力',
                       'を', '有', 'し', 'な', 'い', 'もの', 'と', 'する', '。']

        gold_sent_22 = ['当該', '確定', '数量', 'に', '前項', 'に', '定め', 'る', '単価',
                        'を', '乗じ', 'る', 'こと', 'を', 'も', 'っ', 'て', '、', '本件',
                        '業務', 'の', '実施', 'の', '対価', 'の', '全部', '又',
                        'は', '一部', 'は', '確定', 'する', '。']

        self.assertEqual(gold_sent_6,
                         sent_words_list[6])
        self.assertEqual(gold_sent_15,
                         sent_words_list[15])
        self.assertEqual(gold_sent_22,
                         sent_words_list[22])
