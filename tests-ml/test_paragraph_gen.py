#!/usr/bin/env python3

import json
import unittest

from kirke.utils import osutils
from kirke.utils.ebantdoc4 import html_to_ebantdoc4, pdf_to_ebantdoc4
from kirke.eblearn import annotatorconfig

WORK_DIR = 'dir-work'

osutils.mkpath(WORK_DIR)

def get_antdoc(txt_fname):
    offsets_fname = txt_fname.replace('.txt', '.offsets.json')
    ebantdoc = pdf_to_ebantdoc4(txt_fname,
                                offsets_fname,
                                WORK_DIR)
    return ebantdoc

def get_para_md5(ebantdoc):
    nl_text = ebantdoc.get_nlp_text()
    # using md5 hashes here and rather than modifying the objects to serialize tuples
    # pylint: disable=line-too-long
    para_text = "\n---------\n".join([nl_text[para[0][1].start:para[-1][1].end] for para in ebantdoc.para_indices])
    para_md5 = osutils.get_text_md5(para_text)
    return para_md5

class TestParagraphGen(unittest.TestCase):

    # pylint: disable=too-many-statements
    def test_demo_docs(self):

        # ---------- PDF DOCS ----------

        txt_base_name = '8285.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        self.assertEqual(para_md5, '22cc9c825b76a26cd1aadb7797035af7')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8285_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(serial_para_cands, cands)



        txt_base_name = '8286.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        self.assertEqual(para_md5, '665ceb06fc325b47d00db6ee4935e0a1')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8286_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(serial_para_cands, cands)



        txt_base_name = '8287.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        self.assertEqual(para_md5, 'f2cefdd953c72e4bf711b7cbacb7fcfc')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8287_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(serial_para_cands, cands)

        # ---------- TXT DOCS ----------

        txt_base_name = '8288.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = html_to_ebantdoc4(txt_fname,
                                     WORK_DIR,
                                     is_cache_enabled=False)
        para_md5 = get_para_md5(ebantdoc)
        self.assertEqual(para_md5, '241a21df118755bf43c60a46d5242e02')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8288_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(serial_para_cands, cands)



        txt_base_name = '8289.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = html_to_ebantdoc4(txt_fname,
                                     WORK_DIR,
                                     is_cache_enabled=False)
        para_md5 = get_para_md5(ebantdoc)
        self.assertEqual(para_md5, '0ead8f680a26fb1dd13012d12fe1a8a0')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8289_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(serial_para_cands, cands)



        txt_base_name = '8290.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = html_to_ebantdoc4(txt_fname,
                                     WORK_DIR,
                                     is_cache_enabled=False)
        para_md5 = get_para_md5(ebantdoc)
        self.assertEqual(para_md5, 'a98eafafc48ba69b97540b103fdea234')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8290_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(serial_para_cands, cands)


    def test_para_docs(self):
        txt_base_name = '1953.txt'
        txt_fname = 'paragraph-tests/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        self.assertEqual(para_md5, 'e2a41100569ccb6c3157f263e4af7142')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/1953_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(serial_para_cands, cands)



        txt_base_name = '3388.txt'
        txt_fname = 'paragraph-tests/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        self.assertEqual(para_md5, '020f47452ff9207baad7d46870473db1')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/3388_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(serial_para_cands, cands)



        txt_base_name = '1960.txt'
        txt_fname = 'paragraph-tests/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        self.assertEqual(para_md5, 'c0369eca113edee074ee137bc3e95921')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/1960_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(serial_para_cands, cands)



        txt_base_name = '1964.txt'
        txt_fname = 'paragraph-tests/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        self.assertEqual(para_md5, 'fba6876bac892381699f871ddf3161c0')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/1964_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(serial_para_cands, cands)
