#!/usr/bin/env python3

import json
import unittest

from kirke.utils import osutils
from kirke.utils.ebantdoc4 import html_to_ebantdoc, pdf_to_ebantdoc
from kirke.eblearn import annotatorconfig

WORK_DIR = 'dir-work'

osutils.mkpath(WORK_DIR)

def get_antdoc(txt_fname):
    offsets_fname = txt_fname.replace('.txt', '.offsets.json')
    pdftxt_fname = txt_fname.replace('.txt', '.pdf.xml')
    ebantdoc = pdf_to_ebantdoc(txt_fname,
                               offsets_fname,
                               pdftxt_fname,
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

    # ---------- PDF DOCS ----------
    def test_demo_doc_8285(self):
        txt_base_name = '8285.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        # self.assertEqual(para_md5, '22cc9c825b76a26cd1aadb7797035af7')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8285_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(cands, serial_para_cands)

    def test_demo_doc_8286(self):
        txt_base_name = '8286.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        # self.assertEqual(para_md5, '665ceb06fc325b47d00db6ee4935e0a1')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8286_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(cands, serial_para_cands)


    def test_demo_doc_8287(self):
        txt_base_name = '8287.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        # self.assertEqual(para_md5, '6e0e6e26970b6aad06ec280e7dbdbeaa')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8287_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(cands, serial_para_cands)


    # ---------- TXT DOCS ----------
    def test_demo_doc_8288(self):
        txt_base_name = '8288.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = html_to_ebantdoc(txt_fname,
                                    WORK_DIR,
                                    is_cache_enabled=False)
        para_md5 = get_para_md5(ebantdoc)
        # self.assertEqual(para_md5, '241a21df118755bf43c60a46d5242e02')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8288_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(cands, serial_para_cands)


    def test_demo_doc_8289(self):
        txt_base_name = '8289.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = html_to_ebantdoc(txt_fname,
                                    WORK_DIR,
                                    is_cache_enabled=False)
        para_md5 = get_para_md5(ebantdoc)
        # self.assertEqual(para_md5, '0ead8f680a26fb1dd13012d12fe1a8a0')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8289_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(cands, serial_para_cands)


    def test_demo_doc_8290(self):
        txt_base_name = '8290.txt'
        txt_fname = 'demo-txt/{}'.format(txt_base_name)
        ebantdoc = html_to_ebantdoc(txt_fname,
                                    WORK_DIR,
                                    is_cache_enabled=False)
        para_md5 = get_para_md5(ebantdoc)
        # self.assertEqual(para_md5, 'a98eafafc48ba69b97540b103fdea234')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/8290_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(cands, serial_para_cands)


    def test_para_docs_1953(self):
        txt_base_name = '1953.txt'
        txt_fname = 'paragraph-tests/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        # self.assertEqual(para_md5, 'e2a41100569ccb6c3157f263e4af7142')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/1953_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(cands, serial_para_cands)


    def test_para_docs_3388(self):
        txt_base_name = '3388.txt'
        txt_fname = 'paragraph-tests/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        # self.assertEqual(para_md5, '020f47452ff9207baad7d46870473db1')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/3388_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(cands, serial_para_cands)


    def test_para_docs_1960(self):
        txt_base_name = '1960.txt'
        txt_fname = 'paragraph-tests/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        # self.assertEqual(para_md5, 'c0369eca113edee074ee137bc3e95921')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/1960_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(cands, serial_para_cands)


    def test_para_docs_1964(self):
        txt_base_name = '1964.txt'
        txt_fname = 'paragraph-tests/{}'.format(txt_base_name)
        ebantdoc = get_antdoc(txt_fname)
        para_md5 = get_para_md5(ebantdoc)
        # self.assertEqual(para_md5, '51ee09188609529622d41b96d7db757f')

        config = annotatorconfig.get_ml_annotator_config(['PARAGRAPH'])
        para_gen = config['doc_to_candidates'][0]
        with open('paragraphs/1964_para_cands.json') as infile:
            serial_para_cands = json.load(infile)

        para_cands, _, _ = para_gen.get_candidates_from_ebantdoc(ebantdoc)
        cands = [x['text'] for x in para_cands]
        self.assertEqual(cands, serial_para_cands)