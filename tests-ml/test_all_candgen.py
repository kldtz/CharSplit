#!/usr/bin/env python3

import configparser
import json
import os
import unittest

from typing import List

from kirke.eblearn import annotatorconfig
from kirke.sampleutils import idnumgen

# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'

def init_text_file_names() -> List[str]:
    custid = '9'
    custid_data_dir = 'cust_' + custid

    txt_fnames = []
    for file in os.listdir(custid_data_dir):
        fname = '{}/{}'.format(custid_data_dir, file)
        if file.endswith(".txt"):
            txt_fnames.append(fname)
    return txt_fnames

TEXT_FNAMES = init_text_file_names()

class TestCandGen(unittest.TestCase):

    def test_bespoke_cand_address(self):
        # ------- ADDRESS -------
        self.maxDiff = None

        config = annotatorconfig.get_ml_annotator_config(['ADDRESS'])
        address_gen = config['doc_to_candidates'][0]
        all_address_cands = []
        with open('cands_json/address_cands.json') as infile:
            serial_address_cands = json.load(infile)

        for fname in sorted(TEXT_FNAMES)[:25]:
            with open(fname) as txt_doc:
                doc_text = txt_doc.read()
                address_cands, _, _ = address_gen.get_candidates_from_text(doc_text)
                # for acand in [x['chars'] for x in address_cands]:
                #     print("acand: [{}]".format(acand))
                all_address_cands.extend([x['chars'] for x in address_cands])

        self.assertEqual(serial_address_cands, all_address_cands)

    def test_bespoke_cand_currency(self):
        # ------- CURRENCY -------
        config = annotatorconfig.get_ml_annotator_config(['CURRENCY'])
        currency_gen = config['doc_to_candidates'][0]
        all_currency_cands = []
        with open('cands_json/currency_cands.json') as infile:
            serial_currency_cands = json.load(infile)

        for fname in sorted(TEXT_FNAMES)[:25]:
            with open(fname) as txt_doc:
                doc_text = txt_doc.read()
                currency_cands, _, _ = currency_gen.get_candidates_from_text(doc_text)
                all_currency_cands.extend([x['chars'] for x in currency_cands])

        self.assertEqual(serial_currency_cands, all_currency_cands)

    def test_bespoke_cand_date(self):
        # ------- DATES -------
        config = annotatorconfig.get_ml_annotator_config(['DATE'])
        date_gen = config['doc_to_candidates'][0]
        all_date_cands = []
        with open('cands_json/date_cands.json') as infile:
            serial_date_cands = json.load(infile)

        for fname in sorted(TEXT_FNAMES)[:25]:
            with open(fname) as txt_doc:
                doc_text = txt_doc.read()
                date_cands, _, _ = date_gen.get_candidates_from_text(doc_text)
                all_date_cands.extend([x['chars'] for x in date_cands])

        self.assertEqual(serial_date_cands, all_date_cands)

    def test_bespoke_cand_number(self):
        # ------- NUMBER -------
        config = annotatorconfig.get_ml_annotator_config(['NUMBER'])
        number_gen = config['doc_to_candidates'][0]
        all_number_cands = []
        with open('cands_json/number_cands.json') as infile:
            serial_number_cands = json.load(infile)

        for fname in sorted(TEXT_FNAMES)[:25]:
            with open(fname) as txt_doc:
                doc_text = txt_doc.read()
                number_cands, _, _ = number_gen.get_candidates_from_text(doc_text)
                all_number_cands.extend([x['chars'] for x in number_cands])

        self.assertEqual(serial_number_cands, all_number_cands)

    def test_bespoke_cand_percent(self):
        # ------- PERCENT -------
        config = annotatorconfig.get_ml_annotator_config(['PERCENT'])
        percent_gen = config['doc_to_candidates'][0]
        all_percent_cands = []
        with open('cands_json/percent_cands.json') as infile:
            serial_percent_cands = json.load(infile)

        for fname in sorted(TEXT_FNAMES)[:25]:
            with open(fname) as txt_doc:
                doc_text = txt_doc.read()
                percent_cands, _, _ = percent_gen.get_candidates_from_text(doc_text)
                all_percent_cands.extend([x['chars'] for x in percent_cands])

        self.assertEqual(serial_percent_cands, all_percent_cands)

    def test_bespoke_cand_idnum(self):
        # ------- IDNUM -------
        config = annotatorconfig.get_ml_annotator_config(['IDNUM'])
        idnum_gen = config['doc_to_candidates'][0]
        all_idnum_cands = []
        with open('cands_json/idnum_cands.json') as infile:
            serial_idnum_cands = json.load(infile)

        for fname in sorted(TEXT_FNAMES)[:25]:
            with open(fname) as txt_doc:
                doc_text = txt_doc.read()
                idnum_cands = idnumgen.extract_idnum_list(doc_text, idnum_gen.regex_pat)
                all_idnum_cands.extend([x['chars'] for x in idnum_cands])

        self.assertEqual(serial_idnum_cands, all_idnum_cands)
