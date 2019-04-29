#!/usr/bin/env python3

import configparser
import json
import os
import pprint
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
        self.maxDiff = None
        config = annotatorconfig.get_ml_annotator_config(['CURRENCY'])
        currency_gen = config['doc_to_candidates'][0]
        with open('cands_json/currency_cands.json') as infile:
            fname_currency_cands = json.load(infile)

        for fname in sorted(TEXT_FNAMES)[:25]:
            with open(fname) as txt_doc:
                doc_text = txt_doc.read()
                currency_cands, _, _ = currency_gen.get_candidates_from_text(doc_text)
                cand_st_list = [x['chars'] for x in currency_cands]
                print('currency fname: {}'.format(fname))
                print('  currency_cands: {}'.format(cand_st_list))

                self.assertEqual({'fname': fname,
                                  'cand_st_list': fname_currency_cands[fname]},
                                  {'fname': fname,
                                   'cand_st_list': cand_st_list})

    def test_bespoke_cand_date(self):
        # ------- DATES -------
        self.maxDiff = None
        config = annotatorconfig.get_ml_annotator_config(['DATE'])
        date_gen = config['doc_to_candidates'][0]
        with open('cands_json/date_cands.json') as infile:
            serial_date_cands = json.load(infile)

        all_date_cands = []  # type: List
        for fname in sorted(TEXT_FNAMES)[:25]:
            with open(fname) as txt_doc:
                doc_text = txt_doc.read()
                date_cands, _, _ = date_gen.get_candidates_from_text(doc_text)
                all_date_cands.extend([x['chars'] for x in date_cands])

        self.assertEqual(serial_date_cands, all_date_cands)

    def test_bespoke_cand_number(self):
        # ------- NUMBER -------
        self.maxDiff = None
        config = annotatorconfig.get_ml_annotator_config(['NUMBER'])
        number_gen = config['doc_to_candidates'][0]
        with open('cands_json/number_cands.json') as infile:
            fname_number_cands = json.load(infile)

        for fname in sorted(TEXT_FNAMES)[:25]:
            with open(fname) as txt_doc:
                doc_text = txt_doc.read()
                number_cands, _, _ = number_gen.get_candidates_from_text(doc_text)

                cand_st_list = [x['chars'] for x in number_cands]
                print('fname: {}'.format(fname))
                print('    cands: {}'.format(cand_st_list))

                self.assertEqual({'fname': fname,
                                  'cand_st_list': fname_number_cands[fname]},
                                  {'fname': fname,
                                   'cand_st_list': cand_st_list})



    def test_bespoke_cand_percent(self):
        # ------- PERCENT -------
        self.maxDiff = None
        config = annotatorconfig.get_ml_annotator_config(['PERCENT'])
        percent_gen = config['doc_to_candidates'][0]
        with open('cands_json/percent_cands.json') as infile:
            fname_percent_cands = json.load(infile)

        """
        all_number_cands = []
        fn_cand_st_list_map = {}
        for fname in sorted(TEXT_FNAMES)[:25]:
            with open(fname) as txt_doc:
                doc_text = txt_doc.read()
                percent_cands, _, _ = percent_gen.get_candidates_from_text(doc_text)
                all_percent_cands.extend([x['chars'] for x in percent_cands])

                st_list = [x['chars'] for x in percent_cands]
                fn_cand_st_list_map[fname] = st_list

        print('fn_cand_st_list:')
        # pprint.pprint(fn_cand_st_list_map)
        print(json.dumps(fn_cand_st_list_map))

        # intentionally cause failure to print above
        self.assertEqual(fn_cand_st_list_map, [])
        """

        for fname in sorted(TEXT_FNAMES)[:25]:
            with open(fname) as txt_doc:
                doc_text = txt_doc.read()
                percent_cands, _, _ = percent_gen.get_candidates_from_text(doc_text)

                cand_st_list = [x['chars'] for x in percent_cands]
                print('fname: {}'.format(fname))
                print('    cands: {}'.format(cand_st_list))

                self.assertEqual({'fname': fname,
                                  'cand_st_list': fname_percent_cands[fname]},
                                  {'fname': fname,
                                   'cand_st_list': cand_st_list})


    def test_bespoke_cand_idnum(self):
        # ------- IDNUM -------
        self.maxDiff = None
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
