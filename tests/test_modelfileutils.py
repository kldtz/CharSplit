#!/usr/bin/env python3

import unittest

from kirke.utils import modelfileutils

WORK_DIR = 'dir-work'
MODEL_DIR = 'dir-scut-model'
CUSTOM_MODEL_DIR = 'eb_files_test/pymodel'

class TestModelFileUtils(unittest.TestCase):

    def test_get_custom_model_file_names(self):
        model_files = modelfileutils.get_custom_model_file_names(CUSTOM_MODEL_DIR)
        for model_file in model_files:
            print("model_file: [{}]".format(model_file))
        self.assertGreater(len(model_files), 0)

    def test_find_highest_version(self):
        "Test find highest version of a custom model"

        model_records = modelfileutils.get_custom_model_records(CUSTOM_MODEL_DIR)
        prov_ver = modelfileutils.find_highest_version('cust_9', model_records)
        version = 0
        # skip till "cust_9."
        if prov_ver and prov_ver[7:]:
            version = int(prov_ver[7:])
        self.assertGreater(version, 0)

    def test_update_custom_prov_with_version(self):
        provs = ['abc', 'cust_9', 'cust_101']

        updated_provs = modelfileutils.update_custom_prov_with_version(provs,
                                                                       CUSTOM_MODEL_DIR)
        print('update_provs: {}'.format(updated_provs))
        self.assertEquals(len(updated_provs), 3)
        self.assertTrue('abc' in updated_provs)
        self.assertTrue('cust_101' in updated_provs)
        self.assertFalse('cust_9' in updated_provs)

    def test_get_custom_model_records(self):
        model_records = modelfileutils.get_custom_model_file_names(CUSTOM_MODEL_DIR)
        self.assertGreater(len(model_records), 0)

    def test_get_custom_prov_versions(self):
        prov_set = modelfileutils.get_custom_prov_versions(CUSTOM_MODEL_DIR)
        self.assertTrue('cust_9.1005' in prov_set)

    def test_get_custom_prov_ver_langs(self):
        prov_set = modelfileutils.get_custom_prov_ver_langs(CUSTOM_MODEL_DIR)
        self.assertTrue('cust_9.1005' in prov_set)
        self.assertTrue('cust_38.1010_ro' in prov_set)

    def test_get_custom_model_files(self):
        prov_ver_lang_fname_list = \
            modelfileutils.get_custom_model_files(CUSTOM_MODEL_DIR,
                                                  set(['cust_9.1005',
                                                       'cust_38.1010_ro']))
        print("prov_ver_lang_fname_list: {}".format(prov_ver_lang_fname_list))
        self.assertTrue(('cust_9.1005', 'cust_9.1005_CURRENCY_annotator.v1.0.pkl')
                        in prov_ver_lang_fname_list)
        self.assertTrue(('cust_38.1010_ro', 'cust_38.1010_ro_scutclassifier.v1.2.pkl')
                        in prov_ver_lang_fname_list)

    def test_get_provision_custom_model_files(self):
        prov_ver_lang_fname_list = \
            modelfileutils.get_provision_custom_model_files(CUSTOM_MODEL_DIR,
                                                            'cust_9.1005')
        self.assertTrue(('cust_9.1005', 'cust_9.1005_CURRENCY_annotator.v1.0.pkl')
                        in prov_ver_lang_fname_list)

        prov_ver_lang_fname_list = \
            modelfileutils.get_provision_custom_model_files(CUSTOM_MODEL_DIR,
                                                                   'cust_38.1010')
        self.assertTrue(('cust_38.1010_ro', 'cust_38.1010_ro_scutclassifier.v1.2.pkl')
                        in prov_ver_lang_fname_list)


    def test_get_default_model_file_names(self):
        fnames = modelfileutils.get_default_model_file_names(MODEL_DIR)
        expected = set(['sigdate_scutclassifier.v1.2.1.pkl',
                        'date_scutclassifier.v1.2.pkl',
                        'ea_agreement_termination_scutclassifier.v1.2.pkl',
                        'title_scutclassifier.v1.2.1.pkl',
                        'limliability_scutclassifier.v1.2.1.pkl',
                        'termination_scutclassifier.v1.2.pkl',
                        'term_scutclassifier.v1.2.1.pkl',
                        'force_majeure_scutclassifier.v1.2.1.pkl',
                        'party_scutclassifier.v1.2.pkl',
                        'renewal_scutclassifier.v1.2.pkl',
                        'choiceoflaw_scutclassifier.v1.2.1.pkl',
                        'effectivedate_scutclassifier.v1.2.1.pkl',
                        'noncompete_scutclassifier.v1.2.1.pkl',
                        'remedy_scutclassifier.v1.2.pkl',
                        'warranty_scutclassifier.v1.2.pkl',
                        'change_control_scutclassifier.v1.2.1.pkl',
                        'korean_ko_scutclassifier.v1.2.pkl'])
        print('fnames: {}'.format(fnames))
        self.assertEquals(set(fnames), expected)


    def test_get_default_provisions(self):
        provs = modelfileutils.get_default_provisions(MODEL_DIR)
        print('provs: {}'.format(provs))
        expected = {'effectivedate', 'termination',
                    'noncompete', 'ea_agreement_termination',
                    'party', 'term', 'limliability',
                    'renewal', 'choiceoflaw', 'remedy',
                    'warranty', 'sigdate', 'force_majeure',
                    'change_control', 'date', 'title',
                    'korean'}   # test-only model for Korean
        self.assertEquals(provs, expected)

if __name__ == "__main__":
    unittest.main()

