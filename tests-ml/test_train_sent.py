#!/usr/bin/env python3

import configparser
import unittest

from kirke.eblearn import ebtrainer, scutclassifier


# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'

class TestTrainSent(unittest.TestCase):

    # pylint: disable=too-many-locals
    def test_train_xscut_ea_agreement_termination(self):

        provision = 'ea_agreement_termination'
        work_dir = WORK_DIR
        model_dir = MODEL_DIR
        eb_classifier = scutclassifier.ShortcutClassifier(provision)  # type: EbClassifier
        model_file_name = '{}/{}_scutclassifier.v{}.pkl'.format(model_dir,
                                                                provision,
                                                                SCUT_CLF_VERSION)
        is_cache_enabled = True
        is_doc_structure = True

        _, ant_result, _ = \
            ebtrainer.train_eval_annotator_with_trte(provision,
                                                     work_dir,
                                                     model_dir,
                                                     model_file_name,
                                                     eb_classifier,
                                                     is_cache_enabled=is_cache_enabled,
                                                     is_doc_structure=is_doc_structure)
        print("ant_result:")
        print(ant_result)

        conf_matrix = ant_result['confusion_matrix']

        tn = conf_matrix['tn']
        fp = conf_matrix['fp']
        fn = conf_matrix['fn']
        tp = conf_matrix['tp']

        self.assertEqual(tn, 0)
        self.assertAlmostEqual(fp, 1, delta=1)
        self.assertAlmostEqual(fn, 3, delta=1)
        self.assertAlmostEqual(tp, 11, delta=1)

        #self.assertEqual(conf_matrix,
        #                 {'fn': 3, 'fp': 1, 'tn': 0, 'tp': 11})
        f1 = round(ant_result['f1'], 2)
        self.assertAlmostEqual(f1, 0.85, delta=0.04)

        self.assertEqual(round(ant_result['prec'], 2),
                         0.92)
        self.assertEqual(round(ant_result['recall'], 2),
                         0.79)
        self.assertEqual(round(ant_result['threshold'], 2),
                         0.24)


if __name__ == "__main__":
    unittest.main()
