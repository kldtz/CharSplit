#!/usr/bin/env python3

import configparser
import json
import pprint
import unittest

from kirke.client import postfileutils


# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

RO_ANNOT = [{'cpoint_start': 46076,
             'cpoint_end': 46189,
             'start': 46076,
             # pylint: disable=line-too-long
             'text': 'Cred că este inacceptabil să nu avem curajul sau luciditatea de a vota în favoarea unei rezoluţii după dezbatere.',
             'label': 'cust_39',
             'prob': 1.0,
             'span_list': [{'cpoint_end': 46189,
                            'cpoint_start': 46076,
                            'start': 46076,
                            'end': 46189}],
             'end': 46189,
             'corenlp_start': 46076,
             'corenlp_end': 46189}]

NL_ANNOT = [{'cpoint_start': 79487,
             'cpoint_end': 79535,
             'start': 79487,
             'text': 'Om die reden zullen we voor dit verslag stemmen.',
             'label': 'cust_39',
             'prob': 0.7412521328665254,
             'span_list': [{'cpoint_end': 79535,
                            'cpoint_start': 79487,
                            'start': 79487, 'end': 79535}],
             'end': 79535,
             'corenlp_start': 79487,
             'corenlp_end': 79535}]

class TestBespokeSent(unittest.TestCase):
    '''
    Tagged Romanian and Dutch europarl data. Of the tagged data from other
    testing, these two langs get F1 > 0.0 with only 5 docs, so we use these
    two to keep unit testing as quick as possible.
    '''

    # pylint: disable=too-many-locals
    def test_bespoke_sent(self):

        # train the model with data from both langs
        custid = '39'
        custid_data_dir = 'cust_' + custid
        result_text = \
            postfileutils.upload_train_dir(custid,
                                           custid_data_dir,
                                           candidate_types='SENTENCE',
                                           nbest=-1)

        result_json = json.loads(result_text)
        ro_version_num = result_json['ro']['model_number']
        nl_version_num = result_json['nl']['model_number']

        ro_prov = "{}.{}".format(custid_data_dir, ro_version_num)
        nl_prov = "{}.{}".format(custid_data_dir, nl_version_num)

        print('ro_prov: [{}]'.format(ro_prov))

        # verify expected predictions on a training doc for each
        docid_pred_ajson_map = {}
        ro_doc_test = 'cust_39/2732.txt'
        docid_pred_ajson_map['ro'] = postfileutils.upload_annotate_doc(ro_doc_test, [ro_prov])
        print('3333')
        print(docid_pred_ajson_map['ro'])

        self.assertEqual(1, len(docid_pred_ajson_map['ro']['ebannotations'][custid_data_dir]))
        ro_annot = docid_pred_ajson_map['ro']['ebannotations'][custid_data_dir][0]
        start = ro_annot['start']
        end = ro_annot['end']
        text = ro_annot['text']
        # corenlp_start = ro_annot['corenlp_start']
        # corenlp_end = ro_annot['corenlp_end']
        self.assertEqual(46076, start)
        self.assertEqual(46189, end)
        # self.assertEqual(46076, corenlp_start)
        # self.assertEqual(46189, corenlp_end)
        # pylint: disable=line-too-long
        self.assertEqual('Cred că este inacceptabil să nu avem curajul sau luciditatea de a vota în favoarea unei rezoluţii după dezbatere.', text)

        nl_doc_test = 'cust_39/2783.txt'
        docid_pred_ajson_map['nl'] = postfileutils.upload_annotate_doc(nl_doc_test, [nl_prov])
        self.assertEqual(len(docid_pred_ajson_map['nl']['ebannotations'][custid_data_dir]), 1)
        nl_annot = docid_pred_ajson_map['nl']['ebannotations'][custid_data_dir][0]

        print('nl_annot:')
        pprint.pprint(docid_pred_ajson_map['nl']['ebannotations'][custid_data_dir])

        start = nl_annot['start']
        end = nl_annot['end']
        text = nl_annot['text']
        # corenlp_start = nl_annot['corenlp_start']
        # corenlp_end = nl_annot['corenlp_end']
        self.assertEqual(start, 79487)
        self.assertEqual(end, 79535)
        # self.assertEqual(79487, corenlp_start)
        # self.assertEqual(79535, corenlp_end)
        self.assertEqual('Om die reden zullen we voor dit verslag stemmen.', text)

        # 3rd language without a model, should come back empty
        en_doc_test = 'cust_10/687.txt'
        docid_pred_ajson_map['en'] = postfileutils.upload_annotate_doc(en_doc_test, [nl_prov])
        self.assertEqual([], docid_pred_ajson_map['en']['ebannotations'][custid_data_dir])

if __name__ == "__main__":
    unittest.main()
