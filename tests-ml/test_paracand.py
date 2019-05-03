import json
import os
import pprint
import re
import unittest
from typing import Any, Dict, List, Tuple

from kirke.eblearn import ebrunner
from kirke.utils import osutils, strutils

"""
# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']
"""

# we don't need any model, 
# so set this to a temporary fake, empty dir
MODEL_DIR = 'dir-scut-model-empty'
osutils.mkpath(MODEL_DIR)

WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'



# setup a global annotator
EB_RUNNER = ebrunner.EbRunner(MODEL_DIR, WORK_DIR, CUSTOM_MODEL_DIR)
EB_LANGDETECT_RUNNER = ebrunner.EbLangDetectRunner()


def ant_json_para_to_st_list(ajson: Dict) -> Tuple[List[Tuple[int, int, int]],
                                                   List[str]]:
    para_dict_list = ajson['PARAGRAPH']
    out_str_list = []  # type: List[str]
    out_se_list = []  # type: List[Tuple[int, int]]
    for para_i, para_dict in enumerate(para_dict_list):
        para_text = '{}\t{}\t{}\t{}'.format(para_i,
                                            para_dict['start'],
                                            para_dict['end'],
                                            para_dict['text'].replace('\n', '|'))
        print(para_text)
        print('len = {}, diff = {}'.format(len(para_dict['text']),
                                           para_dict['end'] - para_dict['start']))
                                           
        out_se_list.append((para_i,
                            para_dict['start'],
                            para_dict['end']))
        out_str_list.append(para_dict['text'].replace('\n', '|'))
    return out_se_list, out_str_list


def load_gold_para_tsv(fname: str) -> Tuple[List[Tuple[int, int, int]],
                                            List[str]]:
    gold_st_list = strutils.load_str_list(fname)
    out_str_list = []  # type: List[str]
    out_se_list = []  # type: List[Tuple[int, int]]
    for line in gold_st_list:
        # para_i, start, end, text = line.split('\t')
        cols = line.split('\t')
        # there can be tab in the text file.  Sigh.
        if len(cols) == 4:
            para_i, start, end, text = cols
        else:
            para_i, start, end = cols[0], cols[1], cols[2]
            text = '\t'.join(cols[3:])

        out_se_list.append((int(para_i),
                            int(start),
                            int(end)))
        out_str_list.append(text)
    return out_se_list, out_str_list


def annotate_document(file_name: str) -> Dict[str, Any]:
    atext = strutils.loads(file_name)
    doc_lang = EB_LANGDETECT_RUNNER.detect_lang(atext)
    if not doc_lang:
        doc_lang = 'en'
    print("detected language '%s'" % doc_lang)

    provision_set = set(['PARAGRAPH'])
    prov_labels_map, _ = EB_RUNNER.annotate_document(file_name,
                                                     provision_set=provision_set,
                                                     work_dir=WORK_DIR,
                                                     doc_lang=doc_lang,
                                                     is_doc_structure=True,
                                                     is_dev_mode=False)
    return dict(prov_labels_map)


class TestParaCandGen(unittest.TestCase):

    def test_paracand_9310(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9310.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9310.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)


    def test_paracand_9311(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9311.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9311.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)


    def test_paracand_9312(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9312.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9312.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)


    def test_paracand_9313(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9313.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9313.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)

        
    def test_paracand_9314(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9314.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9314.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)


    def test_paracand_9315(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9315.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9315.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)


    def test_paracand_9316(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9316.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9316.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)


    def test_paracand_9317(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9317.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9317.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)


    def test_paracand_9318(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9318.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9318.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)
        
        
    def test_paracand_9319(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9319.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9319.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)

        
    def test_paracand_9320(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9320.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9320.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)
        

    def test_paracand_9321(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9321.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9321.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)

        
    def test_paracand_9322(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9322.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9322.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)
        

    def test_paracand_9323(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9323.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9323.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)
        

    def test_paracand_9324(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9324.para.tsv')

        ant_result = annotate_document('dir-paracand/demo-txt/9324.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)
        

    # baker doc #1
    def test_paracand_9325(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9325.para.tsv')

        ant_result = annotate_document('dir-paracand/baker-txt/9325.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)
        

    # baker doc #2        
    def test_paracand_9326(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/9326.para.tsv')

        ant_result = annotate_document('dir-paracand/baker-txt/9326.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)
        
                

    # sample doc, 2 column text
    def test_paracand_8308(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8308.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8308.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)
        
        
    # sample doc, template in page 1, 3 column text in page 3, 4, 5
    def test_paracand_8203(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8203.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8203.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)


    # sample doc, normal 1-column pages
    def test_paracand_8075(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8075.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8075.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)

        
    # TODO
    # sample doc, email + other issues.  Font and space change in page 3 is NOT handled
    # correctly.  Passing it for now.
    def test_paracand_3937(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/3937.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/3937.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)


    def test_paracand_3937(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/3937.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/3937.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)


    # normal
    def test_paracand_8075(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8075.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8075.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
        
                
    # normal, but 2 pages only
    def test_paracand_8916(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8916.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8916.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
        

    # 8917.txt is ignorable, in Dutch?

    # Withtout this PR, each line would be a separate paragraph
    # because PDFBox behaved weirdly.
    # originally failed due to 3 new lines between lines inside a
    # paragraph in 8918.paraline.txt
    # Original PDF, "2.3.1.18.9 180718-AXA-TA-3.pdf"
    def test_paracand_8918(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8918.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8918.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
                
        
    def test_paracand_8919(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8919.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8919.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
                

    def test_paracand_8920(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8920.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8920.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                


    # 2 column, right sight russian?
    # otherwise, normal
    def test_paracand_8921(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8921.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8921.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
        

    # normal
    def test_paracand_8923(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8923.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8923.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
        
        
    # normal
    # changed font and line spacing
    def test_paracand_8924(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8924.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8924.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
        

    # normal, just 1 page doc
    def test_paracand_8926(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8926.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8926.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                


    # normal, 57 page doc
    def test_paracand_8927(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8927.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8927.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
        

    # normal, 2 page email
    def test_paracand_8928(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8928.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8928.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
        

    def test_paracand_8929(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8929.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8929.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
        

    def test_paracand_8930(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8930.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8930.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                


    def test_paracand_8931(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8931.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8931.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                


    # Double-spaced text in pages, with just one paragraph per doc in .txt
    # checked the annotation is basically the same
    # TODO, for future
    # On page 19, the UI is messed up on table branch
    # In the detect-one-para-per-page branch, the text for section
    # 23.5 is missing in nlp text.  Something weird is going on there.
    # Won't fix for now.  The rest of the annotaion is OK.
    def test_paracand_8932(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8932.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8932.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
        

    def test_paracand_8933(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8933.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8933.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                


    # Horrible document because of folds in paper messed up sentences in OCR.
    # It is treated as if there is a table in the pages.
    def test_paracand_8934(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8934.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8934.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                


    def test_paracand_8935(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8935.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8935.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                


    def test_paracand_8936(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8936.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8936.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                


    def test_paracand_8937(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8937.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8937.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                


    # document with a lot of tables
    # some font changes.  Paragraphs look good.
    def test_paracand_8938(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8938.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8938.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                

    # only 2 column in page 3, others pages contains lists, but in
    # some-what-2-column format
    def test_paracand_8939(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8939.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8939.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                


    def test_paracand_8940(self):
        gold_se_list, gold_str_list = load_gold_para_tsv('dir-paracand/gold-target/8940.para.tsv')

        ant_result = annotate_document('dir-paracand/samples-100-txt/8940.txt')
        # pprint.pprint(ant_result)
        # pprint.pprint(ant_json_para_to_st_list(ant_result))
        test_se_list, test_str_list = ant_json_para_to_st_list(ant_result)

        # the str_list might not be equal because
        # nl_text seems to have removed page numbers
        self.assertEqual(test_se_list,
                         gold_se_list)                
