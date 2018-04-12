# pylint: disable=no-name-in-module, import-error
from distutils.version import StrictVersion
import itertools
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from kirke.sampleutils import postproc
from kirke.ebrules import addrannotator, dummyannotator, dates
from kirke.sampleutils import sectiongen, regexgen, addrgen, paragen, dategen, transformerutils
from kirke.utils import ebantdoc3, ebantdoc2

# Different provisions can use the same candidate type, such as Currency
# or Number.  The specifications of 'provision' are outside of annotatorconfig.py.
#
# There are "frozen" lists of those config so that developer are aware not to touch
# any of the classes mentioned in the frozen config lists.


ML_ANNOTATOR_CONFIG_LIST = [
    ('DATE', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                     'doc_to_candidates': dategen.DateSpanGenerator(13, 13, 'DATE'),
                     'version': "1.0",
                     'doc_postproc_list': [dates.DateNormalizer(),
                                           postproc.SpanDefaultPostProcessing()],
                     'pipeline': Pipeline([
                         # pylint: disable=line-too-long
                         ('surround_transformer', transformerutils.SurroundWordTransformer()),
                         ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                               shuffle=True, random_state=42,
                                               class_weight={True: 3, False: 1}))]),
                     'threshold': 0.2,
                     'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(3, 7)}}),

    ('ADDRESS', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                        'doc_to_candidates': addrgen.AddrContextGenerator(10, 10, 'ADDRESS'),
                        'version': "1.0",
                        'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                        'pipeline': Pipeline([
                            ('union', FeatureUnion(
                                transformer_list=[
                                    # pylint: disable=line-too-long
                                    ('surround_transformer', transformerutils.SurroundWordTransformer()),
                                    ('is_addr_line_transformer', transformerutils.AddrLineTransformer())
                                ])),
                            ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                  shuffle=True, random_state=42,
                                                  class_weight={True: 3, False: 1}))]),
                        'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(4, 6)},
                        'threshold': 0.25,
                        'kfold': 3}),
    ('CURRENCY', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                         'doc_to_candidates': regexgen.RegexContextGenerator(20,
                                                                             5,
                                                                             # pylint: disable=line-too-long
                                                                             re.compile(r'([\$€₹£¥] *(\d{1,3},?)+([,\.]\d\d)?)[€円]?'),
                                                                             'CURRENCY'),
                         'version': "1.0",
                         'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                         'pipeline': Pipeline([
                             ('union', FeatureUnion(
                                 transformer_list=[
                                     # pylint: disable=line-too-long
                                     ('surround_transformer', transformerutils.SimpleTextTransformer())
                                 ])),
                             ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                   shuffle=True, random_state=42,
                                                   class_weight={True: 3, False: 1}))]),
                         'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(4, 6)},
                         'threshold': 0.25,
                         'kfold': 3}),

    ('NUMBER', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                       'doc_to_candidates': regexgen.RegexContextGenerator(10,
                                                                           10,
                                                                           # pylint: disable=line-too-long
                                                                           re.compile(r'(\(?\d[\d\-\.,\)]+)\s'),
                                                                           'NUMBER'),
                       'version': "1.0",
                       'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                       'pipeline': Pipeline([('union', FeatureUnion(
                           # pylint: disable=line-too-long
                           transformer_list=[('surround_transformer', transformerutils.SimpleTextTransformer())])),
                                             ('clf', SGDClassifier(loss='log',
                                                                   penalty='l2',
                                                                   n_iter=50,
                                                                   shuffle=True,
                                                                   random_state=42,
                                                                   class_weight={True: 3,
                                                                                 False: 1}))]),
                       'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(4, 6)},
                       'threshold': 0.25,
                       'kfold': 3}),

    ('PERCENT', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                        'doc_to_candidates': regexgen.RegexContextGenerator(15,
                                                                            5,
                                                                            re.compile(r'(\d+%)'),
                                                                            'PERCENT'),
                        'version': "1.0",
                        'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                        'pipeline': Pipeline([('union', FeatureUnion(
                            # pylint: disable=line-too-long
                            transformer_list=[('surround_transformer', transformerutils.SimpleTextTransformer())])),
                                              ('clf', SGDClassifier(loss='log',
                                                                    penalty='l2',
                                                                    n_iter=50,
                                                                    shuffle=True,
                                                                    random_state=42,
                                                                    class_weight={True: 3,
                                                                                  False: 1}))]),
                        'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(4, 6)},
                        'threshold': 0.25,
                        'kfold': 3}),

     ('PARAGRAPH', '1.0', {'doclist_to_antdoc_list': ebantdoc2.doclist_to_ebantdoc_list,
                           'doc_to_candidates': paragen.ParagraphGenerator('PARAGRAPH'),
                           'version': "1.0",
                           'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                           'pipeline': Pipeline([('union', FeatureUnion(
                           # pylint: disable=line-too-long
                           transformer_list=[('surround_transformer', transformerutils.SimpleTextTransformer())])),
                                              ('clf', SGDClassifier(loss='log',
                                                                    penalty='l2',
                                                                    n_iter=50,
                                                                    shuffle=True,
                                                                    random_state=42,
                                                                    class_weight={True: 3,
                                                                                  False: 1}))]),
                           'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(4, 6)},
                           'threshold': 0.25,
                           'kfold': 3}),
     ('SECTION', '1.0', {'doclist_to_antdoc_list': ebantdoc2.doclist_to_ebantdoc_list,
                         'doc_to_candidates': sectiongen.SectionGenerator('SECTION'),
                         'version': "1.0",
                         'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                         'pipeline': Pipeline([('union', FeatureUnion(
                         # pylint: disable=line-too-long
                         transformer_list=[('surround_transformer', transformerutils.SimpleTextTransformer())])),
                                              ('clf', SGDClassifier(loss='log',
                                                                    penalty='l2',
                                                                    n_iter=50,
                                                                    shuffle=True,
                                                                    random_state=42,
                                                                    class_weight={True: 3,
                                                                                  False: 1}))]),
                         'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(4, 6)},
                         'threshold': 0.25,
                         'kfold': 3})
]
'''
('l_tenant_notice', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                                'doc_to_candidates': addrgen.AddrContextGenerator(10, 2, 'ADDRESS'),
                                'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                                'version': "1.0",
                                'pipeline': Pipeline([
                                    ('union', FeatureUnion(
                                        transformer_list=[
                                            # pylint: disable=line-too-long
                                            ('surround_transformer', transformerutils.SimpleTextTransformer()),
                                            ('is_addr_line_transformer', transformerutils.AddrLineTransformer())
                                        ])),
                                    ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                          shuffle=True, random_state=42,
                                                          class_weight={True: 3, False: 1}))]),
                                'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(4, 6)},
                                'threshold': 0.25,
                                'kfold': 3}),
'''
RULE_ANNOTATOR_CONFIG_LIST = [
    ('effectivedate', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                              'doc_to_candidates': dategen.DateSpanGenerator(20, 20, 'DATE'),
                              'version': "1.0",
                              'rule_engine': dummyannotator.DummyAnnotator()}),
]

ML_ANNOTATOR_CONFIG_FROZEN_LIST = []  # type: List[Tuple[str, str, Dict]]
RULE_ANNOTATOR_CONFIG_FROZEN_LIST = []  # type: List[Tuple[str, str, Dict]]

def validate_annotator_config_keys(aconfig: Tuple[str, str, Dict]) -> bool:
    label, version, adict = aconfig
    is_valid = True
    for key, value in adict.items():
        if key not in set(['doclist_to_antdoc_list',
                           'doc_to_candidates',
                           'version',
                           'doc_postproc_list',
                           'pipeline',
                           'threshold',
                           'gridsearch_parameters',
                           'kfold',
                           'rule_engine']):
            logging.warning('invalid key, %s, in %s %s config',
                            key, label, version)
            is_valid = False
    return is_valid

def validate_annotators_config_keys(config_list: List[Tuple[str, str, Dict]]) -> bool:
    is_valid = True
    for aconfig in config_list:
        if not validate_annotator_config_keys(aconfig):
            is_valid = False
    return is_valid

validate_annotators_config_keys(ML_ANNOTATOR_CONFIG_LIST)
validate_annotators_config_keys(RULE_ANNOTATOR_CONFIG_LIST)
validate_annotators_config_keys(ML_ANNOTATOR_CONFIG_FROZEN_LIST)
validate_annotators_config_keys(RULE_ANNOTATOR_CONFIG_FROZEN_LIST)

def get_ml_annotator_config(label: str, version: Optional[str] = None) -> Dict:
    configx = get_annotator_config(label,
                                   version,
                                   ML_ANNOTATOR_CONFIG_LIST,
                                   ML_ANNOTATOR_CONFIG_FROZEN_LIST)
    if configx:
        _, _, prop = configx
        return prop
    return {}


def get_rule_annotator_config(label: str, version: Optional[str] = None) -> Dict:
    configx = get_annotator_config(label,
                                   version,
                                   RULE_ANNOTATOR_CONFIG_LIST,
                                   RULE_ANNOTATOR_CONFIG_FROZEN_LIST)
    if configx:
        _, _, prop = configx
        return prop
    return {}



def get_annotator_config(label: str,
                         version: Optional[str],
                         config_list: List[Tuple[str, str, Dict]],
                         config_frozen_list: List[Tuple[str, str, Dict]]) \
                         -> Optional[Tuple[str, str, Dict]]:
    best_annotator_config = None  # type: Optional[Tuple[str, str, Dict]]
    ver = StrictVersion(version)
    for candg_type, candg_ver, candg_property in itertools.chain(config_list,
                                                                 config_frozen_list):
        if candg_type == label:
            if version:
                if version == candg_ver:  # found the desired config
                    return (candg_type, candg_ver, candg_property)
            elif best_annotator_config:
                # pylint: disable=unpacking-non-sequence
                unused_label_, best_candg_ver, unused_prop = best_annotator_config
                best_ver = StrictVersion(best_candg_ver)
                if ver > best_ver:
                    best_annotator_config = (candg_type, candg_ver, candg_property)
            else:
                best_annotator_config = (candg_type, candg_ver, candg_property)

    return best_annotator_config
