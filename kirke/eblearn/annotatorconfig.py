# pylint: disable=no-name-in-module, import-error
from distutils.version import StrictVersion
import itertools
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from kirke.ebrules import addrannotator, dummyannotator, postprocess
from kirke.sampleutils import regexgen, addrgen, dategen, linegen, transformerutils
from kirke.utils import ebantdoc3

# pylint: disable=pointless-string-statement
"""
For the same candidate type', such as 1.Currency or 1.Number,
there can be different versions, but the candidate type name should be
kept the same.  There should be a "frozen" list of those config so
that people are aware not to touch any of the classes mentioned in
those configs.  There will be a default config list, in which people
can modify at will.  The one with the highest version will be used if
no model for that candidate generation type can be found. (edited)

'provision' is outside of annotationconfig.py.
"""


ML_ANNOTATOR_CONFIG_LIST = [
    ('DATE', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                              'docs_to_samples': dategen.DateSpanGenerator(10,10, 'DATE'),
                              'version': "1.0",
                              'pipeline': Pipeline([
                                  # pylint: disable=line-too-long
                                  ('surround_transformer', transformerutils.SurroundWordTransformer()),
                                  ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                        shuffle=True, random_state=42,
                                                        class_weight={True: 3, False: 1}))]),
                              'threshold': 0.2,
                              'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(3, 7)}}),

    ('ADDRESS', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                                'docs_to_samples': addrgen.AddrContextGenerator(10, 10, 'ADDRESS'),
                                'post_process_list': 'l_tenant_notice',
                                'sample_transformers': [addrannotator.SampleAddAddrLineProb()],
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
                                'kfold': 2}),

    ('l_tenant_notice', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                                'docs_to_samples': addrgen.AddrContextGenerator(10, 2, 'ADDRESS'),
                                'post_process_list': 'l_tenant_notice',
                                'sample_transformers': [addrannotator.SampleAddAddrLineProb()],
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
                                'kfold': 2}),

    ('CURRENCY', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                           'docs_to_samples': regexgen.RegexContextGenerator(20,
                                                                             5,
                                                                             # pylint: disable=line-too-long
                                                                             re.compile(r'([\$€₹£¥](\d{1,3},?)+([,\.]\d\d)?)[€円]?'),
                                                                             'CURRENCY'),
                           'version': "1.0",
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
                           'kfold': 2}),

    ('NUMBER', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                         'docs_to_samples': regexgen.RegexContextGenerator(10,
                                                                           10,
                                                                           # pylint: disable=line-too-long
                                                                           re.compile(r'\(?(\d[\d\-\.,]+)[\s\)]'),
                                                                           'NUMBER'),
                         'version': "1.0",
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
                         'kfold': 2}),

    ('PERCENT', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                          'docs_to_samples': regexgen.RegexContextGenerator(15,
                                                                            5,
                                                                            re.compile(r'(\d+%)'),
                                                                            'PERCENT'),
                          'version': "1.0",
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
                          'kfold': 2})
]

RULE_ANNOTATOR_CONFIG_LIST = [
    ('effectivedate', '1.0', {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                              'docs_to_samples': dategen.DateSpanGenerator(20, 20, 'DATE'),
                              'version': "1.0",
                              'rule_engine': dummyannotator.DummyAnnotator()}),
]

ML_ANNOTATOR_CONFIG_FROZEN_LIST = []  # type: List[Tuple[str, str, Dict]]
RULE_ANNOTATOR_CONFIG_FROZEN_LIST = []  # type: List[Tuple[str, str, Dict]]


def get_ml_annotator_config(label: str, version: Optional[str] = None) -> Optional[Dict]:
    configx = get_annotator_config(label,
                                   version,
                                   ML_ANNOTATOR_CONFIG_LIST,
                                   ML_ANNOTATOR_CONFIG_FROZEN_LIST)
    if configx:
        _, _, prop = configx
        return prop
    return None


def get_rule_annotator_config(label: str, version: Optional[str] = None) -> Optional[Dict]:
    configx = get_annotator_config(label,
                                   version,
                                   RULE_ANNOTATOR_CONFIG_LIST,
                                   RULE_ANNOTATOR_CONFIG_FROZEN_LIST)
    if configx:
        _, _, prop = configx
        return prop
    return None



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
