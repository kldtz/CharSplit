# pylint: disable=no-name-in-module, import-error
from distutils.version import StrictVersion
import itertools
import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from kirke.ebrules import dates, dummyannotator
from kirke.sampleutils import postproc
from kirke.sampleutils import addrgen, dategen, idnumgen, paragen
from kirke.sampleutils import regexgen, sentencegen, tablegen
from kirke.sampleutils import transformerutils
from kirke.utils import ebantdoc4

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Different provisions can use the same candidate type, such as Currency
# or Number.  The specifications of 'provision' are outside of annotatorconfig.py.
#
# There are "frozen" lists of those config so that developer are aware not to touch
# any of the classes mentioned in the frozen config lists.

# pylint: disable=line-too-long
CURRENCY_PAT = re.compile(r'(((USD|GBP|JPY|[\$€₹£¥円]) *(\d{1,3}[,\.]?)+([,\.]\d{,2})?( *[tTbBmM]illion| *[tT]housand| *[TMB])?)|'
                          r'((\d{1,3},?)+([,\.]\d{,2})? *([tTbBmM]illion|[tT]housand|[TMB])? *(USD|EUR|INR|GBP|CNY|JPY|[dD]ollars?|[eE]uros?|[rR]upees?|[pP]ounds?|[yY]en|[\$€₹£¥円])))')

# must pick gruop 2 instead of group 1
# pylint: disable=line-too-long
NUMBER_PAT = re.compile(r'(^|\s)\(?(-?([0-9]+([,\.][0-9]{3})*[,\.]?[0-9]*|[,\.][0-9]+))\)?[,\.:;]?(\s|$)')
# pylint: disable=line-too-long
PERCENT_PAT = re.compile(r'(^|\s)\(?(-?([0-9]+([,\.][0-9]{3})*[,\.]?[0-9]*|\.[0-9]+)\s*(%|percent))\)?[,\.:;]?(\s|$)', re.I)


ML_ANNOTATOR_CONFIG_LIST = [
    ('SENTENCE', '1.0', {'doclist_to_antdoc_list': ebantdoc4.doclist_to_ebantdoc_list,
                         'is_use_corenlp': True,
                         'text_type': 'nlp_text',
                         'doc_to_candidates': [sentencegen.SentenceGenerator('SENTENCE')],
                         'version': "1.0",
                         'doc_postproc_list': [postproc.SentDefaultPostProcessing(0.24)],
                         'pipeline': Pipeline([
                             ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                   shuffle=True, random_state=42,
                                                   class_weight={True: 3, False: 1}))]),
                         'threshold': 0.24,
                         'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(3, 8)}}),

    ('DATE', '1.0', {'doclist_to_antdoc_list': ebantdoc4.doclist_to_ebantdoc_list,
                     'is_use_corenlp': False,
                     'doc_to_candidates': [dategen.DateSpanGenerator(30, 30, 'DATE')],
                     'version': "1.0",
                     'doc_postproc_list': [dates.DateNormalizer(),
                                           postproc.SpanDefaultPostProcessing()],
                     'pipeline': Pipeline([
                         # pylint: disable=line-too-long
                         ('surround_transformer', transformerutils.SurroundWordTransformer()),
                         ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                               shuffle=True, random_state=42,
                                               class_weight={True: 3, False: 1}))]),
                     'threshold': 0.5,
                     'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(3, 7)}}),

    ('ADDRESS', '1.0', {'doclist_to_antdoc_list': ebantdoc4.doclist_to_ebantdoc_list,
                        'is_use_corenlp': False,
                        'doc_to_candidates': [addrgen.AddrContextGenerator(30, 30, 'ADDRESS')],
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
                        'threshold': 0.35,
                        'kfold': 3}),

    ('CURRENCY', '1.0', {'doclist_to_antdoc_list': ebantdoc4.doclist_to_ebantdoc_list,
                         'is_use_corenlp': False,
                         'doc_to_candidates':
                         [regexgen.RegexContextGenerator(20,
                                                         5,
                                                         CURRENCY_PAT,
                                                         'CURRENCY')],
                         'version': "1.0",
                         'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                         'pipeline': Pipeline([
                             ('union', FeatureUnion(
                                 transformer_list=[('surround_transformer', transformerutils.SimpleTextTransformer()),
                                                  ])),
                             ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                   shuffle=True, random_state=42,
                                                   class_weight={True: 3, False: 1}))]),
                         'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(4, 6)},
                         'threshold': 0.25,
                         'kfold': 3}),

    ('NUMBER', '1.0', {'doclist_to_antdoc_list': ebantdoc4.doclist_to_ebantdoc_list,
                       'is_use_corenlp': False,
                       'doc_to_candidates': [regexgen.RegexContextGenerator(10,
                                                                            10,
                                                                            NUMBER_PAT,
                                                                            'NUMBER',
                                                                            group_num=2)],
                       'version': "1.0",
                       'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                       'pipeline': Pipeline([('union', FeatureUnion(
                           # pylint: disable=line-too-long
                           transformer_list=[('surround_transformer', transformerutils.SimpleTextTransformer()),
                                            ])),
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

    ('PERCENT', '1.0', {'doclist_to_antdoc_list': ebantdoc4.doclist_to_ebantdoc_list,
                        'is_use_corenlp': False,
                        'doc_to_candidates': \
                        [regexgen.RegexContextGenerator(15,
                                                        5,
                                                        PERCENT_PAT,
                                                        'PERCENT',
                                                        group_num=2)],
                        'version': "1.0",
                        'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                        'pipeline': Pipeline([('union', FeatureUnion(
                            # pylint: disable=line-too-long
                            transformer_list=[('surround_transformer', transformerutils.SimpleTextTransformer()),
                                             ])),
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

    ('IDNUM', '1.0', {'doclist_to_antdoc_list': ebantdoc4.doclist_to_ebantdoc_list,
                      'is_use_corenlp': False,
                      'doc_to_candidates':
                      [idnumgen.IdNumContextGenerator(3,
                                                      3,
                                                      # the first part is to handle phone numbers
                                                      re.compile(r'(\+ \d[^\s]*|[^\s]*\d[^\s]*)'),
                                                      'IDNUM',
                                                      is_join=True,
                                                      length_min=2)],
                      'version': "1.0",
                      'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                      'pipeline': Pipeline([('union', FeatureUnion(
                          # pylint: disable=line-too-long
                          transformer_list=[('surround_transformer', transformerutils.SimpleTextTransformer()),
                                            ('char_transformer', transformerutils.CharacterTransformer())
                                           ])),
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

    ('PARAGRAPH', '1.0', {'doclist_to_antdoc_list': ebantdoc4.doclist_to_ebantdoc_list,
                          'is_use_corenlp': False,
                          'text_type': 'nlp_text',
                          'doc_to_candidates': [paragen.ParagraphGenerator('PARAGRAPH')],
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

    ('TABLE', '1.0', {'doclist_to_antdoc_list': ebantdoc4.doclist_to_ebantdoc_list,
                      'is_use_corenlp': True,
                      'is_doc_structure': True,
                      'doc_to_candidates': [tablegen.TableGenerator('TABLE')],
                      'version': "1.0",
                      'doc_postproc_list': [postproc.TablePostProcessing()],
                      'pipeline': Pipeline([('union', FeatureUnion(
                          # pylint: disable=line-too-long
                          transformer_list=[('table_transformer',
                                             transformerutils.TableTextTransformer())])),
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



RULE_ANNOTATOR_CONFIG_LIST = [
    ('effectivedate', '1.0', {'doclist_to_antdoc_list': ebantdoc4.doclist_to_ebantdoc_list,
                              'is_use_corenlp': False,
                              'doc_to_candidates': dategen.DateSpanGenerator(20, 20, 'DATE'),
                              'version': "1.0",
                              'rule_engine': dummyannotator.DummyAnnotator()}),
]


ML_ANNOTATOR_CONFIG_FROZEN_LIST = []  # type: List[Tuple[str, str, Dict]]
RULE_ANNOTATOR_CONFIG_FROZEN_LIST = []  # type: List[Tuple[str, str, Dict]]

def validate_annotator_config_keys(aconfig: Tuple[str, str, Dict]) -> bool:
    label, version, adict = aconfig
    is_valid = True
    for key, unused_value in adict.items():
        if key not in set(['doclist_to_antdoc_list',
                           'is_use_corenlp',
                           'is_doc_structure',
                           'doc_to_candidates',
                           'version',
                           'doc_postproc_list',
                           'pipeline',
                           'threshold',
                           'gridsearch_parameters',
                           'kfold',
                           'text_type',
                           'rule_engine']):
            logger.warning('invalid key, %s, in %s %s config',
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

def get_ml_annotator_config(label_list: List[str], version: Optional[str] = None) -> Dict:
    if len(label_list) == 1:
        label = label_list[0]
        configx = get_annotator_config(label,
                                       version,
                                       ML_ANNOTATOR_CONFIG_LIST,
                                       ML_ANNOTATOR_CONFIG_FROZEN_LIST)
        if configx:
            _, _, prop = configx
            return prop
        return {}

    generic_prop = {'doclist_to_antdoc_list': ebantdoc4.doclist_to_ebantdoc_list,
                    'version': "1.0",
                    'is_use_corenlp': False,
                    'doc_to_candidates': [],
                    'doc_postproc_list': [postproc.SpanDefaultPostProcessing()],
                    'pipeline': Pipeline([
                        ('union', FeatureUnion(
                            transformer_list=[('surround_transformer', transformerutils.SimpleTextTransformer()),
                                              ('char_transformer', transformerutils.CharacterTransformer())
                                             ])),
                        ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                              shuffle=True, random_state=42,
                                              class_weight={True: 3, False: 1}))]),
                    'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(4, 6)},
                    'threshold': 0.25,
                    'kfold': 3}
    for label in label_list:
        configx = get_annotator_config(label,
                                       version,
                                       ML_ANNOTATOR_CONFIG_LIST,
                                       ML_ANNOTATOR_CONFIG_FROZEN_LIST)
        if configx:
            _, _, prop = configx
            generic_prop['doc_to_candidates'].extend(prop['doc_to_candidates'])
        else:
            logger.error('Invalid Candidate Type: %s', label)
            exit(1)
    return generic_prop


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


def get_all_candidate_types():
    return set([x[0] for x in ML_ANNOTATOR_CONFIG_LIST])
