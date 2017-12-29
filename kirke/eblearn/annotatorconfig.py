
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from kirke.ebrules import addrannotator, dummyannotator, postprocess
from kirke.sampleutils import dategen, linegen, transformerutils
from kirke.utils import ebantdoc2, ebantdoc3

ml_annotator_config_map = \
    {'effectivedate': {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                       'docs_to_samples': dategen.DateSpanGenerator(20, 20),
                       'version': 1.0,
                       'pipeline': Pipeline([
                           ('surround_transformer', transformerutils.SurroundWordTransformer()),
                           ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                 shuffle=True, random_state=42,
                                                 class_weight={True: 3, False: 1}))]),
                       'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(3, 7)}},
     'l_tenant_notice': {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                         'docs_to_samples': linegen.LineSpanGenerator(20, 20),
                         'version': 1.0,
                         'pipeline': Pipeline([
                             ('surround_transformer', transformerutils.SurroundWordTransformer()),
                             ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                   shuffle=True, random_state=42,
                                                   class_weight={True: 3, False: 1}))]),
                         'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(3, 7)}}
    }

rule_annotator_config_map = \
    {'effectivedate': {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                       'docs_to_samples': dategen.DateSpanGenerator(20, 20),
                       'version': 1.0,
                       'rule_engine': dummyannotator.DummyAnnotator()},
     'l_tenant_notice': {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                         'docs_to_samples': linegen.LineSpanGenerator(20, 20),
                         'version': 1.0,
                         'rule_engine': addrannotator.AddressAnnotator(),
                         'post_process_list': [postprocess.AdjacentLineMerger()]}
    }


def get_ml_annotator_config(label: str):
    return ml_annotator_config_map.get(label)

def get_rule_annotator_config(label: str):
    return rule_annotator_config_map.get(label)
