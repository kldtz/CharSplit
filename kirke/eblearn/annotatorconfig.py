
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

from kirke.ebrules import addrannotator, dummyannotator, postprocess
from kirke.sampleutils import dollargen, addrgen, dategen, linegen, transformerutils
from kirke.utils import ebantdoc2, ebantdoc3

ml_annotator_config_map = \
    {'effectivedate': {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                       'docs_to_samples': dategen.DateSpanGenerator(20, 20),
                       'version': "1.0",
                       'pipeline': Pipeline([
                           ('surround_transformer', transformerutils.SurroundWordTransformer()),
                           ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                 shuffle=True, random_state=42,
                                                 class_weight={True: 3, False: 1}))]),
                       'threshold': 0.2,
                       'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(3, 7)}},
     'l_tenant_notice': {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                         'docs_to_samples': addrgen.AddrContextGenerator(2,0),
                         'post_process_list': 'l_tenant_notice',
                         'sample_transformers': [addrannotator.SampleAddAddrLineProb()],
                         'version': "1.0",
                         'pipeline': Pipeline([
                             ('union', FeatureUnion(
                                 transformer_list=[
                                     ('surround_transformer', transformerutils.SimpleTextTransformer()),
                                     ('is_addr_line_transformer', transformerutils.AddrLineTransformer())
                                 ])),
                             ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                   shuffle=True, random_state=42,
                                                   class_weight={True: 3, False: 1}))]),
                         'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(4, 6)},
                         'threshold': 0.25,
                         'kfold': 2},
     'purchase_price': {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                        'docs_to_samples': dollargen.DollarContextGenerator(2,0),
                        'post_process_list': 'purchase_price',
                        'version': "1.0",
                        'pipeline': Pipeline([ 
                            ('union', FeatureUnion(
                            transformer_list=[
                                ('surround_transformer', transformerutils.SimpleTextTransformer())
                            ])),
                        ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                   shuffle=True, random_state=42,
                                                   class_weight={True: 3, False: 1}))]),
                         'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(4, 6)},
                         'threshold': 0.25,
                         'kfold': 2}
    }

rule_annotator_config_map = \
    {'effectivedate': {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                       'docs_to_samples': dategen.DateSpanGenerator(20, 20),
                       'version': "1.0",
                       'rule_engine': dummyannotator.DummyAnnotator()},
     'l_tenant_notice': {'doclist_to_antdoc_list': ebantdoc3.doclist_to_ebantdoc_list,
                         'docs_to_samples': linegen.LineSpanGenerator(20, 20),
                         'version': "1.0",
                         'rule_engine': addrannotator.AddressAnnotator(),
                         'post_process_list': [postprocess.AdjacentLineMerger()]}
    }


def get_ml_annotator_config(label: str):
    return ml_annotator_config_map.get(label)

def get_rule_annotator_config(label: str):
    return rule_annotator_config_map.get(label)
