
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from kirke.sampleutils import dategen
from kirke.sampleutils import transformerutils

annotator_config_map = \
    {'effectivedate': {'docs_to_samples': dategen.DateSpanGenerator(20, 20),
                       'pipeline': Pipeline([
                           ('surround_transformer', transformerutils.SurroundWordTransformer()),
                           ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=50,
                                                 shuffle=True, random_state=42,
                                                 class_weight={True: 3, False: 1}))]),
                       'gridsearch_parameters': {'clf__alpha': 10.0 ** -np.arange(3, 7)}}}

def get_annotator_config(label: str):
    return annotator_config_map.get(label)
    

