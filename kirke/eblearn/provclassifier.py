#!/usr/bin/env python3

import logging

from kirke.eblearn import ebattrvec

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pylint: disable=C0301
# based on http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py


GLOBAL_THRESHOLD = 0.12

# The value in this provision_threshold_map is manually
# set by inspecting the result.  Using 0.06 in general
# produces too many false positives.
PROVISION_THRESHOLD_MAP = {'change_control': 0.42,
                           'confidentiality': 0.24,
                           'equitable_relief': 0.24,
                           'events_default': 0.18,
                           'sublicense': 0.24,
                           'survival': 0.24,
                           'termination': 0.36,
                           'l_alterations': 0.3}

PROVISION_ATTRLISTS_MAP = {'party': (ebattrvec.PARTY_BINARY_ATTR_LIST,
                                     ebattrvec.PARTY_NUMERIC_ATTR_LIST,
                                     ebattrvec.PARTY_CATEGORICAL_ATTR_LIST),
                           'default': (ebattrvec.DEFAULT_BINARY_ATTR_LIST,
                                       ebattrvec.DEFAULT_NUMERIC_ATTR_LIST,
                                       ebattrvec.DEFAULT_CATEGORICAL_ATTR_LIST)}

# pylint: disable=invalid-name
def get_transformer_attr_list_by_provision(provision: str):
    if PROVISION_ATTRLISTS_MAP.get(provision):
        return PROVISION_ATTRLISTS_MAP.get(provision)
    return PROVISION_ATTRLISTS_MAP.get('default')


def get_provision_threshold(provision: str):
    return PROVISION_THRESHOLD_MAP.get(provision, GLOBAL_THRESHOLD)

def adapt_pipeline_params(best_params):
    # params = copy.deepcopy(best_params)
    # # del the key because it has object (not-json)
    # params.pop('steps', None)
    # params.pop('clf', None)
    # params.pop('eb_transformer', None)

    result = {}
    for param_name, param_val in best_params.items():
        if param_name.startswith('clf__'):
            result[param_name[5:]] = param_val
        else:
            pass  # skip eb_transformer_* and others
    return result
