import datetime
import json
import logging
import pprint
import time

from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, train_test_split

from kirke.eblearn import ebannotator, ebpostproc
from kirke.eblearn import annotatorconfig, lineannotator, ruleannotator, spanannotator
from kirke.utils import evalutils, splittrte, strutils, ebantdoc2, ebantdoc3
from kirke.eblearn import ebattrvec
from kirke.ebrules import titles

DEFAULT_CV = 5

# MIN_FULL_TRAINING_SIZE = 30
## this is the original val
# MIN_FULL_TRAINING_SIZE = 50
# MIN_FULL_TRAINING_SIZE = 400

# MIN_FULL_TRAINING_SIZE = 150
MIN_FULL_TRAINING_SIZE = 100



# Take all the data for training.
# Unless you know what you are doing, don't use this function, use
# train_eval_annotator() instead.
def _train_classifier(txt_fn_list, work_dir, model_file_name, eb_classifier):
    eb_classifier.train(txt_fn_list, work_dir, model_file_name)
    return eb_classifier


# Take 1/5 of the data out for testing
# Train on 4/5 of the data
# pylint: disable=R0915, R0913, R0914
def train_eval_annotator_with_trte(provision: str,
                                   work_dir: str,
                                   model_dir: str,
                                   model_file_name: str,
                                   eb_classifier,
                                   is_doc_structure=False) -> ebannotator.ProvisionAnnotator:
    logging.info("training_eval_annotator_with_trte(%s) called", provision)
    logging.info("    work_dir = %s", work_dir)
    logging.info("    model_dir = %s", model_dir)
    logging.info("    model_file_name = %s", model_file_name)

    train_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, provision)
    X_train = ebantdoc2.doclist_to_ebantdoc_list(train_doclist_fn,
                                                 work_dir,
                                                 is_doc_structure=is_doc_structure)
    eb_classifier.train_antdoc_list(X_train, work_dir, model_file_name)
    X_train = None  # free that memory

    test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
    X_test = ebantdoc2.doclist_to_ebantdoc_list(test_doclist_fn,
                                                work_dir,
                                                is_doc_structure=is_doc_structure)
    pred_status = eb_classifier.predict_and_evaluate(X_test, work_dir)

    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    ant_status = prov_annotator.test_antdoc_list(X_test)

    ant_status['provision'] = provision
    ant_status['pred_status'] = pred_status
    prov_annotator.eval_status = ant_status
    pprint.pprint(ant_status)

    model_status_fn = model_dir + '/' +  provision + ".status"
    strutils.dumps(json.dumps(ant_status), model_status_fn)

    with open('provision_model_stat.tsv', 'a') as pmout:
        pstatus = pred_status['pred_status']
        pcfmtx = pstatus['confusion_matrix']
        astatus = ant_status['ant_status']
        acfmtx = astatus['confusion_matrix']
        timestamp = int(time.time())
        aline = [datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                 str(timestamp),
                 provision,
                 pcfmtx['tp'], pcfmtx['fn'], pcfmtx['fp'], pcfmtx['tn'],
                 pred_status['best_params_']['alpha'],
                 pstatus['prec'], pstatus['recall'], pstatus['f1'],
                 acfmtx['tp'], acfmtx['fn'], acfmtx['fp'], acfmtx['tn'],
                 astatus['threshold'],
                 astatus['prec'], astatus['recall'], astatus['f1']]
        print('\t'.join([str(x) for x in aline]), file=pmout)

    return prov_annotator


# Take 1/5 of the data out for testing
# Train on 4/5 of the data
# pylint: disable=R0915, R0913, R0914
def train_eval_span_annotator_with_trte(label: str,
                                        work_dir: str,
                                        model_dir: str) -> spanannotator.SpanAnnotator:
    config = annotatorconfig.get_ml_annotator_config(label)
    model_file_name = '{}/{}_annotator.v{}.pkl'.format(model_dir,
                                                       label,
                                                       config['version'])

    span_annotator = spanannotator.SpanAnnotator(label,
                                                 version=config['version'],
                                                 doclist_to_antdoc_list=config['doclist_to_antdoc_list'],
                                                 docs_to_samples=config['docs_to_samples'],
                                                 sample_transformers=config.get('sample_transformers', []),
                                                 pipeline=config['pipeline'],
                                                 gridsearch_parameters=config['gridsearch_parameters'],
                                                 threshold=config.get('threshold', 0.5),
                                                 kfold=config.get('kfold', 3))

    logging.info("training_eval_span_annotator_with_trte(%s) called", label)
    logging.info("    work_dir = %s", work_dir)
    logging.info("    model_file_name = %s", model_file_name)

    train_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, label)
    train_antdoc_list = span_annotator.doclist_to_antdoc_list(train_doclist_fn,
                                                              work_dir,
                                                              is_doc_structure=False)

    samples, label_list, group_id_list = span_annotator.documents_to_samples(train_antdoc_list, label)

    logging.info("after span_annotator.documents_to_samples(), {}".format(strutils.to_pos_neg_count(label_list)))

    # span_annotator.estimator
    span_annotator.train_antdoc_list(samples,
                                     label_list,
                                     group_id_list,
                                     span_annotator.pipeline,
                                     span_annotator.gridsearch_parameters,
                                     work_dir)

    test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, label)
    test_antdoc_list = ebantdoc3.doclist_to_ebantdoc_list(test_doclist_fn,
                                                          work_dir,
                                                          is_doc_structure=False)

    span_annotator.pred_status = span_annotator.predict_and_evaluate(test_antdoc_list, work_dir)
    print("pred_status x24: {}".format(span_annotator.pred_status))
    span_annotator.ant_status = span_annotator.test_antdoc_list(test_antdoc_list)
    print("ant_status x24: {}".format(span_annotator.ant_status))

    span_annotator.save(model_file_name)
    span_annotator.print_eval_status(model_dir)

    return span_annotator


def eval_rule_annotator_with_trte(label,
                                  model_dir='dir-model',
                                  work_dir='dir-work',
                                  is_doc_structure=False,
                                  is_train_mode=False):
    config = annotatorconfig.get_rule_annotator_config(label)

    rule_annotator = ruleannotator.RuleAnnotator(label,
                                                 doclist_to_antdoc_list=config['doclist_to_antdoc_list'],
                                                 docs_to_samples=config['docs_to_samples'],
                                                 rule_engine=config['rule_engine'],
                                                 post_process=config.get('post_process', []))

    logging.info("eval_rule_annotator_with_trte(%s) called", label)

    # Normally, we compare the test results
    # During development, use is_train_mode to peek at the data for improvements
    if is_train_mode:
        test_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, label)
    else:
        test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, label)

    test_antdoc_list = rule_annotator.doclist_to_antdoc_list(test_doclist_fn,
                                                             work_dir,
                                                             is_doc_structure=False)

    rule_annotator.ant_status = rule_annotator.test_antdoc_list(test_antdoc_list)
    print("ant_status x24: {}".format(rule_annotator.ant_status))

    # rule_annotator.save(model_file_name)
    rule_annotator.print_eval_status(model_dir)

    return rule_annotator


def eval_annotator(txt_fn_list, work_dir, model_file_name):
    eb_classifier = joblib.load(model_file_name)
    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    # update the hashmap of annotators
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    provision_status_map['ant_status'] = prov_annotator.test_antdoc_list(ebantdoc_list)

    pprint.pprint(provision_status_map)


def eval_ml_rule_annotator(txt_fn_list, work_dir, model_file_name):
    eb_classifier = joblib.load(model_file_name)
    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    # update the hashmap of annotators
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    provision_status_map['ant_status'] = prov_annotator.test_antdoc_list(ebantdoc_list)

    pprint.pprint(provision_status_map)


def eval_line_annotator_with_trte(provision,
                                  model_dir='dir-scut-model',
                                  work_dir='dir-work',
                                  is_doc_structure=False):

    test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
    ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(test_doclist_fn,
                                                       work_dir=work_dir,
                                                       is_doc_structure=is_doc_structure)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    provision_status_map = {'provision': provision}
    # update the hashmap of annotators
    prov_annotator = lineannotator.LineAnnotator('title', titles.TitleAnnotator('title'))
    # we need ebantdoc_list because it has the annotations
    provision_status_map['ant_status'] = prov_annotator.test_antdoc_list(ebantdoc_list)

    pprint.pprint(provision_status_map)


def eval_classifier(txt_fn_list, work_dir, model_file_name):
    eb_classifier = joblib.load(model_file_name)
    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    pprint.pprint(provision_status_map)


# utility function
# this is mainly used for the outer testing (real hold out)
def calc_scut_predict_evaluate(scut_classifier, attrvec_list, y_pred, y_te):
    logging.info('calc_scut_predict_evaluate()...')

    sent_st_list = [attrvec.bag_of_words for attrvec in attrvec_list]
    overrides = ebpostproc.gen_provision_overrides(scut_classifier.provision, sent_st_list)

    threshold = scut_classifier.threshold

    scut_classifier.pred_status['classifer_type'] = 'scutclassifier'
    scut_classifier.pred_status['pred_status'] = evalutils.calc_pred_status_with_prob(y_pred, y_te)
    scut_classifier.pred_status['override_status'] = (
        # evalutils.calc_pred_override_status(y_pred, y_te, overrides))
        evalutils.calc_prob_override_status(y_pred, y_te, threshold, overrides))
    scut_classifier.pred_status['best_params_'] = scut_classifier.best_parameters

    return scut_classifier.pred_status
