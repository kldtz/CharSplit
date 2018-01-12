import json
import logging
from pprint import pprint

from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import GroupKFold

from kirke.eblearn import ebannotator, ebpostproc, lineannotator
from kirke.utils import evalutils, splittrte, strutils, ebantdoc2
from kirke.eblearn import ebattrvec
from kirke.ebrules import titles, parties, dates

from datetime import datetime
import time

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
def train_eval_annotator(provision, 
                         txt_fn_list,
                         work_dir, 
                         model_dir, 
                         model_file_name, 
                         eb_classifier,
                         is_doc_structure=False,
                         custom_training_mode=False,
                         doc_lang="en"):
    logging.info("training_eval_annotator(%s) called", provision)
    logging.info("    txt_fn_list = %s", txt_fn_list)
    logging.info("    work_dir = %s", work_dir)
    logging.info("    model_dir = %s", model_dir)
    logging.info("    model_file_name = %s", model_file_name)
    logging.info("    is_doc_structure= %s", is_doc_structure)
    # is_combine_line should be file dependent, PDF than False
    # HTML is True.
    eb_traindoc_list = ebantdoc2.doclist_to_traindoc_list(txt_fn_list,
                                                          work_dir,
                                                          is_bespoke_mode=custom_training_mode,
                                                          is_doc_structure=is_doc_structure,
                                                          doc_lang=doc_lang)

    attrvec_list = []
    group_id_list = []
    for group_id, eb_traindoc in enumerate(eb_traindoc_list):
        tmp_attrvec_list = eb_traindoc.get_attrvec_list()
        attrvec_list.extend(tmp_attrvec_list)
        group_id_list.extend([group_id] * len(tmp_attrvec_list))

    num_pos_label, num_neg_label = 0, 0
    
    for attrvec in attrvec_list: 
        if provision in attrvec.labels:
            num_pos_label += 1
            # print("\npositive training for {}".format(provision))
            # print("    [[{}]]".format(attrvec.bag_of_words))
        else:
            num_neg_label += 1
    # pylint: disable=C0103
    X = eb_traindoc_list
    y = [provision in eb_traindoc.get_provision_set()
         for eb_traindoc in eb_traindoc_list]

    num_doc_pos, num_doc_neg = 0, 0
    for yval in y:
        if yval:
            num_doc_pos += 1
        else:
            num_doc_neg += 1
    print("provision: {}, pos= {}, neg= {}".format(provision, num_doc_pos, num_doc_neg))
    # TODO, jshaw, hack, such as for sechead
    if num_doc_neg < 2:
        y[0] = 0
        y[1] = 0

    # only in custom training mode and the positive training instances are too few
    # only train, no independent testing
    if custom_training_mode and num_pos_label < MIN_FULL_TRAINING_SIZE:
        logging.info("training with %d instances, no test (<%d) .  num_pos= %d, num_neg= %d",
                     len(attrvec_list), MIN_FULL_TRAINING_SIZE, num_pos_label, num_neg_label)
        X_train = X
        # y_train = y
        train_doclist_fn = "{}/{}_{}_train_doclist.txt".format(model_dir, provision, doc_lang)
        splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)
        # We use cv_scores to generate a more detailed status resport
        # than just a score number for cross-validation folds.
        _, cv_scores = eb_classifier.train_antdoc_list(X_train, work_dir, model_file_name)

        print("eb_classifier.best_parameters")
        best_parameters = eb_classifier.best_parameters
        pprint(best_parameters)

        y_label_list = [provision in attrvec.labels for attrvec in attrvec_list]

        # this setup eb_classifier.status
        pred_status = calc_scut_predict_evaluate(eb_classifier,
                                                 attrvec_list,
                                                 cv_scores,
                                                 y_label_list)

        # make the classifier into an annotator
        prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)

        ant_status = {'provision' : provision,
                      'pred_status' : pred_status}
        prov_annotator.eval_status = ant_status
        pprint(ant_status)

        model_status_fn = model_dir + '/' +  provision + ".status"
        strutils.dumps(json.dumps(ant_status), model_status_fn)
        return prov_annotator

    logging.info("training with %d instances, num_pos= %d, num_neg= %d",
                 len(attrvec_list), num_pos_label, num_neg_label)

    if custom_training_mode:
        test_size = 0.25
    else:
        test_size = 0.2

    # we have enough positive training instances, so we do testing
    X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size,
                                             random_state=42, stratify=y)

    train_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, provision)
    splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)
    test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
    splittrte.save_antdoc_fn_list(X_test, test_doclist_fn)

    eb_classifier.train_antdoc_list(X_train, work_dir, model_file_name)
    pred_status = eb_classifier.predict_and_evaluate(X_test, work_dir)

    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)

    # X_test is now traindoc, not ebantdoc.  The testing docs are loaded one by one
    # using generator, instead of all loaded at once.
    X_test_antdoc_list = ebantdoc2.traindoc_list_to_antdoc_list(X_test, work_dir)
    ant_status = prov_annotator.test_antdoc_list(X_test_antdoc_list)

    ant_status['provision'] = provision
    ant_status['pred_status'] = pred_status
    prov_annotator.eval_status = ant_status
    pprint(ant_status)

    model_status_fn = model_dir + '/' +  provision + ".status"
    strutils.dumps(json.dumps(ant_status), model_status_fn)

    with open('provision_model_stat.tsv', 'a') as pmout:
        pstatus = pred_status['pred_status']
        pcfmtx = pstatus['confusion_matrix']
        astatus = ant_status['ant_status']
        acfmtx = astatus['confusion_matrix']
        timestamp = int(time.time())
        aline = [datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
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
def train_eval_annotator_with_trte(provision,
                                   work_dir, model_dir, model_file_name, eb_classifier,
                                   is_doc_structure=False):
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
    pprint(ant_status)

    model_status_fn = model_dir + '/' +  provision + ".status"
    strutils.dumps(json.dumps(ant_status), model_status_fn)

    with open('provision_model_stat.tsv', 'a') as pmout:
        pstatus = pred_status['pred_status']
        pcfmtx = pstatus['confusion_matrix']
        astatus = ant_status['ant_status']
        acfmtx = astatus['confusion_matrix']
        timestamp = int(time.time())
        aline = [datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
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

    pprint(provision_status_map)


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

    pprint(provision_status_map)


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
    if provision == 'title':
        prov_annotator = lineannotator.LineAnnotator('title', titles.TitleAnnotator('title'))
    elif provision == 'party':
        prov_annotator = lineannotator.LineAnnotator('party', parties.PartyAnnotator('party'))
    elif provision == 'date':
        prov_annotator = lineannotator.LineAnnotator('date', dates.DateAnnotator('date'))
    # we need ebantdoc_list because it has the annotations
    provision_status_map['ant_status'] = prov_annotator.test_antdoc_list(ebantdoc_list)

    pprint(provision_status_map)


def eval_classifier(txt_fn_list, work_dir, model_file_name):
    eb_classifier = joblib.load(model_file_name)
    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    pprint(provision_status_map)


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
