from collections import defaultdict
from datetime import datetime
import json
import logging
import os
from pprint import pprint
import time
from typing import Dict, List, Optional, Tuple

from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import GroupKFold

from kirke.eblearn import ebannotator, ebpostproc, lineannotator
from kirke.utils import ebantdoc2, evalutils, splittrte, strutils, txtreader
from kirke.eblearn import ebattrvec
from kirke.ebrules import titles, parties, dates

DEFAULT_CV = 3

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

def log_model_eval_status(ant_status):
    with open('provision_model_stat.tsv', 'a') as pmout:
        pstatus = ant_status['pred_status']['pred_status']
        pcfmtx = pstatus['confusion_matrix']
        astatus = ant_status['ant_status']
        acfmtx = astatus['confusion_matrix']
        timestamp = int(time.time())
        provision = ant_status['provision']
        aline = [datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                 str(timestamp),
                 provision,
                 pcfmtx['tp'], pcfmtx['fn'], pcfmtx['fp'], pcfmtx['tn'],
                 ant_status['pred_status']['best_params_']['alpha'],
                 pstatus['prec'], pstatus['recall'], pstatus['f1'],
                 acfmtx['tp'], acfmtx['fn'], acfmtx['fp'], acfmtx['tn'],
                 astatus['threshold'],
                 astatus['prec'], astatus['recall'], astatus['f1']]
        print('\t'.join([str(x) for x in aline]), file=pmout)


def log_custom_model_eval_status(ant_status):
    with open('provision_model_stat.tsv', 'a') as pmout:
        pstatus = ant_status['ant_status']  # only ant status is available for custom training
        pcfmtx = pstatus['confusion_matrix']
        astatus = ant_status['ant_status']
        acfmtx = astatus['confusion_matrix']
        timestamp = int(time.time())
        provision = ant_status['provision']
        aline = [datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                 str(timestamp),
                 provision,
                 pcfmtx['tp'], pcfmtx['fn'], pcfmtx['fp'], pcfmtx['tn'],
                 # pred_status['best_params_']['alpha'],
                 -9999.99,
                 pstatus['prec'], pstatus['recall'], pstatus['f1'],
                 acfmtx['tp'], acfmtx['fn'], acfmtx['fp'], acfmtx['tn'],
                 astatus['threshold'],
                 astatus['prec'], astatus['recall'], astatus['f1']]
        print('\t'.join([str(x) for x in aline]), file=pmout)


def cv_train_at_annotation_level(provision,
                                 x_traindoc_list,
                                 bool_list,
                                 eb_classifier_orig,
                                 model_file_name: str,
                                 model_num: int,
                                 model_dir: str,
                                 work_dir: str):
    # we do 3-fold cross validation, as the big set for custom training
    # test_size = 0.33
    # this will be looped mutliple times, so a list, not a generator
    x_antdoc_list = list(ebantdoc2.traindoc_list_to_antdoc_list(x_traindoc_list, work_dir))

    num_fold = DEFAULT_CV
    # distribute positives to all buckets
    pos_list, neg_list = [], []
    for x_antdoc, label in zip(x_antdoc_list, bool_list):
        if label:
            pos_list.append((x_antdoc, label))
        else:
            neg_list.append((x_antdoc, label))
    pos_list.extend(neg_list)
    bucket_x_map = defaultdict(list)
    for count, (x_antdoc, label) in enumerate(pos_list):
        # bucket_x_map[count % num_fold].append((x_antdoc, label))
        bucket_x_map[count % num_fold].append(x_antdoc)

    #for bnum, alist in bucket_x_map.items():
    #    print("-----")
    #    for ebantdoc, y in alist:
    #        print("{}\t{}\t{}".format(bnum, ebantdoc.file_id, y))

    log_list = {}
    cv_ant_status_list = []
    for bucket_num in range(num_fold):  # cross train each bucket

        train_buckets = []
        test_bucket = None
        for bnum, bucket_x in bucket_x_map.items():
            if bnum != bucket_num:
                train_buckets.extend(bucket_x)
            else:  # bnum == bucket_num
                test_bucket = bucket_x

        cv_eb_classifier = eb_classifier_orig.make_bare_copy()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        cv_eb_classifier_fn = '/tmp/{}-{}-{}'.format(provision, timestr, bucket_num)
        cv_eb_classifier.train_antdoc_list(train_buckets,
                                           work_dir,
                                           cv_eb_classifier_fn)
        cv_prov_annotator = ebannotator.ProvisionAnnotator(cv_eb_classifier, work_dir)

        cv_ant_status, cv_log_json = cv_prov_annotator.test_antdoc_list(test_bucket)
        # print("cv ant_status, bucket_num = {}:".format(bucket_num))
        # print(cv_ant_status)

        log_list.update(cv_log_json)
        cv_ant_status_list.append(cv_ant_status)

    # now build the annotator using ALL training data
    eb_classifier = eb_classifier_orig.make_bare_copy()
    eb_classifier.train_antdoc_list(x_antdoc_list,
                                    work_dir,
                                    model_file_name)
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    log_json = log_list
    merged_ant_status = evalutils.aggregate_ant_status_list(cv_ant_status_list)['ant_status']

    ant_status = {'provision': provision}
    ant_status['ant_status'] = merged_ant_status
    ant_status['pred_status'] = {'pred_status': merged_ant_status}  # we are going to fake it for now
    prov_annotator.eval_status = ant_status
    pprint(ant_status)

    model_status_fn = '{}/{}.{}.status'.format(model_dir,
                                               provision,
                                               model_num)
    strutils.dumps(json.dumps(ant_status), model_status_fn)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    result_fn = model_dir + '/' + provision + "-ant_result-" + timestr + ".json"
    logging.info('wrote result file at: %s', result_fn)
    strutils.dumps(json.dumps(log_json), result_fn)

    log_custom_model_eval_status({'provision': provision,
                                  'ant_status': merged_ant_status})

    return prov_annotator, log_list


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
                         doc_lang="en",
                         model_num: Optional[int] = None) \
                         -> Tuple[ebannotator.ProvisionAnnotator, Dict[str, Dict]]:
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
    num_pos_ant = 0
    for group_id, eb_traindoc in enumerate(eb_traindoc_list):
        tmp_attrvec_list = eb_traindoc.get_attrvec_list()
        attrvec_list.extend(tmp_attrvec_list)
        group_id_list.extend([group_id] * len(tmp_attrvec_list))

        human_ant_list = eb_traindoc.prov_annotation_list
        for human_ant in human_ant_list:
            if provision == human_ant.label:
                num_pos_ant += 1
    # based on human annotations only, we don't know the num_neg_ant
    logging.info("num_pos_ant: %d", num_pos_ant)

    # these are for sentences
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
    logging.info("provision: %s, num_doc_pos= %d, num_doc_neg= %d", provision, num_doc_pos, num_doc_neg)
    # TODO, jshaw, hack, such as for sechead
    if num_doc_neg < 2:
        y[0] = 0
        y[1] = 0

    # only in custom training mode and the positive training instances are too few
    # only train, no independent testing
    # corss validation is applied to all Bespoke training
    if custom_training_mode:  #  and num_pos_label < MIN_FULL_TRAINING_SIZE:
        logging.info("training with %d instances, no test (<%d) .  num_inst_pos= %d, num_inst_neg= %d",
                     len(attrvec_list), MIN_FULL_TRAINING_SIZE, num_pos_label, num_neg_label)
        X_train = X
        train_doclist_fn = "{}/{}_{}_train_doclist.txt".format(model_dir, provision, doc_lang)
        splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)

        prov_annotator2, combined_log_json = \
            cv_train_at_annotation_level(provision,
                                         X_train,
                                         y,
                                         eb_classifier,
                                         model_file_name,
                                         model_num,
                                         model_dir,
                                         work_dir)

        return prov_annotator2, combined_log_json

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
    ant_status, log_json = prov_annotator.test_antdoc_list(X_test_antdoc_list)

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
    return prov_annotator, log_json


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
    ant_status, log_json = prov_annotator.test_antdoc_list(X_test)

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

    return prov_annotator, log_json


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


def skip_ebantdoc_list(ebantdoc_list: List[ebantdoc2.EbAnnotatedDoc2],
                       txt_fnlist: str):
    fn_list = txtreader.load_str_list(txt_fnlist)
    skip_fileid_set = set([])  # type: Set[str]
    for line in fn_list:
        cols = line.split(' ')
        if cols[0]:  # just in case a blank line
            skip_fileid_set.add(cols[0])
    result = [ebantdoc for ebantdoc in ebantdoc_list if ebantdoc.file_id not in skip_fileid_set]
    print("skip_ebantdoc_list(), orig = {}, out = {}".format(len(ebantdoc_list), len(result)))
    return result


def eval_line_annotator_with_trte(provision: str,
                                  txt_fn_list_fn: str,
                                  model_dir: str = 'dir-scut-model',
                                  work_dir: str = 'dir-work',
                                  is_doc_structure: bool = False):
    print('eval_line_annotator_with_trte(), provision: [{}]'.format(provision))
    ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(txt_fn_list_fn,
                                                       work_dir=work_dir,
                                                       is_doc_structure=is_doc_structure)
    # Sometimes annotation can be wrong to due changed guidelines, such as
    # composite date logic.  To avoid such cases, problematic annotated documents
    # can be removed per provision by adding skip files below.
    prov_skip_txt_fnlist = 'dict/{}_skip_doclist.txt'.format(provision)
    if os.path.exists(prov_skip_txt_fnlist):
        ebantdoc_list = skip_ebantdoc_list(ebantdoc_list, prov_skip_txt_fnlist)

    print("txt_fn_list_fn = [%s], len(ebantdoc) = %d" % (txt_fn_list_fn, len(ebantdoc_list)))

    provision_status_map = {'provision': provision}
    # update the hashmap of annotators
    if provision == 'title':
        prov_annotator = lineannotator.LineAnnotator('title', titles.TitleAnnotator('title'))
    elif provision == 'party':
        prov_annotator = lineannotator.LineAnnotator('party', parties.PartyAnnotator('party'))
    elif provision == 'date':
        prov_annotator = lineannotator.LineAnnotator(provision, dates.DateAnnotator(provision))

    # we need ebantdoc_list because it has the annotations
    provision_status_map['ant_status'] = prov_annotator.test_antdoc_list(ebantdoc_list)

    pprint(provision_status_map)


def eval_classifier(txt_fn_list,
                    work_dir: str,
                    model_file_name: str) -> None:
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
