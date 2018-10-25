from collections import defaultdict
from datetime import datetime
import json
import logging
import os
import pprint
import time
# pylint: disable=unused-import
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from kirke.eblearn import annotatorconfig, ebannotator, ebattrvec, ebpostproc
from kirke.eblearn import lineannotator, ruleannotator, spanannotator
from kirke.ebrules import titles, parties, dates
from kirke.utils import  ebantdoc4, evalutils, splittrte, strutils, txtreader

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IS_DEBUG_DOC_ORDER = False


DEFAULT_CV = 3

# MIN_FULL_TRAINING_SIZE = 30
## this is the original val
# MIN_FULL_TRAINING_SIZE = 50
# MIN_FULL_TRAINING_SIZE = 400

# MIN_FULL_TRAINING_SIZE = 150
MIN_FULL_TRAINING_SIZE = 100

# training using cross validation of 600 docs took around 38 minutes
# MAX_DOCS_FOR_TRAIN_CROSS_VALIDATION = 600

# Take all the data for training.
# Unless you know what you are doing, don't use this function, use
# train_eval_annotator() instead.
def _train_classifier(txt_fn_list, work_dir, model_file_name, eb_classifier):
    eb_classifier.train(txt_fn_list, work_dir, model_file_name)
    return eb_classifier


def log_model_eval_status(ant_status: Dict[str, Any]) -> None:
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


def log_custom_model_eval_status(ant_status: Dict[str, Any]) -> None:
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


# pylint: disable=too-many-arguments, too-many-locals
def cv_train_at_annotation_level(provision,
                                 x_antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                 bool_list,
                                 nbest,
                                 eb_classifier_orig,
                                 model_file_name: str,
                                 model_num: int,
                                 model_dir: str,
                                 work_dir: str):
    # we do 3-fold cross validation, as the big set for custom training
    # test_size = 0.33
    # this will be looped mutliple times, so a list, not a generator

    num_fold = DEFAULT_CV
    # distribute positives to all buckets
    pos_list, neg_list = [], []
    for x_antdoc, label in zip(x_antdoc_list, bool_list):
        if label:
            pos_list.append((x_antdoc, label))
        else:
            neg_list.append((x_antdoc, label))
    if IS_DEBUG_DOC_ORDER:
        print('len(pos_list) = {}, len(neg_list) = {}'.format(len(pos_list),
                                                              len(neg_list)))
    pos_list.extend(neg_list)
    # this is a list of ebantdoc
    bucket_x_map = defaultdict(list)  # type: DefaultDict[int, List]
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

        train_buckets = []  # type: List
        test_bucket = []  # type: List
        for bnum, bucket_x in bucket_x_map.items():
            if bnum != bucket_num:
                train_buckets.extend(bucket_x)
            else:  # bnum == bucket_num
                test_bucket = bucket_x

        if IS_DEBUG_DOC_ORDER:
            for tr_num, tr_doc in enumerate(train_buckets):
                print("bucknum={}, tr_num={}, fn={}".format(bucket_num, tr_num, tr_doc.file_id))
            for te_num, te_doc in enumerate(test_bucket):
                print("bucknum={}, test_num={}, fn={}".format(bucket_num, te_num, te_doc.file_id))

        cv_eb_classifier = eb_classifier_orig.make_bare_copy()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        cv_eb_classifier_fn = '/tmp/{}-{}-{}'.format(provision, timestr, bucket_num)
        cv_eb_classifier.train_antdoc_list(train_buckets,
                                           work_dir,
                                           cv_eb_classifier_fn)
        cv_prov_annotator = ebannotator.ProvisionAnnotator(cv_eb_classifier, work_dir, nbest=nbest)

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
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir, nbest=nbest)
    log_json = log_list
    merged_ant_status = \
        evalutils.aggregate_ant_status_list(cv_ant_status_list)['ant_status']

    ant_status = {'provision': provision}
    ant_status['ant_status'] = merged_ant_status
    # we are going to fake it for now
    ant_status['pred_status'] = {'pred_status': merged_ant_status}
    prov_annotator.eval_status = ant_status
    pprint.pprint(ant_status)

    model_status_fn = '{}/{}.{}.status'.format(model_dir,
                                               provision,
                                               model_num)
    strutils.dumps(json.dumps(ant_status), model_status_fn)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    result_fn = model_dir + '/' + provision + "-ant_result-" + \
                timestr + ".json"
    logger.info('wrote result file at: %s', result_fn)
    strutils.dumps(json.dumps(log_json), result_fn)

    log_custom_model_eval_status({'provision': provision,
                                  'ant_status': merged_ant_status})

    return prov_annotator, log_list


# TODO, because we are using ebantdoc4 instead of ebantdoc4, I am
# doing code copying right now.  Maybe merge with cv_train_at_annotation_level()
# in future.
# pylint: disable=too-many-arguments, too-many-locals, invalid-name, too-many-statements
def cv_candg_train_at_annotation_level(provision: str,
                                       # pylint: disable=invalid-name
                                       antdoc_candidatex_list: List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                                                          List[Dict],
                                                                          List[bool],
                                                                          List[int]]],
                                       antdoc_bool_list: List[bool],
                                       sp_annotator_orig: spanannotator.SpanAnnotator,
                                       model_dir: str,
                                       work_dir: str):
    # we do 4-fold cross validation, as the big set for custom training
    # test_size = 0.25

    num_fold = 3
    all_candidates = []  # type: List[Dict]
    all_candidate_labels = []  # type: List[bool]
    all_group_ids = []  # type: List[int]

    # distribute positives to all buckets
    pos_list = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
    neg_list = []  # type: List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]
    num_pos_after_candgen = 0
    # pylint: disable=line-too-long
    for label, (x_antdoc, x_candidates, x_candidate_label_list, x_group_ids) in zip(antdoc_bool_list,
                                                                                    antdoc_candidatex_list):

        if label:
            pos_list.append((x_antdoc, x_candidates, x_candidate_label_list, x_group_ids))
        else:
            neg_list.append((x_antdoc, x_candidates, x_candidate_label_list, x_group_ids))
        all_candidates.extend(x_candidates)
        all_candidate_labels.extend(x_candidate_label_list)
        all_group_ids.extend(x_group_ids)
        if True in x_candidate_label_list:
            num_pos_after_candgen += 1

    pos_list.extend(neg_list)
    if num_pos_after_candgen < 6:
        exc = Exception("INSUFFICIENT_EXAMPLES: Too few documents with positive candidates, {} found".format(num_pos_after_candgen))
        exc.user_message = "INSUFFICIENT_EXAMPLES"  # type: ignore
        raise exc
    # pylint: disable=line-too-long
    bucket_x_map = defaultdict(list)  # type: DefaultDict[int, List[Tuple[ebantdoc4.EbAnnotatedDoc4, List[Dict], List[bool], List[int]]]]
    for count, (x_antdoc, x_candidates, x_candidate_label_list, x_group_ids) in enumerate(pos_list):
        # bucket_x_map[count % num_fold].append((x_antdoc, label))
        bucket_x_map[count % num_fold].append((x_antdoc, x_candidates, x_candidate_label_list, x_group_ids))

    #for bnum, alist in bucket_x_map.items():
    #    print("-----")
    #    for ebantdoc, y in alist:
    #        print("{}\t{}\t{}".format(bnum, ebantdoc.file_id, y))

    log_list = {}  # type: Dict
    cv_ant_status_list = []
    for bucket_num in range(num_fold):  # cross train each bucket

        train_bucket_candidates = []   # type: List[Dict]
        train_bucket_candidate_labels = []   # type: List[bool]
        train_bucket_group_ids = []   # type: List[int]
        test_bucket_candidates = []   # type: List[Dict]
        test_bucket_candidate_labels = []   # type: List[bool]
        test_bucket_group_ids = []   # type: List[int]
        test_bucket_antdoc_list = []  # type: List[ebantdoc4.EbAnnotatedDoc4]
        for bnum, bucket_docxyz_list in bucket_x_map.items():
            if bnum != bucket_num:
                for docxyz in bucket_docxyz_list:
                    train_bucket_candidates.extend(docxyz[1])
                    train_bucket_candidate_labels.extend(docxyz[2])
                    train_bucket_group_ids.extend(docxyz[3])
            else:  # bnum == bucket_num
                for docxyz in bucket_docxyz_list:
                    test_bucket_antdoc_list.append(docxyz[0])
                    test_bucket_candidates.extend(docxyz[1])
                    test_bucket_candidate_labels.extend(docxyz[2])
                    test_bucket_group_ids.extend(docxyz[3])

        cv_sp_annotator = sp_annotator_orig.make_bare_copy()
        cv_sp_annotator.train_candidates(train_bucket_candidates,
                                         train_bucket_candidate_labels,
                                         train_bucket_group_ids,
                                         cv_sp_annotator.pipeline,
                                         cv_sp_annotator.gridsearch_parameters,
                                         work_dir)

        # annotates the test set and runs through evaluation
        # pred_status = cv_sp_annotator.predict_and_evaluate(test_bucket_candidates,
        #                                                    test_bucket_candidate_labels,
        #                                                    work_dir)

        # ant_status, log_json = \
        #    cv_sp_annotator.test_antdoc_list(test_bucket_antdoc_list
        #                                     cv_sp_annotator.threshold)

        cv_ant_status, cv_log_json = cv_sp_annotator.test_antdoc_list(test_bucket_antdoc_list,
                                                                      cv_sp_annotator.threshold)
        # print("cv ant_status, bucket_num = {}:".format(bucket_num))
        # print(cv_ant_status)

        log_list.update(cv_log_json)
        cv_ant_status_list.append(cv_ant_status)

    # now build the annotator using ALL training data
    sp_annotator = sp_annotator_orig.make_bare_copy()
    sp_annotator.train_candidates(all_candidates,
                                  all_candidate_labels,
                                  all_group_ids,
                                  sp_annotator.pipeline,
                                  sp_annotator.gridsearch_parameters,
                                  work_dir)

    prov_annotator = sp_annotator
    log_json = log_list
    merged_ant_status = \
        evalutils.aggregate_ant_status_list(cv_ant_status_list)['ant_status']

    ant_status = {'provision': provision}
    prov_annotator.classifier_status['eval_status'] = merged_ant_status
    prov_annotator.ant_status['eval_status'] = merged_ant_status
    # pprint.pprint(prov_annotator.ant_status)

    model_status_fn = model_dir + '/' +  provision + ".status"
    strutils.dumps(json.dumps(ant_status), model_status_fn)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    result_fn = model_dir + '/' + provision + "-ant_result-" + \
                timestr + ".json"
    logger.info('wrote result file at: %s', result_fn)
    strutils.dumps(json.dumps(log_json), result_fn)

    log_custom_model_eval_status({'provision': provision,
                                  'ant_status': merged_ant_status})

    return prov_annotator, log_list



# For our default provisions, we take 4/5 for training, 1/5 for testing
# For bespoke, < 600 docs, train/test split we take 3/4 for training, 1/4 for testing
# For bespoke, >= 600 docs, cross validate
# pylint: disable=too-many-branches
def train_eval_annotator(provision: str,
                         model_num: int,
                         doc_lang: str,
                         nbest: int,
                         txt_fn_list,
                         work_dir,
                         model_dir,
                         model_file_name,
                         eb_classifier,
                         is_bespoke_mode: bool = False) \
                         -> Tuple[ebannotator.ProvisionAnnotator, Dict[str, Dict]]:
    logger.info("training_eval_annotator(%s) called", provision)
    logger.info("    txt_fn_list = %s", txt_fn_list)
    logger.info("    work_dir = %s", work_dir)
    logger.info("    model_dir = %s", model_dir)
    logger.info("    model_file_name = %s", model_file_name)
    logger.info("    is_bespoke_mode= %r", is_bespoke_mode)

    # group_id is used to ensure all the attrvec for a document are together
    # and not distributed between both training and testing.  A document can
    # be either in training or testing, but not both.  This parameter is used
    # scikit learn crosss validation.  James implemented something similar in
    # an earlier commit.  After realizing scikit learn supports this functionality
    # directly, we switched to scikit learn's mechanism.
    eb_antdoc_list = \
        ebantdoc4.doclist_to_ebantdoc_list(txt_fn_list,
                                           work_dir,
                                           doc_lang=doc_lang,
                                           is_doc_structure=True,
                                           is_bespoke_mode=is_bespoke_mode,
                                           # to keep file order stable to ensure
                                           # consistent numbers
                                           is_sort_by_file_id=True)
    num_docs = len(eb_antdoc_list)
    attrvec_list = []  # type: List[ebattrvec.EbAttrVec]
    group_id_list = []  # type: List[int]
    # the y is document-based for ebantdoc, not for attrvec
    # pylint: disable=invalid-name
    y = []  # type: List[bool]
    num_doc_pos, num_doc_neg = 0, 0
    num_pos_ant = 0
    num_pos_label, num_neg_label = 0, 0
    for group_id, eb_antdoc in enumerate(eb_antdoc_list):
        tmp_attrvec_list = eb_antdoc.get_attrvec_list()
        attrvec_list.extend(tmp_attrvec_list)
        group_id_list.extend([group_id] * len(tmp_attrvec_list))

        # pos counts only for this doc
        doc_pos_ant, doc_pos_label = 0, 0
        human_ant_list = eb_antdoc.prov_annotation_list
        for human_ant in human_ant_list:
            if provision == human_ant.label:
                num_pos_ant += 1
                doc_pos_ant += 1

        has_pos_label = False
        for attrvec in tmp_attrvec_list:
            if provision in attrvec.labels:
                num_pos_label += 1
                doc_pos_label += 1
                has_pos_label = True
            else:
                num_neg_label += 1

        if has_pos_label:
            num_doc_pos += 1
            y.append(True)
        else:
            num_doc_neg += 1
            y.append(False)

        if doc_pos_ant != doc_pos_label:
            # sometimes a human annotation can go across mutliple sentences
            logger.info('doc %s has %d human annotation, found %d in positive sentences',
                        eb_antdoc.file_id, doc_pos_ant, doc_pos_label)

    if num_pos_ant != num_pos_label:
        # sometimes a human annotation can go across mutliple sentences
        logger.info('train_eval_annotator, found %d human annotation, '
                    'found %d in positive sentences',
                    num_pos_ant, num_pos_label)

    # X is sorted by file_id already
    # pylint: disable=invalid-name
    X = eb_antdoc_list

    logger.info("provision: %s, num_doc = %d, num_doc_pos= %d, num_doc_neg= %d",
                provision, num_docs, num_doc_pos, num_doc_neg)

    # TODO, jshaw, hack, such as for sechead
    if num_doc_neg < 2:
        y[0] = False
        y[1] = False

    # make sure nbest is know to everyone else
    eb_classifier.nbest = nbest

    # runs when custom training mode positive training instances are too few
    # only train, no independent testing
    # corss validation is applied to all Bespoke training
    # Always do cross validation to keep N constant as what people expect.
    if is_bespoke_mode:
        # pylint: disable=line-too-long
        logger.info("training using cross validation with %d instances across %d documents.  num_inst_pos= %d, num_inst_neg= %d",
                    len(attrvec_list), len(eb_antdoc_list), num_pos_label, num_neg_label)

        X_train = X
        train_doclist_fn = "{}/{}_{}_train_doclist.txt".format(model_dir, provision, doc_lang)

        #splits training doclist for cross validation
        splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)

        prov_annotator2, combined_log_json = \
            cv_train_at_annotation_level(provision,
                                         X_train,
                                         y,
                                         nbest,
                                         eb_classifier,
                                         model_file_name,
                                         model_num,
                                         model_dir,
                                         work_dir)

        return prov_annotator2, combined_log_json

    # pylint: disable=line-too-long
    logger.info("training using train/test split with %d instances.  num_inst_pos= %d, num_inst_neg= %d",
                len(attrvec_list), num_pos_label, num_neg_label)

    if is_bespoke_mode:
        test_size = 0.25
    else:
        test_size = 0.2

    #splits training and testing data, saves to a doclist
    X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size,
                                             random_state=42, stratify=y)
    train_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, provision)
    splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)
    test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
    splittrte.save_antdoc_fn_list(X_test, test_doclist_fn)

    #trains on the training data, evaluates the testing data
    eb_classifier.train_antdoc_list(X_train, work_dir, model_file_name)
    pred_status = eb_classifier.predict_and_evaluate(X_test, work_dir)

    #make the classifier into an annotator
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir, nbest=nbest)
    # X_test is now traindoc, not ebantdoc.  The testing docs are loaded one by one
    # using generator, instead of all loaded at once.

    # X_test_antdoc_list = ebantdoc4.traindoc_list_to_antdoc_list(X_test, work_dir)
    # ant_status, log_json = prov_annotator.test_antdoc_list(X_test_antdoc_list)
    # X_test_antdoc_list = ebantdoc4.traindoc_list_to_antdoc_list(X_test, work_dir)
    ant_status, log_json = prov_annotator.test_antdoc_list(X_test)

    #prints evaluation results and saves status
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
def train_eval_annotator_with_trte(provision: str,
                                   work_dir: str,
                                   model_dir: str,
                                   model_file_name: str,
                                   eb_classifier,
                                   is_cache_enabled=True) \
                                   -> Tuple[ebannotator.ProvisionAnnotator,
                                            Dict[str, Any],
                                            Dict[str, Dict]]:
    logger.info("training_eval_annotator_with_trte(%s) called", provision)
    logger.info("    work_dir = %s", work_dir)
    logger.info("    model_dir = %s", model_dir)
    logger.info("    model_file_name = %s", model_file_name)

    train_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, provision)
    # pylint: disable=invalid-name
    X_train = ebantdoc4.doclist_to_ebantdoc_list(train_doclist_fn,
                                                 work_dir,
                                                 is_cache_enabled=is_cache_enabled,
                                                 is_sort_by_file_id=True)
    eb_classifier.train_antdoc_list(X_train, work_dir, model_file_name)
    X_train = None  # free that memory

    test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
    # pylint: disable=invalid-name
    X_test = ebantdoc4.doclist_to_ebantdoc_list(test_doclist_fn,
                                                work_dir,
                                                is_cache_enabled=is_cache_enabled)
    pred_status = eb_classifier.predict_and_evaluate(X_test, work_dir)

    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    ant_status, log_json = prov_annotator.test_antdoc_list(X_test)

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

    return prov_annotator, ant_status['ant_status'], log_json


# For our default provisions, we take 4/5 for training, 1/5 for testing
# For bespoke, < 600 docs, train/test split we take 3/4 for training, 1/4 for testing
# For bespoke, >= 600 docs, cross validate
# TODO, for both is_doc_structure==True and is_doc_structure==False, the results so far is the
# same.  But in the future, will try to move toward is_doc_structure==True when we get
# a better PDF document text.
def train_eval_span_annotator(provision: str,
                              model_num: int,
                              # TODO, why is doc_lang not used?
                              # For now, there is no lang specific spanannotator?
                              unused_doc_lang: str,
                              nbest: int,
                              candidate_types: List[str],
                              work_dir: str,
                              model_dir: str,
                              txt_fn_list: Optional[str] = None,
                              model_file_name: Optional[str] = None,
                              is_doc_structure: bool = True,
                              is_bespoke_mode: bool = False) \
            -> Tuple[spanannotator.SpanAnnotator,
                     Dict[str, Dict]]:

    #get configs based on candidate_type or provision
    config = annotatorconfig.get_ml_annotator_config(candidate_types)
    if not model_file_name:
        model_file_name = '{}/{}_{}_annotator.v{}.pkl'.format(model_dir,
                                                              provision,
                                                              "-".join(candidate_types),
                                                              config['version'])
    #initializes annotator based on configs
    span_annotator = \
        spanannotator.SpanAnnotator(provision,
                                    candidate_types,
                                    version=config['version'],
                                    nbest=nbest,
                                    text_type=config.get('text_type', 'text'),
                                    doclist_to_antdoc_list=config['doclist_to_antdoc_list'],
                                    is_use_corenlp=config['is_use_corenlp'],
                                    doc_to_candidates=config['doc_to_candidates'],
                                    candidate_transformers=config.get('candidate_transformers', []),
                                    doc_postproc_list=config.get('doc_postproc_list', []),
                                    pipeline=config['pipeline'],
                                    gridsearch_parameters=config['gridsearch_parameters'],
                                    # we prefer recall over precision
                                    threshold=config.get('threshold', 0.24),
                                    kfold=config.get('kfold', 3))

    logger.info("training_eval_span_annotator(%s) called", provision)
    logger.info("    work_dir = %s", work_dir)
    logger.info("    model_file_name = %s", model_file_name)

    # this is mainly for paragraph
    if span_annotator.text_type == 'nlp_text':
        is_doc_structure = True

    if is_bespoke_mode:
        # converts all docs to ebantdocs
        # intentially don't specify is_no_corenlp here.  It has to be done
        # using ebantdoc4.doclist_to_antdoc_list_no_corenlp
        eb_antdoc_list = \
            span_annotator.doclist_to_antdoc_list(txt_fn_list,
                                                  work_dir,
                                                  is_bespoke_mode=is_bespoke_mode,
                                                  is_doc_structure=is_doc_structure,
                                                  # pylint: disable=line-too-long
                                                  is_use_corenlp=span_annotator.get_is_use_corenlp(),
                                                  # we need the files to be sorted in order to
                                                  # ensure stable training.
                                                  is_sort_by_file_id=True)
        # num_docs = len(eb_antdoc_list)

        #split training and test data, save doclists
        X = eb_antdoc_list
        y = [provision in eb_antdoc.get_provision_set()
             for eb_antdoc in eb_antdoc_list]
        num_pos_label, num_neg_label = 0, 0
        for doc_bool_label in y:
            if doc_bool_label:
                num_pos_label += 1
            else:
                num_neg_label += 1

        # candidate generation on the whole training set
        X_all_antdoc_candidatex_list = \
            span_annotator.documents_to_candidates(X, provision)
        # pylint: disable=line-too-long
        logger.info("%s extracted %d candidates across %d documents.",
                    candidate_types, sum([len(x[1]) for x in X_all_antdoc_candidatex_list]), len(eb_antdoc_list))

        prov_annotator2, combined_log_json = \
            cv_candg_train_at_annotation_level(provision,
                                               X_all_antdoc_candidatex_list,
                                               y,
                                               span_annotator,
                                               model_dir,
                                               work_dir)
        prov_annotator2.print_eval_status(model_dir, model_num)
        prov_annotator2.save(model_file_name)

        return prov_annotator2, combined_log_json

    # this is NOT bespoke
    train_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, provision)
    test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
    # loads existing doclists
    X_train = \
        span_annotator.doclist_to_antdoc_list(train_doclist_fn,
                                              work_dir,
                                              is_bespoke_mode=is_bespoke_mode,
                                              is_doc_structure=is_doc_structure,
                                              is_use_corenlp=span_annotator.get_is_use_corenlp(),
                                              # we need the file order to be stable, even if input
                                              # file order is changed.
                                              is_sort_by_file_id=True)
    # we don't care about the file order for testing
    X_test = span_annotator.doclist_to_antdoc_list(test_doclist_fn,
                                                   work_dir,
                                                   is_bespoke_mode=is_bespoke_mode,
                                                   is_doc_structure=is_doc_structure,
                                                   # pylint: disable=line-too-long
                                                   is_use_corenlp=span_annotator.get_is_use_corenlp())

    # candidate generation on training set
    train_antdoc_candidatex_list = \
        span_annotator.documents_to_candidates(X_train, provision)
    # We use the whole doc for test later, in span_annotator.test_antdoc_list(X_test, ...)
    # candidate generation on test set
    # test_antdoc_candidatex_list = \
    #     span_annotator.documents_to_candidates(X_test, provision)

    train_candidates, train_label_list, train_group_ids = \
        spanannotator.antdoc_candidatex_list_to_candidatex(train_antdoc_candidatex_list)

    # test_candidates, test_label_list, unused_test_group_ids = \
    #     spanannotator.antdoc_candidatex_list_to_candidatex(test_antdoc_candidatex_list)

    # trains an annotator
    span_annotator.train_candidates(train_candidates,
                                    train_label_list,
                                    train_group_ids,
                                    span_annotator.pipeline,
                                    span_annotator.gridsearch_parameters,
                                    work_dir)

    # annotates the test set and runs through evaluation
    # pred_status = \
    #     span_annotator.predict_and_evaluate(test_candidates,
    #                                         test_label_list,
    #                                         work_dir)

    unused_ant_status, log_json = \
        span_annotator.test_antdoc_list(X_test,
                                        span_annotator.threshold)
    span_annotator.print_eval_status(model_dir, model_num)
    span_annotator.save(model_file_name)

    return span_annotator, log_json


def eval_rule_annotator_with_trte(label: str,
                                  model_dir='dir-scut-model',
                                  work_dir='dir-work',
                                  is_train_mode=False):
    config = annotatorconfig.get_rule_annotator_config(label)

    rule_annotator = \
        ruleannotator.RuleAnnotator(label,
                                    version=config['version'],
                                    doclist_to_antdoc_list=config['doclist_to_antdoc_list'],
                                    is_use_corenlp=config['is_use_corenlp'],
                                    doc_to_candidates=config['doc_to_candidates'],
                                    rule_engine=config['rule_engine'],
                                    post_process=config.get('post_process', []))

    logger.info("eval_rule_annotator_with_trte(%s) called", label)

    # Normally, we compare the test results
    # During development, use is_train_mode to peek at the data for improvements
    if is_train_mode:
        test_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, label)
    else:
        test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, label)

    test_antdoc_list = \
        rule_annotator.doclist_to_antdoc_list(test_doclist_fn,
                                              work_dir,
                                              is_doc_structure=False,
                                              is_use_corenlp=rule_annotator.get_is_use_corenlp())

    # the eval result is already saved in span_annotator
    unused_ant_status = rule_annotator.test_antdoc_list(test_antdoc_list)
    print("ant_status x24: {}".format(rule_annotator.ant_status))

    # rule_annotator.save(model_file_name)
    rule_annotator.print_eval_status(model_dir)

    return rule_annotator


def eval_annotator(txt_fn_list, work_dir, model_file_name):
    eb_classifier = joblib.load(model_file_name)
    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebantdoc4.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
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

    ebantdoc_list = ebantdoc4.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    # update the hashmap of annotators
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    provision_status_map['ant_status'] = prov_annotator.test_antdoc_list(ebantdoc_list)

    pprint.pprint(provision_status_map)


def skip_ebantdoc_list(ebantdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
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
                                  work_dir: str = 'dir-work',
                                  is_doc_structure: bool = False):
    print('eval_line_annotator_with_trte(), provision: [{}]'.format(provision))
    ebantdoc_list = ebantdoc4.doclist_to_ebantdoc_list(txt_fn_list_fn,
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

    pprint.pprint(provision_status_map)


def eval_classifier(txt_fn_list,
                    work_dir: str,
                    model_file_name: str) -> None:
    eb_classifier = joblib.load(model_file_name)
    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebantdoc4.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    pprint.pprint(provision_status_map)


# utility function
# this is mainly used for the outer testing (real hold out)
def calc_scut_predict_evaluate(scut_classifier, attrvec_list, y_pred, y_te):
    logger.info('calc_scut_predict_evaluate()...')

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
