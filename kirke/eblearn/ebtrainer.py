import json
import logging
from pprint import pprint

from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict, train_test_split

from kirke.eblearn import ebannotator, ebpostproc, ebtext2antdoc
from kirke.utils import evalutils, splittrte, strutils
from kirke.eblearn import ebattrvec

DEFAULT_CV = 3

# MIN_FULL_TRAINING_SIZE = 30
MIN_FULL_TRAINING_SIZE = 50
# MIN_FULL_TRAINING_SIZE = 400


# Take all the data for training.
# Unless you know what you are doing, don't use this function, use
# train_eval_annotator() instead.
def _train_classifier(txt_fn_list, work_dir, model_file_name, eb_classifier):
    eb_classifier.train(txt_fn_list, work_dir, model_file_name)
    return eb_classifier


# Take 1/5 of the data out for testing
# Train on 4/5 of the data
# pylint: disable=R0915, R0913, R0914
def train_eval_annotator(provision, txt_fn_list,
                         work_dir, model_dir, model_file_name, eb_classifier,
                         custom_training_mode=False):
    logging.info("training_eval_annotator(%s) called", provision)
    logging.info("    txt_fn_list = %s", txt_fn_list)
    logging.info("    work_dir = %s", work_dir)
    logging.info("    model_dir = %s", model_dir)
    logging.info("    model_file_name = %s", model_file_name)

    ebantdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fn_list, work_dir)
    attrvec_list = []
    for eb_antdoc in ebantdoc_list:
        attrvec_list.extend(eb_antdoc.get_attrvec_list())

    num_pos_label, num_neg_label = 0, 0
    for attrvec in attrvec_list:
        if provision in attrvec[ebattrvec.LABELS_INDEX]:
            num_pos_label += 1
        else:
            num_neg_label += 1

    # pylint: disable=C0103
    X = ebantdoc_list
    y = [provision in ebantdoc.get_provision_set()
         for ebantdoc in ebantdoc_list]

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
    # only train, no testing
    if custom_training_mode and num_pos_label < MIN_FULL_TRAINING_SIZE:
        logging.info("training with %d instances, no test (<%d) .  num_pos= %d, num_neg= %d",
                     len(attrvec_list), MIN_FULL_TRAINING_SIZE, num_pos_label, num_neg_label)
        X_train = X
        # y_train = y
        train_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, provision)
        splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)
        eb_classifier.train_antdoc_list(X_train, work_dir, model_file_name)

        # set up the status of the classifier, based on the best parameter
        print("eb_classifier.best_parameters")
        best_parameters = eb_classifier.eb_grid_search.best_estimator_.get_params()
        pprint(best_parameters)
        # for param_name in sorted(parameters.keys()):
        #    print("\t%s: %r" % (param_name, best_parameters[param_name]))
        alpha = best_parameters['alpha']
        print("alpha xxx = {}".format(alpha))

        iterations = 10
        # now X and y are different
        X_sent = eb_classifier.transformer.transform(attrvec_list)
        y_label_list = [provision in attrvec.labels for attrvec in attrvec_list]

        # the goal here is to provide some status information
        # no guarantee that it is consistent with eb_classifier status yet
        tmp_sgd_clf = SGDClassifier(loss='log', penalty='l2', alpha=alpha, n_iter=iterations,
                                    shuffle=True, random_state=42,
                                    class_weight={True: 10, False: 1})
        tmp_preds = cross_val_predict(tmp_sgd_clf, X_sent, y_label_list, cv=DEFAULT_CV)
        # this setup eb_classifier.status
        pred_status = calc_scut_predict_evaluate(eb_classifier,
                                                 attrvec_list, tmp_preds, y_label_list)

        # make the classifier into an annotator
        prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)

        ant_status = {'provisoin' : provision,
                      'pred_status' : pred_status}
        prov_annotator.eval_status = ant_status
        pprint(ant_status)

        model_status_fn = model_dir + '/' +  provision + ".status"
        strutils.dumps(json.dumps(ant_status), model_status_fn)
        return prov_annotator

    logging.info("training with %d instances, num_pos= %d, num_neg= %d",
                 len(attrvec_list), num_pos_label, num_neg_label)

    # we have enough positive training instances, so we do testing
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2,
                                             random_state=42, stratify=y)

    train_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, provision)
    splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)
    test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
    splittrte.save_antdoc_fn_list(X_test, test_doclist_fn)

    eb_classifier.train_antdoc_list(X_train, work_dir, model_file_name)
    pred_status = eb_classifier.predict_and_evaluate(X_test, work_dir)

    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    ant_status = prov_annotator.test_antdoc_list(X_test)

    ant_status['provision'] = provision
    ant_status['pred_status'] = pred_status
    prov_annotator.eval_status = ant_status
    pprint(ant_status)

    model_status_fn = model_dir + '/' +  provision + ".status"
    strutils.dumps(json.dumps(ant_status), model_status_fn)

    return prov_annotator


def eval_annotator(txt_fn_list, work_dir, model_file_name):
    eb_classifier = joblib.load(model_file_name)
    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    # update the hashmap of annotators
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    provision_status_map['ant_status'] = prov_annotator.test_antdoc_list(ebantdoc_list)

    pprint(provision_status_map)


def eval_classifier(txt_fn_list, work_dir, model_file_name):
    eb_classifier = joblib.load(model_file_name)
    provision = eb_classifier.provision
    print("provision = {}".format(provision))

    ebantdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    pred_status = eb_classifier.predict_and_evaluate(ebantdoc_list, work_dir)

    provision_status_map = {'provision': provision,
                            'pred_status': pred_status}

    pprint(provision_status_map)


# utility function
# this is mainly used for the outer testing (real hold out)
def calc_scut_predict_evaluate(scut_classifier, attrvec_list, y_pred, y_te):
    logging.info('calc_scut_predict_evaluate()...')

    sent_st_list = [attrvec[ebattrvec.TOKENS_TEXT_INDEX] for attrvec in attrvec_list]
    overrides = ebpostproc.gen_provision_overrides(scut_classifier.provision, sent_st_list)

    scut_classifier.pred_status['classifer_type'] = 'scutclassifier'
    scut_classifier.pred_status['pred_status'] = evalutils.calc_pred_status_with_prob(y_pred, y_te)
    scut_classifier.pred_status['override_status'] = (
        evalutils.calc_pred_override_status(y_pred, y_te, overrides))
    scut_classifier.pred_status['best_params_'] = scut_classifier.eb_grid_search.best_params_

    return scut_classifier.pred_status
