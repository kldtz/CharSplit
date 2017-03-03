import json
import logging
from pprint import pprint

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from eblearn import ebtext2antdoc, ebannotator
from utils import strutils, splittrte


MIN_FULL_TRAINING_SIZE = 30


# Take all the data for training.
# Unless you know what you are doing, don't use this function, use
# train_eval_annotator() instead.
def _train_classifier(provision, txt_fn_list, work_dir, model_file_name, eb_classifier):
    eb_classifier.train(txt_fn_list, work_dir, model_file_name)
    return eb_classifier


# Take 1/5 of the data out for testing
# Train on 4/5 of the data
def train_eval_annotator(provision, txt_fn_list, work_dir, model_dir, model_file_name, eb_classifier,
                         custom_training_mode=False):
    logging.info("training_eval_annotator({}) called".format(provision))
    logging.info("    txt_fn_list = {}".format(txt_fn_list))
    logging.info("    work_dir = {}".format(work_dir))
    logging.info("    model_dir = {}".format(model_dir))
    logging.info("    model_file_name = {}".format(model_file_name))
    
    ebantdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fn_list, work_dir)
    ebsent_list = []
    for eb_antdoc in ebantdoc_list:
        ebsent_list.extend(eb_antdoc.get_ebsent_list())

    num_pos_label, num_neg_label = 0, 0
    for ebsent in ebsent_list:
        if provision in ebsent.labels:
            num_pos_label += 1
        else:
            num_neg_label += 1

    X = ebantdoc_list
    y = [provision in ebantdoc.get_provision_set()
         for ebantdoc in ebantdoc_list]

    # only in custom training mode and the positive training instances are too few
    # only train, no testing
    if custom_training_mode and num_pos_label < MIN_FULL_TRAINING_SIZE:
        logging.info("training with {} instances, no test (<{}) .  num_pos= {}, num_neg= {}".format(len(ebsent_list),
                                                                                                    MIN_FULL_TRAINING_SIZE,
                                                                                                    num_pos_label,
                                                                                                    num_neg_label))
        X_train = X
        train_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, provision)
        splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)
        eb_classifier.train_antdoc_list(X_train, work_dir, model_file_name)
        prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
        return prov_annotator

    logging.info("training with {} instances, num_pos= {}, num_neg= {}".format(len(ebsent_list),
                                                                               num_pos_label,
                                                                               num_neg_label))
    # we have enough positive training instances, so we do testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, provision)
    splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)
    test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
    splittrte.save_antdoc_fn_list(X_test, test_doclist_fn)

    eb_classifier.train_antdoc_list(X_train, work_dir, model_file_name)
    pred_status = eb_classifier.predict_and_evaluate(X_test, work_dir)

    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)
    ant_status = prov_annotator.test_antdoc_list(X_test, work_dir)

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
    provision_status_map['ant_status'] = prov_annotator.test_antdoc_list(ebantdoc_list, work_dir)
    
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
