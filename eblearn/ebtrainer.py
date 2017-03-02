
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from eblearn import ebtext2antdoc, ebannotator, ebrunner
from utils import strutils, splittrte
from pprint import pprint
import json


# Take all the data for training.
# Unless you know what you are doing, don't use this function, use
# train_eval_annotator() instead.
def _train_classifier(provision, txt_fn_list, work_dir, model_file_name, eb_classifier):
    eb_classifier.train(txt_fn_list, work_dir, model_file_name)
    return eb_classifier


# Take 1/5 of the data out for testing
# Train on 4/5 of the data
def train_eval_annotator(provision, txt_fn_list, work_dir, model_dir, model_file_name, eb_classifier):
    ebantdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fn_list, work_dir)
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    ebsent_list = []
    for eb_antdoc in ebantdoc_list:
        ebsent_list.extend(eb_antdoc.get_ebsent_list())
    # label_list = [provision in ebsent.labels for ebsent in ebsent_list]        
    print("len(ebantdoc_list) = {}".format(len(ebantdoc_list)))

    doc_labellist_list = []
    for eb_antdoc in ebantdoc_list:
        ebsent_list = eb_antdoc.get_ebsent_list()
        labellist_list = [provision in ebsent.labels for ebsent in ebsent_list]
        doc_labellist_list.append(labellist_list)

    X = ebantdoc_list
    y = doc_labellist_list
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_doclist_fn = "{}/{}_train_doclist.txt".format(model_dir, provision)        
    splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)

    eb_classifier.train_antdoc_list(X_train, work_dir, model_file_name)

    test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
    splittrte.save_antdoc_fn_list(X_test, test_doclist_fn)        

    pred_status = eb_classifier.predict_and_evaluate(X_test, work_dir)

    # update the hashmap of annotators
    prov_annotator = ebannotator.ProvisionAnnotator(eb_classifier, work_dir)

    ant_status = prov_annotator.test_antdoc_list(X_test, work_dir)

    ant_status['provision'] = provision
    ant_status['pred_status'] = pred_status
    prov_annotator.eval_status = ant_status

    model_status_fn = model_dir + '/' +  provision + ".status"
    strutils.dumps(json.dumps(ant_status), model_status_fn)

    pprint(ant_status)

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
