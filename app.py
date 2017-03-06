# take a provision and a CSV file, and train the classifier, saving it to disk

# TODO should we be able to train individual provisions separately

import glob
import json
import logging
import os
import os.path
from pprint import pprint
import sys
import tempfile

from flask import Flask, request, jsonify
from sklearn.externals import joblib

from kirke.eblearn import ebrunner
from kirke.utils import osutils, strutils

import ebrevia.learn.learner as learner

# TODO: jshaw
# remove the following line to get rid of all logging messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
# app.debug = True
eb_files = os.environ['EB_FILES']
eb_models = os.environ['EB_MODELS']
print("eb files is: ", eb_files)
prefix = eb_files + "pymodel/"
learner_fns = [
    eb_models + 'learner.pkl',
    eb_models + 'learner_lease.pkl',
    eb_models + 'learner_oldlease.pkl',
    eb_models + 'learner_datePartyTitle.pkl']
## TODO, jshaw
## The code in extractor/dev branch has extra models, synchronize this
## manually
## eb_models + 'learner_affiliates.pkl']

# autoload all custom field classifiers
learner_fns = learner_fns + glob.glob(prefix + 'learner_cust_*.pkl')

ls = []

for learner_fn in learner_fns:
    if (os.path.exists(learner_fn)):
        # try to load existing learner and simply add new provisions
        l = joblib.load(learner_fn)
        l.prefix = prefix
        print("Loaded existing learner with provisions:", l.clfs.keys())
    else:
        l = learner.Learner(prefix)
    ls.append(l)

# hack due to using older model without re-pickling
ls[0].bag_transform.newschool = False


@app.route('/', methods=['POST'])
def classifyArff():
    with tempfile.NamedTemporaryFile() as tmp:
        arff = tmp.name
        request.files['arff'].save(arff)
        types = json.loads(request.form['types'])
        print("types are ", types)x
        pred = {}
        for l in ls:
            learner_types = set(types).intersection(l.clfs.keys())
            pred.update(l.predict(arff, learner_types))
        return jsonify({'predictions': pred})


# classifiers
WORK_DIR = 'data_from_web'
MODEL_DIR = 'sample_data2.model'
CUSTOM_MODEL_DIR = 'sample_data2.custmodel'
eb_runner = ebrunner.EbRunner(MODEL_DIR, WORK_DIR, CUSTOM_MODEL_DIR)

@app.route('/annotate-doc', methods=['POST'])
def annotate_uploaded_document():
    request_file_name = request.files['file'].filename
    if request_file_name:
        if not request_file_name.endswith('.txt'):
            request_file_name += '.txt'
        txt_basename = os.path.basename(request_file_name)
        file_name = os.path.normpath(os.path.join(WORK_DIR, txt_basename))
    else:
        file_name = tempfile.NamedTemporaryFile(dir=WORK_DIR).name + '.txt'

    request.files['file'].save(file_name)
    provisions_st = request.form.get('types')
    provision_set = set(provisions_st.split(',') if provisions_st else [])

    prov_labels_map = eb_runner.annotate_document(file_name, provision_set=provision_set)
    ebannotations = {'ebannotations': prov_labels_map}
    # pprint(prov_labels_map)
    pprint(ebannotations)
    return json.dumps(ebannotations)
    # print("simple stuff: type({})".format(type(prov_labels_map)))
    # print("simple stuff: {}".format(json.dumps(prov_labels_map)))
    # print("complex stuff: {}".format(jsonify(prov_labels_map)))
    # return json.dumps(prov_labels_map)
    # return jsonify(prov_labels_map)


@app.route('/cust-train/<cust_id>', methods=['POST'])
def cust_train(cust_id):
    name = 'cust_' + cust_id
    train_file = eb_files + 'models/sentencesV1_cust_' + cust_id + '.arff'
    test_file = eb_files + 'models/test-data/sentencesV1_cust_' + cust_id + '.arff'
    provisions = [name]
    fname = prefix + 'learner_cust_' + cust_id + '.pkl'

    newL = learner.Learner(prefix, False)
    newL.train(train_file, test_file, provisions)
    newL.save(fname)

    # add learner to running set of classifiers
    found = False
    for i, l in enumerate(ls):
        if name in l.clfs.keys():
            print("REPLACING CLASSIFIER")
            ls[i] = newL
            found = True
    if not found:
        print("ADDING CLASSIFIER")
        ls.append(newL)

    # return some json accuracy info
    return jsonify(newL.clfs[name].sgdClassifier.stats)


@app.route('/custom-train/<cust_id>', methods=['POST'])
def custom_train(cust_id):
    # to ensure that no accidental file name overlap
    print("cust_id = {}".format(cust_id))
    provision = 'cust_{}'.format(cust_id)
    tmp_dir = '{}/{}'.format(WORK_DIR, provision)
    osutils.mkpath(tmp_dir)
    fn_list = request.files.getlist('file')
    ant_fnames = []
    txt_fnames = []
    full_txt_fnames = []
    for name in [fstorage.filename for fstorage in fn_list]:
        if name.endswith('.ant'):
            ant_fnames.append(name)
        elif name.endswith('.txt'):
            txt_fnames.append(name)
            full_txt_fnames.append('{}/{}'.format(tmp_dir, name))
    print("txt_fnames (size={}) = {}".format(len(txt_fnames), txt_fnames))
    print("ant_fnames (size={})= {}".format(len(ant_fnames), ant_fnames))

    # name2 = request.form.get('custom_id')
    # print("name2 = {}".format(name2))
    # if passed in, use 'provision', otherwise, use cust_id
    # provision = request.form.get('provision', cust_id)
    # print("provision = {}".format(provision))

    for fstorage in fn_list:
        fn = '{}/{}'.format(tmp_dir, fstorage.filename)
        print("saving file '{}'".format(fn))
        fstorage.save(fn)
    txt_fn_list_fn = '{}/{}'.format(tmp_dir, 'txt_fnames.list')
    strutils.dumps('\n'.join(full_txt_fnames), txt_fn_list_fn)

    # When deploying, swap below two lines.  Otherwise, the error says
    # sample size 0 because no document has such provision.
    # provision = 'change_control'
    # Following the logic in the original code.
    eval_status = eb_runner.custom_train_provision_and_evaluate(txt_fn_list_fn,
                                                                provision,
                                                                CUSTOM_MODEL_DIR)
    # return some json accuracy info
    return jsonify(eval_status)


if __name__ == '__main__':
    train_file = '/Users/jakem/training-data/sentencesV1_party_small.arff'
    test_file = '/Users/jakem/training-data/test-data/sentencesV1_party_small.arff'
    provisions = ['date', 'party', 'title']
    fname = eb_models + 'learner_datePartyTitle.pkl'
    # where to store temporary output files

    if (len(sys.argv) == 2 and sys.argv[1] == '--train'):
        l = learner.Learner(prefix, False)
        # doesn't run on weekends
        l.train(train_file, test_file, provisions)
        l.save(fname)
    elif (len(sys.argv) == 3 and sys.argv[1] == '--cust-train'):
        cust_train(sys.argv[2])
    elif (len(sys.argv) == 2 and sys.argv[1] == '--test'):
        # predict
        ls[1].predict(test_file)
    else:
        app.run()
