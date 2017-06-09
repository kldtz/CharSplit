# take a provision and a CSV file, and train the classifier, saving it to disk

# TODO should we be able to train individual provisions separately

import json
import logging
import os
import os.path
from pprint import pprint
import tempfile

from flask import Flask, request, jsonify

from kirke.eblearn import ebrunner
from kirke.utils import osutils, strutils

# NOTE: Remove the following line to get rid of all logging messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')


app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
# app.debug = True
eb_files = os.environ['EB_FILES']
eb_models = os.environ['EB_MODELS']
print("eb files is: ", eb_files)
print("eb models is: ", eb_models)


# classifiers
WORK_DIR = 'data-from-web'
MODEL_DIR = eb_models
CUSTOM_MODEL_DIR = eb_files + 'pymodel'
osutils.mkpath(WORK_DIR)
osutils.mkpath(MODEL_DIR)
osutils.mkpath(CUSTOM_MODEL_DIR)

eb_runner = ebrunner.EbRunner(MODEL_DIR, WORK_DIR, CUSTOM_MODEL_DIR)

IS_SUPPORT_DOC_CLASSIFICATION = False
eb_doccat_runner = None
doccat_model_fn = MODEL_DIR + '/ebrevia_docclassifier.pkl'
if IS_SUPPORT_DOC_CLASSIFICATION and os.path.exists(doccat_model_fn):
    eb_doccat_runner = ebrunner.EbDocCatRunner(MODEL_DIR)


@app.route('/annotate-doc', methods=['POST'])
def annotate_uploaded_document():
    request_work_dir = request.form.get('workdir')
    if request_work_dir:
        work_dir = request_work_dir
        print("work_dir = {}".format(work_dir))
        osutils.mkpath(work_dir)
    else:
        work_dir = WORK_DIR

    request_file_name = request.files['file'].filename
    if request_file_name:
        if not request_file_name.endswith('.txt'):
            request_file_name += '.txt'
        txt_basename = os.path.basename(request_file_name)
        file_name = os.path.normpath(os.path.join(work_dir, txt_basename))
    else:
        file_name = tempfile.NamedTemporaryFile(dir=work_dir).name + '.txt'

    request.files['file'].save(file_name)
    provisions_st = request.form.get('types')
    provision_set = set(provisions_st.split(',') if provisions_st else [])

    if provision_set:
        # print("got provision_set: {}".format(sorted(provision_set)))
        provision_set.add('date')
        provision_set.add('sigdate')
        provision_set.add('effectivedate')
        if "effectivedate_auto" in provision_set:
            provision_set.remove('effectivedate_auto')
        # make sure these are removed due to low accuracy
        if "lic_licensee" in provision_set:
            provision_set.remove('lic_licensee')
        if "lic_licensor" in provision_set:
            provision_set.remove('lic_licensor')

    prov_labels_map, _ = eb_runner.annotate_document(file_name,
                                                     provision_set=provision_set,
                                                     work_dir=work_dir)
    ebannotations = {'ebannotations': prov_labels_map}
    # pprint(prov_labels_map)
    # pprint(ebannotations)

    if eb_doccat_runner != None:
        doc_catnames = eb_doccat_runner.classify_document(file_name)
        ebannotations['tags'] = doc_catnames

    return json.dumps(ebannotations)


# pylint: disable=too-many-locals
@app.route('/custom-train/<cust_id>', methods=['POST'])
def custom_train(cust_id):
    request_work_dir = request.form.get('workdir')
    if request_work_dir:
        work_dir = request_work_dir
        print("work_dir = {}".format(work_dir))
        osutils.mkpath(work_dir)
    else:
        work_dir = WORK_DIR

    # to ensure that no accidental file name overlap
    print("cust_id = {}".format(cust_id))
    provision = 'cust_{}'.format(cust_id)
    tmp_dir = '{}/{}'.format(work_dir, provision)
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
    # print("txt_fnames (size={}) = {}".format(len(txt_fnames), txt_fnames))
    # print("ant_fnames (size={})= {}".format(len(ant_fnames), ant_fnames))

    # name2 = request.form.get('custom_id')
    # print("name2 = {}".format(name2))
    # if passed in, use 'provision', otherwise, use cust_id
    # provision = request.form.get('provision', cust_id)
    # print("provision = {}".format(provision))

    for fstorage in fn_list:
        fstorage.save('{}/{}'.format(tmp_dir, fstorage.filename))
    txt_fn_list_fn = '{}/{}'.format(tmp_dir, 'txt_fnames.list')
    strutils.dumps('\n'.join(full_txt_fnames), txt_fn_list_fn)

    # When deploying, swap below two lines.  Otherwise, the error says
    # sample size 0 because no document has such provision.
    # provision = 'change_control'
    # Following the logic in the original code.
    eval_status = eb_runner.custom_train_provision_and_evaluate(txt_fn_list_fn,
                                                                provision,
                                                                CUSTOM_MODEL_DIR,
                                                                work_dir=work_dir)
    # copy the result into the expected format for client
    pred_status = eval_status['pred_status']['pred_status']
    cf_mtx = pred_status['confusion_matrix']
    status = {'confusion_matrix': [[cf_mtx['tn'], cf_mtx['fp']], [cf_mtx['fn'], cf_mtx['tp']]],
              'fscore': pred_status['f1'],
              'precision': pred_status['prec'],
              'recall': pred_status['recall']}

    print("status:")
    pprint(status)

    # return some json accuracy info
    return jsonify(status)


if __name__ == '__main__':
    app.run()
