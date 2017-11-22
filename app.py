# take a provision and a CSV file, and train the classifier, saving it to disk

# TODO should we be able to train individual provisions separately

import configparser
import copy
import json
import logging
import os.path


from flask import Flask, request, jsonify

from kirke.eblearn import ebrunner
from kirke.utils import osutils, strutils

config = configparser.ConfigParser()
config.read('kirke.ini')

# NOTE: Remove the following line to get rid of all logging messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
# app.debug = True
eb_files = os.environ['EB_FILES']
eb_models = os.environ['EB_MODELS']
print("eb files is: ", eb_files)
print("eb models is: ", eb_models)

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

# classifiers
WORK_DIR = 'data-from-web'
MODEL_DIR = eb_models
CUSTOM_MODEL_DIR = eb_files + 'pymodel'
osutils.mkpath(WORK_DIR)
osutils.mkpath(MODEL_DIR)
osutils.mkpath(CUSTOM_MODEL_DIR)

eb_runner = ebrunner.EbRunner(MODEL_DIR, WORK_DIR, CUSTOM_MODEL_DIR)

if not eb_runner:
    logging.error('problem initializing ebrunner')
    exit(1)

@app.route('/annotate-doc', methods=['POST'])
def annotate_uploaded_document():
    request_work_dir = request.form.get('workdir')
    file_title = request.form.get('fileName')
    if request_work_dir:
        work_dir = request_work_dir
        print("work_dir = {}".format(work_dir))
        osutils.mkpath(work_dir)
    else:
        work_dir = WORK_DIR

    request_file_name, pdf_offsets_file_name = '', ''
    fn_list = request.files.getlist('file')
    for fstorage in fn_list:
        fn = '{}/{}'.format(work_dir, fstorage.filename)
        print("saving file '{}'".format(fn))
        fstorage.save(fn)

        if fn.endswith('.offsets.json'):
            pdf_offsets_file_name = fn
        elif fn.endswith('.txt'):
            txt_file_name = fn

            # save the file name
            meta_fn = '{}/{}'.format(work_dir,
                                     fstorage.filename.replace('.txt', '.meta'))
            print("wrote meta_fn: '{}'".format(meta_fn))
            print("doc title: '{}'".format(file_title))
            with open(meta_fn, 'wt') as meta_out:
                print('pdf_file\t{}'.format(file_title), file=meta_out)
                print('txt_file\t{}'.format(fstorage.filename), file=meta_out)
        else:
            logging.warning('unknown file extension in annotate_uploaded_document(): "{}"'.format(fn))

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
        # 'rate_table' is a special provision, with only rule-based model
        # avoid apploying the normal ML model
        if "rate_table" in provision_set:
            provision_set.remove('rate_table')

    # provision_set = set(['date', 'effectivedate', 'party', 'sigdate', 'term', 'title'])

    prov_labels_map, _ = eb_runner.annotate_document(txt_file_name,
                                                     provision_set=provision_set,
                                                     work_dir=work_dir)

    # because special case of 'effectivdate_auto'
    if prov_labels_map.get('effectivedate'):
        effectivedate_annotations = copy.deepcopy(prov_labels_map.get('effectivedate', []))
        for eff_ant in effectivedate_annotations:
            eff_ant['label'] = 'effectivedate_auto'
        prov_labels_map['effectivedate_auto'] = effectivedate_annotations
        del prov_labels_map['effectivedate']

    ebannotations = {'ebannotations': prov_labels_map}
    # pprint(prov_labels_map)
    # pprint(ebannotations)

    return json.dumps(ebannotations)


@app.route('/custom-train/<cust_id>', methods=['POST'])
def custom_train(cust_id):
    request_work_dir = request.form.get('workdir')
    if request_work_dir:
        work_dir = request_work_dir
        logging.info("work_dir = {}".format(work_dir))
        osutils.mkpath(work_dir)
    else:
        work_dir = WORK_DIR

    # to ensure that no accidental file name overlap
    logging.info("cust_id = {}".format(cust_id))
    provision = 'cust_{}'.format(cust_id)
    tmp_dir = '{}/{}'.format(work_dir, provision)
    osutils.mkpath(tmp_dir)
    fn_list = request.files.getlist('file')
    
    # save all the uploaded files in a location
    for fstorage in fn_list:
        fn = '{}/{}'.format(tmp_dir, fstorage.filename)
        # print("saving file '{}'".format(fn))
        fstorage.save(fn)

    ant_fnames = []
    txt_fnames = []
    full_txt_fnames = []
    txt_offsets_fn_map = {}
    for name in [fstorage.filename for fstorage in fn_list]:
        if name.endswith('.ant'):
            ant_fnames.append(name)
        elif name.endswith('.txt'):
            txt_fnames.append(name)
            full_txt_fnames.append('{}/{}'.format(tmp_dir, name))
        elif name.endswith('.offsets.json'):
            # create txt -> offsets.json map in order to do sent4nlp processing
            tmp_txt_fn = name.replace(".offsets.json", ".txt")
            txt_offsets_fn_map[tmp_txt_fn] = name
        else:
            logging.warning('unknown file extension in custom_train(): "{}"'.format(fn))

    print("full_txt_fnames (size={}) = {}".format(len(full_txt_fnames),
                                                  full_txt_fnames))

    txt_fn_list_fn = '{}/{}'.format(tmp_dir, 'txt_fnames.list')
    strutils.dumps('\n'.join(full_txt_fnames), txt_fn_list_fn)

    base_model_fname = '{}_scutclassifier.v{}.pkl'.format(provision, SCUT_CLF_VERSION)

    # Following the logic in the original code.
    eval_status, log_json = eb_runner.custom_train_provision_and_evaluate(txt_fn_list_fn,
                                                                          provision,
                                                                          CUSTOM_MODEL_DIR,
                                                                          base_model_fname,
                                                                          is_doc_structure=True,
                                                                          work_dir=work_dir)
    # copy the result into the expected format for client
    ant_status = eval_status['ant_status']
    cf = ant_status['confusion_matrix']
    status = {'confusion_matrix': [[cf['tn'], cf['fp']], [cf['fn'], cf['tp']]],
              'fscore': ant_status['f1'],
              'precision': ant_status['prec'],
              'recall': ant_status['recall']}

    logging.info("status:")
    pprint(status)
              
    # return some json accuracy info
    status_and_antana = {"stats": status,
                         "antana": log_json}
    # return jsonify(status)
    return jsonify(status_and_antana)
