# take a provision and a CSV file, and train the classifier, saving it to disk

# TODO should we be able to train individual provisions separately

import json
import logging
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

eb_doccat_runner = None
doccat_model_fn = MODEL_DIR + '/ebrevia_docclassifier.pkl'
if os.path.exists(doccat_model_fn):
    eb_doccat_runner = ebrunner.EbDocCatRunner(MODEL_DIR)

eb_langdetect_runner = ebrunner.EbLangDetectRunner()

@app.route('/annotate-doc', methods=['POST'])
def annotate_uploaded_document():

    # verify if the request is for document classification or language detection only
    provisions_st = request.form.get('types')
    provision_set = set(provisions_st.split(',') if provisions_st else [])
    is_classify_doc = request.form.get('classify-doc')
    is_detect_lang = request.form.get('detect-lang')

    ebannotations = {}

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

    # cannot just access the request.files['file'].read() earlier, which
    # make it unavailable to the rest of the code.
    if is_detect_lang:
        atext = strutils.loads(txt_file_name)
        detect_lang = eb_langdetect_runner.detect_lang(atext)
        ebannotations['lang'] = detect_lang
        logging.info("detected language '{}'".format(detect_lang))
        # if no other classification is specified, return early
        if not provision_set and not is_classify_doc:
            return json.dumps(ebannotations)

    if is_classify_doc:
        if eb_doccat_runner != None:
            logging.info("classify document '{}'".format(txt_file_name))
            doc_catnames = eb_doccat_runner.classify_document(txt_file_name)
            ebannotations['tags'] = doc_catnames
        else:
            logging.warning('is_classify_doc is specified, but no models for eb_doccat_runner')
            ebannotations['tags'] = []

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

    # TODO, hacked, remove when not debugging
    # provision_set = set(['date', 'effectivedate', 'party', 'sigdate', 'term', 'title'])

    if pdf_offsets_file_name:  # original file is pdf or word file
        prov_labels_map, doc_text = eb_runner.annotate_pdfboxed_document(txt_file_name,
                                                                         pdf_offsets_file_name,
                                                                         provision_set=provision_set,
                                                                         work_dir=work_dir)
    else:
        # only text file, no offsets.  Original file is .html or .txt
        prov_labels_map, doc_text = eb_runner.annotate_htmled_document(txt_file_name,
                                                                       provision_set=provision_set,
                                                                       work_dir=work_dir)

    ebannotations['ebannotations'] = prov_labels_map
    # pprint(prov_labels_map)
    # pprint(ebannotations)

    return json.dumps(ebannotations)


# pylint: disable=too-many-locals
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

    logging.info("full_txt_fnames (size={}) = {}".format(len(full_txt_fnames),
                                                         full_txt_fnames))

    txt_fn_list_fn = '{}/{}'.format(tmp_dir, 'txt_fnames.list')
    strutils.dumps('\n'.join(full_txt_fnames), txt_fn_list_fn)

    # Following the logic in the original code.
    eval_status = eb_runner.custom_train_provision_and_evaluate(txt_fn_list_fn,
                                                                provision,
                                                                CUSTOM_MODEL_DIR,
                                                                is_doc_structure=False,
                                                                work_dir=work_dir)
    # copy the result into the expected format for client
    pred_status = eval_status['pred_status']['pred_status']
    cf_mtx = pred_status['confusion_matrix']
    status = {'confusion_matrix': [[cf_mtx['tn'], cf_mtx['fp']], [cf_mtx['fn'], cf_mtx['tp']]],
              'fscore': pred_status['f1'],
              'precision': pred_status['prec'],
              'recall': pred_status['recall']}

    logging.info("status: {}".format(status))
              
    # return some json accuracy info
    return jsonify(status)


@app.route('/classify-doc', methods=['POST'])
def classify_uploaded_document():
    """
        Categorize a document.

        :return: returns a list of strings representing document categories.
    """

    request_work_dir = request.form.get('workdir')
    if request_work_dir:
        work_dir = request_work_dir
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
    ebannotations = {}
    if eb_doccat_runner != None:
        logging.info("classify document '{}'".format(file_name))
        doc_catnames = eb_doccat_runner.classify_document(file_name)
        ebannotations['tags'] = doc_catnames
    else:
        logging.warning('is_classify_doc is specified, but no models for eb_doccat_runner')
        ebannotations['tags'] = []

    return json.dumps(ebannotations)


# https://github.com/Mimino666/langdetect
# This language dectect library is a port of Google's language detection
# library.  It supports 55 languages, based on the above link:
# af, ar, bg, bn, ca, cs, cy, da, de, el, en, es, et, fa, fi, fr, gu, he, hi,
# hr, hu, id, it, ja, kn, ko, lt, lv, mk, ml, mr, ne, nl, no, pa, pl, pt, ro,
# ru, sk, sl, so, sq, sv, sw, ta, te, th, tl, tr, uk, ur, vi, zh-cn, zh-tw
@app.route('/detect-lang', methods=['POST'])
def detect_lang():
    """
        Detect a language after read a file from a HTTP POST request.

        :return: returns a string containing the language.
    """
    atext = request.files['file'].read().decode('utf-8')

    detect_lang = eb_langdetect_runner.detect_lang(atext)
    logging.info("detected language '{}'".format(detect_lang))
    return json.dumps({'lang': detect_lang })


@app.route('/detect-langs', methods=['POST'])
def detect_langs():
    """
        Detect top languages and their probabilities after read a file from
        a HTTP POST request.

        :return: returns a string containing comma separated pairs of "lang=prob",
                 i.e., "fi=0.8571380931883487,pl=0.14285995413090066"
    """
    atext = request.files['file'].read().decode('utf-8')

    detect_langs = eb_langdetect_runner.detect_langs(atext)
    logging.info("detected languages '{}'".format(detect_langs))
    return json.dumps({'lang-probs': detect_langs })

