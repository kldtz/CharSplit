
from collections import defaultdict
import configparser
import copy
import json
import logging
import os.path
# pylint: disable=unused-import
from typing import DefaultDict, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify

from kirke.eblearn import ebrunner
from kirke.utils import osutils, strutils

# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

# NOTE: Remove the following line to get rid of all logging messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')

# pylint: disable=invalid-name
app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True
# app.debug = True
EB_FILES = os.environ['EB_FILES']
EB_MODELS = os.environ['EB_MODELS']
logging.info("eb files is [%s]", EB_FILES)
logging.info("eb models is [%s]", EB_MODELS)

SCUT_CLF_VERSION = config['ebrevia.com']['SCUT_CLF_VERSION']

# classifiers
WORK_DIR = 'data-from-web'
MODEL_DIR = EB_MODELS
CUSTOM_MODEL_DIR = EB_FILES + 'pymodel'
osutils.mkpath(WORK_DIR)
osutils.mkpath(MODEL_DIR)
osutils.mkpath(CUSTOM_MODEL_DIR)

eb_runner = ebrunner.EbRunner(MODEL_DIR, WORK_DIR, CUSTOM_MODEL_DIR)

# pylint: disable=invalid-name
eb_doccat_runner = ebrunner.EbDocCatRunner(MODEL_DIR)
# pylint: disable=invalid-name
eb_langdetect_runner = ebrunner.EbLangDetectRunner()

if not eb_runner:
    logging.error('problem initializing ebrunner')
    exit(1)

@app.route('/annotate-doc', methods=['POST'])
# pylint: disable=too-many-locals, too-many-branches, too-many-statements
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

    fn_list = request.files.getlist('file')
    for fstorage in fn_list:
        fn = '{}/{}'.format(work_dir, fstorage.filename)
        print("saving file '{}'".format(fn))
        fstorage.save(fn)

        if fn.endswith('.offsets.json'):
            # pdf_offsets_file_name = fn
            pass
        elif fn.endswith('.txt'):
            txt_file_name = fn

            # save the file name
            meta_fn = '{}/{}'.format(work_dir,
                                     fstorage.filename.replace('.txt', '.meta'))
            # print("wrote meta_fn: '{}'".format(meta_fn))
            # print("doc title: '{}'".format(file_title))
            with open(meta_fn, 'wt') as meta_out:
                print('pdf_file\t{}'.format(file_title), file=meta_out)
                print('txt_file\t{}'.format(fstorage.filename), file=meta_out)
        else:
            logging.warning('unknown file extension in annotate_uploaded_document(%s)', fn)

    # cannot just access the request.files['file'].read() earlier, which
    # make it unavailable to the rest of the code.

    atext = strutils.loads(txt_file_name)
    doc_lang = eb_langdetect_runner.detect_lang(atext)
    logging.info("detected language '%s'", doc_lang)
    if doc_lang is None:
        ebannotations['lang'] = doc_lang
        ebannotations['ebannotations'] = {}
        ebannotations['tags'] = []
        return json.dumps(ebannotations)
    if is_detect_lang:
        ebannotations['lang'] = doc_lang

    # if no other classification is specified, return early
    if not provision_set and not is_classify_doc:
        return json.dumps(ebannotations)

    if is_classify_doc:
        if eb_doccat_runner.is_initialized:
            logging.info("classify document '%s'", txt_file_name)
            doc_catnames = eb_doccat_runner.classify_document(txt_file_name)
            ebannotations['tags'] = doc_catnames
        else:
            logging.warning('is_classify_doc is True, but no model exists for eb_doccat_runner')
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

    provision_set = [x + "_" + doc_lang if ("cust_" in x and doc_lang != "en") else x
                     for x in provision_set]
    # provision_set = set(['date', 'effectivedate', 'party', 'sigdate', 'term', 'title'])
    prov_labels_map, _ = eb_runner.annotate_document(txt_file_name,
                                                     provision_set=provision_set,
                                                     work_dir=work_dir,
                                                     doc_lang=doc_lang)

    # because special case of 'effectivdate_auto'
    if prov_labels_map.get('effectivedate'):
        effectivedate_annotations = copy.deepcopy(prov_labels_map.get('effectivedate', []))
        for eff_ant in effectivedate_annotations:
            eff_ant['label'] = 'effectivedate_auto'
        prov_labels_map['effectivedate_auto'] = effectivedate_annotations
        del prov_labels_map['effectivedate']

    ebannotations['ebannotations'] = prov_labels_map
    return json.dumps(ebannotations)


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
@app.route('/custom-train/<cust_id>', methods=['POST'])
def custom_train(cust_id: str):
    request_work_dir = request.form.get('workdir')
    if request_work_dir:
        work_dir = request_work_dir
        logging.info("work_dir = '%s'", work_dir)
        osutils.mkpath(work_dir)
    else:
        work_dir = WORK_DIR

    candidate_type = request.form.get('candidate_type')
    candidate_type = 'CURRENCY' ####DELETE LATER
    if not candidate_type:
        candidate_type = 'SENTENCE'

    # to ensure that no accidental file name overlap
    logging.info("cust_id = '%s'", cust_id)
    provision = 'cust_{}'.format(cust_id)
    tmp_dir = '{}/{}'.format(work_dir, provision)
    osutils.mkpath(tmp_dir)
    fn_list = request.files.getlist('file')

    # save all the uploaded files in a location
    for fstorage in fn_list:
        fn = '{}/{}'.format(tmp_dir, fstorage.filename)
        # print("saving file '{}'".format(fn))
        fstorage.save(fn)

    fname_provtypes_map = {}
    txt_fnames = []
    # dict of lang, with list of file in that lang
    full_txt_fnames = defaultdict(list)  # type: DefaultDict[str, List[str]]
    txt_offsets_fn_map = {}
    for name in [fstorage.filename for fstorage in fn_list]:
        file_id = name.split('.')[0]
        full_path = '{}/{}'.format(tmp_dir, name)
        if name.endswith('.ant'):
            ants_map = json.loads(strutils.loads(full_path))
            ants = [x['type'] for x in ants_map]
            fname_provtypes_map[file_id] = ants
        elif name.endswith('.txt'):
            txt_fnames.append(name)
            atext = strutils.loads(full_path)
            doc_lang = eb_langdetect_runner.detect_lang(atext)
            full_txt_fnames[doc_lang].append(file_id)
        elif name.endswith('.offsets.json'):
            # create txt -> offsets.json map in order to do sent4nlp processing
            tmp_txt_fn = name.replace(".offsets.json", ".txt")
            txt_offsets_fn_map[tmp_txt_fn] = name
        else:
            logging.warning('unknown file extension in custom_train(%s)', fn)

    #logging.info("full_txt_fnames (size={}) = {}".format(len(full_txt_fnames), full_txt_fnames))
    all_stats = {}
    for doc_lang, names_per_lang in full_txt_fnames.items():
        if not doc_lang:  # if a document has no text, its langid can be None
            continue
        ant_count = sum([fname_provtypes_map[x].count(provision) for x in names_per_lang])
        logging.info('Number of annotations for %s: %d', doc_lang, ant_count)
        if ant_count >= 6:
            txt_fn_list_fn = '{}/{}'.format(tmp_dir, 'txt_fnames_{}.list'.format(doc_lang))
            fnames_paths = ['{}/{}.txt'.format(tmp_dir, x) for x in names_per_lang]
            strutils.dumps('\n'.join(fnames_paths), txt_fn_list_fn)
            if candidate_type == 'SENTENCE':
                base_model_fname = '{}_scutclassifier.v{}.pkl'.format(provision, SCUT_CLF_VERSION)
                if doc_lang != "en":
                    base_model_fname = '{}_{}_scutclassifier.v{}.pkl'.format(provision,
                                                                             doc_lang,
                                                                             SCUT_CLF_VERSION)
            else:
                base_model_fname = '{}_{}_annotator.v{}.pkl'.format(provision, candidate_type, SCUT_CLF_VERSION)
                if doc_lang != "en":
                    base_model_fname = '{}_{}_{}_annotator.v{}.pkl'.format(provision,
                                                                           doc_lang,
                                                                           candidate_type,
                                                                           SCUT_CLF_VERSION)

            # Following the logic in the original code.
            eval_status, _ = \
                eb_runner.custom_train_provision_and_evaluate(txt_fn_list_fn,
                                                              provision,
                                                              CUSTOM_MODEL_DIR,
                                                              base_model_fname,
                                                              candidate_type,
                                                              is_doc_structure=True,
                                                              work_dir=work_dir,
                                                              doc_lang=doc_lang)
            # copy the result into the expected format for client
            ant_status = eval_status['ant_status']
            cf = ant_status['confusion_matrix']
            status = {'confusion_matrix': [[cf['tn'], cf['fp']], [cf['fn'], cf['tp']]],
                      'fscore': ant_status['f1'],
                      'precision': ant_status['prec'],
                      'recall': ant_status['recall']}

            logging.info("status: %s", str(status))

            # return some json accuracy info
            all_stats[doc_lang] = status
        else:
            all_stats[doc_lang] = {'confusion_matrix': [[]],
                                   'fscore': -1.0,
                                   'precision': -1.0,
                                   'recall': -1.0}
    return jsonify(all_stats)


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

    detect_lang_x = eb_langdetect_runner.detect_lang(atext)
    logging.info("detected language '%s'", detect_lang_x)
    return json.dumps({'lang': detect_lang_x})


@app.route('/detect-langs', methods=['POST'])
def detect_langs():
    """
        Detect top languages and their probabilities after read a file from
        a HTTP POST request.

        :return: returns a string containing comma separated pairs of "lang=prob",
                 i.e., "fi=0.8571380931883487,pl=0.14285995413090066"
    """
    atext = request.files['file'].read().decode('utf-8')

    detect_langs_x = eb_langdetect_runner.detect_langs(atext)
    logging.info("detected languages '%s'", detect_langs_x)
    return json.dumps({'lang-probs': detect_langs_x})
