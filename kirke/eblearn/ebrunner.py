# pylint: disable=too-many-lines
from collections import defaultdict
import copy
import concurrent.futures
from datetime import datetime
import json
import logging
import os
import time
# pylint: disable=unused-import
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import langdetect
from langdetect.lang_detect_exception import LangDetectException
from langdetect import DetectorFactory
import psutil

from sklearn.externals import joblib

from kirke.docstruct import fromtomapper, htmltxtparser, pdftxtparser
from kirke.eblearn import annotatorconfig, ebannotator, ebpostproc, ebtrainer, lineannotator
from kirke.eblearn import provclassifier, scutclassifier, spanannotator
from kirke.ebrules import titles, parties, dates
from kirke.utils import ebantdoc4, evalutils, lrucache, modelfileutils, osutils, strutils

from kirke.utils.ebantdoc4 import EbDocFormat, prov_ants_cpoint_to_cunit

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s : %(levelname)s : %(message)s')
# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)
logger.setLevel(logging.INFO)


DEBUG_MODE = False

DOCCAT_MODEL_FILE_NAME = 'ebrevia_docclassifier.v1.pkl'

EBRUN_PROCESS = psutil.Process(os.getpid())
MAX_CUSTOM_MODEL_CACHE_SIZE = 100

# to ensure that langdetect is stable
DetectorFactory.seed = 0


def annotate_provision(eb_annotator,
                       eb_antdoc: ebantdoc4.EbAnnotatedDoc4) -> List[Dict]:
    """
    if isinstance(eb_annotator, spanannotator.SpanAnnotator):
        return eb_annotator.annotate_antdoc(eb_antdoc)
    """
    annotations, _ = eb_annotator.annotate_antdoc(eb_antdoc)
    return annotations


def test_provision(eb_annotator,
                   eb_antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                   threshold) -> Tuple[Dict[str, Dict],
                                       Dict[str, Dict]]:
    print("test_provision, type(eb_annotator) = {}".format(type(eb_annotator)))
    return eb_annotator.test_antdoc_list(eb_antdoc_list, threshold)


def remove_invalid_date_ant_list(date_ant_list: List[Dict]) -> List[Dict]:
    out_list = []  # type: List[Dict]
    for date_dict in date_ant_list:
        ant_text = date_dict['text']
        if ant_text.isdigit() and \
           not dates.is_valid_year(ant_text):
            continue
        out_list.append(date_dict)
    return out_list

# now adjust the date using domain specific logic
# this operation is destructive
def update_dates_by_domain_rules(ant_result_dict):

    # special handling for dates, as in PythonDateOfAgreementClassifier.java
    date_annotations = ant_result_dict.get('date')
    # print('date_annotations date: {}'.format(date_annotations))
    if not date_annotations:
        effectivedate_annotations = ant_result_dict.get('effectivedate', [])
        # print("effectivedate_annotation = {}".format(effectivedate_annotations))
        if effectivedate_annotations:
            # make a copy to preserve original list
            effectivedate_annotations = copy.deepcopy(effectivedate_annotations)
            for eff_ant in effectivedate_annotations:
                eff_ant['label'] = 'date'
            ant_result_dict['date'] = effectivedate_annotations
        else:
            # update 'date' with 'sigdate' if 'date' is empty
            sigdate_annotations = ant_result_dict.get('sigdate', [])
            if not ant_result_dict.get('date') and sigdate_annotations:
                # make a copy to preserve original list
                sigdate_annotations = copy.deepcopy(sigdate_annotations)
                for sig_ant in sigdate_annotations:
                    sig_ant['label'] = 'date'
                ant_result_dict['date'] = sigdate_annotations

    # if 'l_execution_date' is being annotated, replace it with 'date'
    l_execution_dates = ant_result_dict.get('l_execution_date')
    if l_execution_dates is not None:
        l_execution_date_annotations = copy.deepcopy(ant_result_dict.get('date', []))
        for date_ant in l_execution_date_annotations:
            date_ant['label'] = 'l_execution_date'
        ant_result_dict['l_execution_date'] = l_execution_date_annotations


# pylint: disable=too-many-arguments
def assemble_model_base_fnames(provision: str,
                               candidate_types: List[str],
                               next_model_num: int,
                               doc_lang: str,
                               # pylint: disable=invalid-name
                               scut_version: str,
                               # pylint: disable=invalid-name
                               candg_version: str) \
                               -> Tuple[str, str, str]:
    if doc_lang == 'en':
        base_fname_no_ext = '{}.{}'.format(provision,
                                           next_model_num)
    else:
        base_fname_no_ext = '{}.{}_{}'.format(provision,
                                              next_model_num,
                                              doc_lang)

    if len(candidate_types) == 1 and candidate_types[0] == 'SENTENCE':
        base_model_fname = '{}_scutclassifier.v{}.pkl'.format(base_fname_no_ext,
                                                              scut_version)
        base_status_fname = '{}.status'.format(base_fname_no_ext)
        base_result_fname = '{}-ant_result.json'.format(base_fname_no_ext)
    else:
        base_model_fname = \
            '{}_{}_annotator.v{}.pkl'.format(base_fname_no_ext,
                                             "-".join(candidate_types),
                                             candg_version)
        base_status_fname = '{}_{}.status'.format(base_fname_no_ext,
                                                  "-".join(candidate_types))
        base_result_fname = '{}_{}-ant_result.json'.format(base_fname_no_ext,
                                                           "-".join(candidate_types))

    return base_model_fname, base_status_fname, base_result_fname


# pylint: disable=too-many-instance-attributes
class EbRunner:

    # pylint: disable=too-many-locals,too-many-statements, too-many-branches
    def __init__(self, model_dir: str, work_dir: str, custom_model_dir: str) -> None:
        osutils.mkpath(model_dir)
        osutils.mkpath(work_dir)
        osutils.mkpath(custom_model_dir)
        self.model_dir = model_dir
        self.work_dir = work_dir
        self.custom_model_dir = custom_model_dir

        self.provisions = set([])  # type: Set[str]
        # This is for standard models
        # 'Any' should be a KirkeBaseModel
        self.provision_annotator_map = {}  # type: Dict[str, Any]

        # keep track of the timestamp of each custom model to decide to reload
        # model file or not.
        self.custom_model_timestamp_map = {}  # type: Dict[str, datetime]
        # Kirke can have different custom model versions for the same provision right
        # after training new ones. Need to keep them unique by remove the old one.
        # Must syncrhonize the update of self.provision_annotator_map and
        # self.custom_model_fn_map.
        self.custom_model_fn_map = {}  # type: Dict[str, str]
        # This is for custom models
        # pylint: disable=line-too-long
        self.custom_annotator_map = lrucache.LRUCache(MAX_CUSTOM_MODEL_CACHE_SIZE)  # type: lrucache.LRUCache

        orig_mem_usage = EBRUN_PROCESS.memory_info()[0] / 2**20
        logger.info('original memory use: %d Mbytes', orig_mem_usage)
        prev_mem_usage = orig_mem_usage
        num_model = 0

        # load the available classifiers from dir_model
        model_files = modelfileutils.get_model_file_names(model_dir)
        provision_classifier_map = {}
        for model_fn in model_files:
            full_model_fn = '{}/{}'.format(model_dir, model_fn)
            prov_classifier = joblib.load(full_model_fn)

            # before 2019-07-30, models do not have language info
            model_rec = modelfileutils.parse_default_model_file_name(model_fn)
            if model_rec:
                if not hasattr(prov_classifier, 'lang'):
                    prov_classifier.lang = model_rec.lang
                    prov_classifier.transformer.lang = model_rec.lang
            else:
                logger.warning('failed to parse model fname: [%s]', model_fn)
                # a spanannotator in default models.  All default models are in 'en'
                # by default
                prov_classifier.lang = 'en'

            clf_provision = prov_classifier.provision
            if hasattr(prov_classifier, 'version'):
                prov_classifier_version = prov_classifier.version
            else:
                prov_classifier_version = '1.1'
            logger.info("loading #%d: %s, ver=%s, model_fn=%s",
                        num_model, clf_provision, prov_classifier_version, full_model_fn)
            if clf_provision in self.provisions:
                logger.warning("*** WARNING ***  Replacing an existing provision: %s",
                               clf_provision)
            provision_classifier_map[clf_provision] = prov_classifier
            self.provisions.add(clf_provision)
            if DEBUG_MODE:
                # print out memory usage info
                memory_use = EBRUN_PROCESS.memory_info()[0] / 2**20
                # pylint: disable=line-too-long
                print('loading #{} {:<50}, mem = {:.2f}, diff {:.2f}'.format(num_model,
                                                                             full_model_fn,
                                                                             memory_use,
                                                                             memory_use - prev_mem_usage))
                prev_mem_usage = memory_use
            num_model += 1

        for provision in self.provisions:
            pclassifier = provision_classifier_map[provision]
            # in case we want to override
            if isinstance(pclassifier, spanannotator.SpanAnnotator):
                self.provision_annotator_map[provision] = pclassifier
            else:
                prov_threshold = provclassifier.get_provision_threshold(provision)
                self.provision_annotator_map[provision] = \
                    ebannotator.ProvisionAnnotator(pclassifier,
                                                   self.work_dir,
                                                   threshold=prov_threshold)

        self.title_annotator = lineannotator.LineAnnotator('title', titles.TitleAnnotator('title'))
        self.party_annotator = lineannotator.LineAnnotator('party', parties.PartyAnnotator('party'))
        self.date_annotator = lineannotator.LineAnnotator('date', dates.DateAnnotator('date'))

        if num_model == 0:
            logger.error('No model is loaded from %s and %s.', model_dir, custom_model_dir)
            logger.error('Please verify model file names match the filter in modelfileutils.get_model_file_names()')
            return

        total_mem_usage = EBRUN_PROCESS.memory_info()[0] / 2**20
        avg_model_mem = (total_mem_usage - orig_mem_usage) / num_model
        logger.info('total mem: %0.2f,  model mem: %0.2f,  avg: %0.2f',
                    total_mem_usage, total_mem_usage - orig_mem_usage, avg_model_mem)
        logger.info('EbRunner is initiated.')


    def get_provision_annotator(self, provision: str) -> Any:
        """Get the annotator, depending on if it is a bespoke model or standard models."""

        if provision.startswith('cust_'):
            # self.cust_annotator_map is lrucache.LRUCache.  Must use get().
            return self.custom_annotator_map.get(provision)
        # this is where we return all the candidate annotations
        # such as TABLE, DATE, NUMBER, CURRENCY, PERCENT
        if provision in annotatorconfig.get_all_candidate_types():
            config = annotatorconfig.get_ml_annotator_config([provision])
            return spanannotator.SpanAnnotator(provision,
                                               [provision],
                                               nbest=-1,
                                               version=config['version'],
                                               # pylint: disable=line-too-long
                                               doclist_to_antdoc_list=config['doclist_to_antdoc_list'],
                                               is_use_corenlp=config['is_use_corenlp'],
                                               doc_to_candidates=config['doc_to_candidates'],
                                               # pylint: disable=line-too-long
                                               candidate_transformers=config.get('candidate_transformers', []),
                                               # pylint: disable=line-too-long
                                               doc_postproc_list=config.get('doc_postproc_list', []),
                                               pipeline=config['pipeline'],
                                               # pylint: disable=line-too-long
                                               gridsearch_parameters=config['gridsearch_parameters'],
                                               threshold=0.0,
                                               kfold=config.get('kfold', 3))
        return self.provision_annotator_map[provision]


    def run_annotators_in_parallel(self,
                                   eb_antdoc: ebantdoc4.EbAnnotatedDoc4,
                                   lang_provision_set: Optional[Set[str]] = None) \
                                   -> Dict[str, List]:
        doc_lang = eb_antdoc.doc_lang
        if not lang_provision_set:
            lang_provision_set = self.provisions
        #else:
        #    logger.info("user specified lang_provision list: %s", lang_provision_set)
        both_default_custom_provs = set(self.provision_annotator_map.keys())
        both_default_custom_provs.update(self.custom_annotator_map.keys())
        # this is where we add all candidate types, such as TABLE, DATE, NUMBER, CURRENCY, PERCENT
        both_default_custom_provs.update(annotatorconfig.get_all_candidate_types())

        # print('custom_annotator_map.keys() = {}'.format(self.custom_annotator_map.keys()))

        annotations = defaultdict(list)  # type: DefaultDict[str, List]
        # Make sure all the provision's model are there
        prov_annotator_map = {}  # type: Dict[str, Any]
        prov_not_found_list = []  # type: List[str]
        to_remove_lang_provisions = []  # type: List[str]
        for lang_provision in lang_provision_set:
            if lang_provision in both_default_custom_provs:
                prov_annotator_map[lang_provision] = self.get_provision_annotator(lang_provision)
            else:
                if lang_provision.startswith('cust_'):
                    logger.warning('skipping custom model %s because not found.',
                                   lang_provision)
                    logger.warning('custom_model_dir = [%s]', self.custom_model_dir)

                    # there is langid which we created at the end of the lang_provision
                    # add that original provision name back, plus the missing language
                    tmp_prov_name = lang_provision.split('.')[0]
                    annotations[tmp_prov_name] = []
                    to_remove_lang_provisions.append(lang_provision)
                else:
                    prov_not_found_list.append(lang_provision)

        if prov_not_found_list:
            # pylint: disable=line-too-long
            raise Exception("error: Cannot find model file for provisions, {}.".format(prov_not_found_list))

        # remove provisions that have no specific trained language models
        for to_rm_prov in to_remove_lang_provisions:
            lang_provision_set.remove(to_rm_prov)

        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            tmp_prov_annotator = prov_annotator_map[lang_provision]
            logger.info('lang_provision = {}, type(tmp_prov_annotator)={}'.format(lang_provision,
                                                                                 type(tmp_prov_annotator)))
            if hasattr(tmp_prov_annotator, 'lang'):
                logger.info('tmp_prov_annotator.lang = {}'.format(tmp_prov_annotator.lang))
            else:
                tmp_prov_annotator.lang = None
                logger.info('tmp_prov_annotator.lang = None')

            # Remove wrong-language provisions at the last moment
            annotator = prov_annotator_map[lang_provision]
            future_to_provision = {executor.submit(annotate_provision,
                                                   annotator,
                                                   eb_antdoc):
            for future in concurrent.futures.as_completed(future_to_provision):
                lang_provision = future_to_provision[future]
                ant_list = future.result()
                # want to collapse language-specific cust models to one provision

                # Use just provision name instead of lang_provision in
                # the annotation result.
                provision_name = lang_provision
                if 'cust_' in lang_provision and ant_list:
                    provision_name = ant_list[0]['label']

                # modify 'DATE' to 'CAND_DATE'
                # before passing the result back to extractor
                if provision_name == 'DATE':
                    provision_name = 'CAND_DATE'
                    ant_list = update_ant_list_with_provision(ant_list, 'CAND_DATE')

                if '.' in provision_name:  # in case there is no ant_list
                    # remove version, chop off after '.'
                    provision_name = provision_name.split('.')[0]
                # aggregates all annotations across languages for cust models
                annotations[provision_name].extend(ant_list)
        return annotations


    # pylint: disable=invalid-name
    def get_provision_custom_model_files(self, provision: str) -> List[str]:
        """Get the list of files that satisfy the provision."""

        lang_provision_fname_list = \
            modelfileutils.get_provision_custom_model_files(self.custom_model_dir,
                                                            provision)
        result = [fname for unused_lang_provision, fname in lang_provision_fname_list]
        return result


    def update_custom_models(self, lang_provision_set: Set[str]):
        """Update internal data structure to load all custom models, if it is updated.

        lang_provision_set has version and langid specified.

        """
        provision_classifier_map = {}  # this can be either a spanannotator or scutclassifier
        orig_mem_usage = EBRUN_PROCESS.memory_info()[0] / 2**20
        num_model = 0

        cust_lang_provision_set = set([provision for provision in lang_provision_set
                                       if provision.startswith('cust_')])

        start_time_1 = time.time()
        # print('update_custom_models lang= {}, dir={}:'.format(lang, self.custom_model_dir))
        # print('cust_lang_provision_set: {}'.format(cust_lang_provision_set))

        # pylint: disable=line-too-long
        for cust_lang_provision, fname in modelfileutils.get_custom_model_files(self.custom_model_dir,
                                                                                cust_lang_provision_set):

            # print('cust_lang_provision = [{}], fname= [{}]'.format(cust_lang_provision, fname))

            mtime = os.path.getmtime(os.path.join(self.custom_model_dir, fname))
            last_modified_date = datetime.fromtimestamp(mtime)
            old_timestamp = self.custom_model_timestamp_map.get(fname)

            # we already touched the cache, so if the model file is there,
            # it will not be deleted (assume number of provision_set will not
            # flush this out)
            # There is still some possibility that people might delete a version
            # while this is running.  For now, deletion operation should not
            # remove the file yet.
            prov_annotator = self.custom_annotator_map.get(cust_lang_provision)

            is_update_model = False
            if prov_annotator:  # found in cache
                if old_timestamp and old_timestamp == last_modified_date:
                    pass
                else:  # but outdated
                    is_update_model = True
            else:  # not found in cache, load it
                is_update_model = True

            if is_update_model:
                full_custom_model_fn = '{}/{}'.format(self.custom_model_dir, fname)
                prov_classifier = joblib.load(full_custom_model_fn)

                # before 2019-07-30, models do not have language info
                model_rec = modelfileutils.parse_custom_model_file_name(fname)
                if model_rec:
                    if not hasattr(prov_classifier, 'lang'):
                        prov_classifier.lang = model_rec.lang

                        # only sentence candidate type has transformer
                        if hasattr(prov_classifier, 'transformer') and \
                           prov_classifier.transformer is not None:
                            prov_classifier.transformer.lang = model_rec.lang
                else:
                    logger.warning('failed to parse custom model fname: [%s]', fname)

                # if we loaded this for a particular custom field type ("cust_52")
                # it must produce annotations with that label, not with whatever is "embedded"
                # in the saved model file (since the file could have been imported from another
                # server)
                prov_name = cust_lang_provision.split('.')[0]
                logger.info('updating custom provision model to annotate with %s', prov_name)
                # print(prov_classifier)
                logger.info(prov_classifier)
                prov_classifier.provision = prov_name
                if hasattr(prov_classifier, 'transformer') and \
                   prov_classifier.transformer is not None:
                    prov_classifier.transformer.provision = prov_name
                # print("prov_classifier, {}".format(fname))
                # print("type, {}".format(type(prov_classifier)))
                provision_classifier_map[cust_lang_provision] = prov_classifier
                logger.info('update custom model %s, [%s]',
                            cust_lang_provision, full_custom_model_fn)

                #if cust_lang_provision in self.provisions:
                #    logger.warning("*** WARNING ***  Replacing an existing provision: %s",
                #                   cust_lang_provision)

                self.custom_model_timestamp_map[fname] = last_modified_date
                self.custom_model_fn_map[cust_lang_provision] = full_custom_model_fn
                self.provisions.add(cust_lang_provision)
                num_model += 1

        if provision_classifier_map:
            for provision in provision_classifier_map:
                pclassifier = provision_classifier_map[provision]
                # Make sure all xxx_annotators are really annotator, not scut_classifier
                if isinstance(pclassifier, spanannotator.SpanAnnotator):
                    # pylint: disable=line-too-long
                    xxx_annotator = pclassifier  # type: Union[spanannotator.SpanAnnotator, ebannotator.ProvisionAnnotator]
                else:
                    prov_threshold = provclassifier.get_provision_threshold(provision)
                    xxx_annotator = ebannotator.ProvisionAnnotator(pclassifier,
                                                                   self.work_dir,
                                                                   threshold=prov_threshold)
                self.custom_annotator_map.set(provision, xxx_annotator)

            total_mem_usage = EBRUN_PROCESS.memory_info()[0] / 2**20
            avg_model_mem = (total_mem_usage - orig_mem_usage) / num_model
            # pylint: disable=line-too-long
            logger.debug('total mem: %.2f,  model mem: %.2f,  avg: %.2f',
                         total_mem_usage, total_mem_usage - orig_mem_usage, avg_model_mem)
            start_time_2 = time.time()
            logger.info('updating custom models took %.0f msec', (start_time_2 - start_time_1) * 1000)


    # pylint: disable=too-many-arguments
    def annotate_document(self,
                          file_name: str,
                          provision_set: Optional[Set[str]] = None,
                          work_dir: Optional[str] = None,
                          is_doc_structure: bool = True,
                          doc_lang: str = 'en',
                          is_dev_mode: bool = False) \
                          -> Tuple[Dict[str, List],
                                   ebantdoc4.EbAnnotatedDoc4]:
        """Annotate a document with the provisions specified in provision_set.

        'provision_set' is really a set of lang_provision, i.e., 'change_control',
        'cust_12345.9393_pt'
        """
        time1 = time.time()
        if not work_dir:
            work_dir = self.work_dir

        if not provision_set:
            if is_dev_mode:
                # no provision specified.  Must be doing testing.
                lang_provision_set = osutils.get_all_custom_provisions(self.custom_model_dir)
                lang_provision_set.update(self.provisions)
                # also get ALL custom provision set, since we are doing testing
                logger.info("custom_model_dir: %s", self.custom_model_dir)
                logger.info("lang_provision_set: %r", lang_provision_set)
            else:
                logger.warning("annotate_document(%s), provision_set is empty", file_name)
                empty_result_nx = {}  # type: Dict[str, List]
                # this is just to keep the API consistent for now
                # TODO, in the future, maybe change the API to not pass back ebantdoc
                eb_antdoc = ebantdoc4.text_to_ebantdoc(file_name,
                                                       work_dir=work_dir,
                                                       is_doc_structure=is_doc_structure,
                                                       doc_lang=doc_lang)
                return empty_result_nx, eb_antdoc
        else:
            # logger.info('user specified provision list: %s', provision_set)
            lang_provision_set = provision_set

        # replace 'CAND_DATE' with 'DATE'
        # extractor cannot use 'DATE' as provision name
        # because of MySQL conflicts with 'date'
        lang_provision_set = normalize_provision_set(lang_provision_set)

        # update custom models if necessary by checking dir.
        # custom models can be update by other workers
        # print("provision_set: {}".format(provision_set))
        self.update_custom_models(lang_provision_set)

        eb_antdoc = ebantdoc4.text_to_ebantdoc(file_name,
                                               work_dir=work_dir,
                                               doc_lang=doc_lang)

        # if the file contains too few words, don't bother
        # otherwise, might cause classifier error if only have 1 error because of minmax
        if len(eb_antdoc.text) < 100:
            empty_result_2 = {}  # type: Dict[str, List]
            for prov in lang_provision_set:
                # for custom models that has version information
                if '.' in prov:
                    # remove version, chop off after '.'
                    prov = prov.split('.')[0]
                empty_result_2[prov] = []
            # we always return eb_antdoc, not eb_antdoc3
            return empty_result_2, eb_antdoc

        # this execute the annotators in parallel
        prov_labels_map = self.run_annotators_in_parallel(eb_antdoc,
                                                          lang_provision_set=lang_provision_set)
        # this update the 'start_end_span_list' in each antx in-place
        # docutils.update_ants_gap_spans(prov_labels_map, eb_antdoc.gap_span_list, eb_antdoc.text)
        # update prov_labels_map based on rules
        self.apply_line_annotators(prov_labels_map,
                                   eb_antdoc,
                                   work_dir=work_dir)

        prov_labels_map['sigdate'] = remove_invalid_date_ant_list(prov_labels_map.get('sigdate',
                                                                                      []))
        # apply composite date logic
        update_dates_by_domain_rules(prov_labels_map)

        # Up to this point, all annotation's offsets are based on codepoints.
        # Map all offsets to Java's UTF-16 code units.
        # This is a in-place update
        prov_ants_cpoint_to_cunit(prov_labels_map, eb_antdoc.codepoint_to_cunit_mapper)

        # save the prov_labels_map
        prov_ants_fn = file_name.replace('.txt', '.prov.ants.json')
        prov_ants_st = json.dumps(prov_labels_map)
        strutils.dumps(prov_ants_st, prov_ants_fn)

        time2 = time.time()
        logger.info('annotate_document(%s) took %0.2f sec', file_name, (time2 - time1))
        # we always return eb_antdoc, not eb_antdoc3
        return prov_labels_map, eb_antdoc


    def apply_line_annotators(self,
                              prov_labels_map,
                              eb_antdoc,
                              work_dir: str) -> None:
        if eb_antdoc.doc_format == EbDocFormat.pdf:
            # For PDF files, we use *.paraline.txt as the input to lineannotator.
            # We simply redo the whole processing to get the input to lineannotator
            # using htmltxtparser instead of coding the necessary logic again on PDF.
            txt_base_fname = os.path.basename(eb_antdoc.file_id)
            paraline_fname = txt_base_fname.replace('.txt', '.paraline.txt')

            # nlp_paras_with_attrs, nlp_doc_text, unused_gap_span_list, unused_orig_doc_text = \
            html_text_doc = htmltxtparser.parse_document('{}/{}'.format(work_dir,
                                                                        paraline_fname),
                                                         work_dir=work_dir,
                                                         is_combine_line=False)

            origin_sx_lnpos_list, nlp_sx_lnpos_list = \
                fromtomapper.paras_to_fromto_lists(html_text_doc.nlp_paras_with_attrs)

            # there is no offset map because paraline is the same
            self.apply_line_annotators_aux(prov_labels_map,
                                           html_text_doc.nlp_paras_with_attrs,
                                           html_text_doc.get_nlp_text(),
                                           nlp_sx_lnpos_list,
                                           origin_sx_lnpos_list,
                                           eb_antdoc.get_nl_text())
        else:
            self.apply_line_annotators_aux(prov_labels_map,
                                           eb_antdoc.nlp_paras_with_attrs,
                                           eb_antdoc.get_nlp_text(),
                                           eb_antdoc.get_nlp_sx_lnpos_list(),
                                           eb_antdoc.get_origin_sx_lnpos_list(),
                                           eb_antdoc.get_nl_text())


    def apply_line_annotators_aux(self,
                                  prov_labels_map,
                                  paraline_with_attrs,
                                  paraline_text,
                                  paraline_sx_lnpos_list,
                                  origin_sx_lnpos_list,
                                  nl_text: str):

        fromto_mapper = fromtomapper.FromToMapper('an offset mapper',
                                                  paraline_sx_lnpos_list,
                                                  origin_sx_lnpos_list)

        # title works on the nlp_doc_text, not original text. so the
        # offsets needs to be adjusted, just like for text4nlp stuff.
        # The offsets here differs from above because of line break differs.
        # As a result, probably more page numbers are detected correctly and skipped.
        title_ant_list = self.title_annotator.annotate_antdoc(paraline_with_attrs,
                                                              paraline_text,
                                                              fromto_mapper,
                                                              nl_text)
        # we always replace the title using rules
        prov_labels_map['title'] = title_ant_list

        party_ant_list = self.party_annotator.annotate_antdoc(paraline_with_attrs,
                                                              paraline_text,
                                                              fromto_mapper,
                                                              nl_text)
        # if rule found parties, replace it.  Otherwise, keep the old ones
        if party_ant_list:
            prov_labels_map['party'] = party_ant_list
        # comment out all the date code below to disable applying date rule
        date_ant_list = self.date_annotator.annotate_antdoc(paraline_with_attrs,
                                                            paraline_text,
                                                            fromto_mapper,
                                                            nl_text)
        if date_ant_list:
            xx_effective_date_list = []
            xx_date_list = []

            for antx in date_ant_list:
                if antx['label'] == 'effectivedate':
                    xx_effective_date_list.append(antx)
                else:
                    xx_date_list.append(antx)
            if xx_effective_date_list:
                prov_labels_map['effectivedate'] = xx_effective_date_list
                ## replace date IFF classification date is very large
                ## replace the case wehre "1001" is matched as a date, with prob 0.4
                ## This modification is anecdotal, not firmly verified.
                ## this is hacking on the date threshold.
                # ml_date = prov_labels_map.get('date')
                # print("ml_date = {}".format(ml_date))
                # if ml_date and ml_date[0]['prob'] <= 0.5:

                #    # let override later in update_dates_by_domain_rules()
                #    prov_labels_map['date'] = []

                # prov_labels_map['effectivedate'] = xx_effective_date_list
            if xx_date_list:
                prov_labels_map['date'] = xx_date_list


    def test_annotators(self,
                        txt_fns_file_name: str,
                        provision_set: Set[str],
                        threshold: Optional[float] = None) \
                        -> Dict[str, Tuple[Dict[str, Any],
                                           Dict[str, Dict]]]:
        if not provision_set:
            provision_set = self.provisions
        else:
            logger.info('user specified provision list: %s', provision_set)

        # in reality, we only use 1 provision
        ebantdoc_list = []  # type: List[ebantdoc4.EbAnnotatedDoc4]
        if len(provision_set) == 1:
            provision = list(provision_set)[0]
            annotator2 = self.provision_annotator_map[provision]
            if isinstance(annotator2, spanannotator.SpanAnnotator):
                ebantdoc_list = \
                    ebantdoc4.doclist_to_ebantdoc_list(txt_fns_file_name,
                                                       self.work_dir,
                                                       # for TABLE, is_use_corenlp == True
                                                       # all other are False
                                                       is_use_corenlp=annotator2.is_use_corenlp)
            else:
                ebantdoc_list = ebantdoc4.doclist_to_ebantdoc_list(txt_fns_file_name,
                                                                   self.work_dir)
        else:
            ebantdoc_list = ebantdoc4.doclist_to_ebantdoc_list(txt_fns_file_name,
                                                               self.work_dir)

        prov_antlist_logjson_map = {}
        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            future_to_provision = {executor.submit(test_provision,
                                                   self.provision_annotator_map[provision],
                                                   ebantdoc_list,
                                                   threshold):
                                   provision for provision in provision_set}
            for future in concurrent.futures.as_completed(future_to_provision):
                provision = future_to_provision[future]
                ant_status, log_json = future.result()
                prov_antlist_logjson_map[provision] = (ant_status, log_json)

        return prov_antlist_logjson_map

    # pylint: disable=C0103
    def custom_train_provision_and_evaluate(self,
                                            txt_fn_list,
                                            provision,
                                            custom_model_dir,
                                            base_model_fname,
                                            base_status_fname,
                                            base_result_fname,
                                            candidate_types: List[str],
                                            nbest: int,
                                            work_dir=None,
                                            doc_lang="en") \
                                            -> Tuple[Dict[str, Any], Dict[str, Dict]]:
        logger.info("txt_fn_list_fn: %s", txt_fn_list)
        if not work_dir:
            work_dir = self.work_dir
        model_file_name = '{}/{}'.format(custom_model_dir, base_model_fname)
        model_status_fname = '{}/{}'.format(custom_model_dir, base_status_fname)
        model_result_fname = '{}/{}'.format(custom_model_dir, base_result_fname)
        logger.info("custom_model_file: %s", model_file_name)
        # logger.info("custom_model_status_file: %s", model_status_fname)
        # logger.info("custom_model_result_file: %s", model_result_fname)

        # SENTENCE runs the standard pipeline, if specified candidate type run candidate generation
        if len(candidate_types) == 1 and candidate_types[0] == 'SENTENCE':
            eb_classifier = scutclassifier.ShortcutClassifier(provision)
            # It is know that 'eb_annotator' is ProvisionAnnotator, mypy.
            # Conflicts with below.
            # Please note, for 'sentence', we use is_doc_structure=True
            eb_annotator_scut, log_json = \
                ebtrainer.train_eval_annotator(provision,
                                               doc_lang,
                                               nbest,
                                               txt_fn_list,
                                               work_dir,
                                               custom_model_dir,
                                               model_file_name=model_file_name,
                                               model_status_fname=model_status_fname,
                                               model_result_fname=model_result_fname,
                                               eb_classifier=eb_classifier,
                                               is_doc_structure=True,
                                               # pylint: disable=line-too-long
                                               is_bespoke_mode=True)  # type: Tuple[Optional[ebannotator.ProvisionAnnotator], Dict[str, Any]]
            if eb_annotator_scut:
                return eb_annotator_scut.get_eval_status(), log_json

            # eb_annotator_span == None, or the training failed
            # return the error message stored in in log_json_span
            return log_json, {}

        # Please note, for 'non-sentence', we use is_doc_structure=False
        eb_annotator_span, log_json_span = \
            ebtrainer.train_eval_span_annotator(provision,
                                                doc_lang,
                                                nbest,
                                                candidate_types,
                                                work_dir,
                                                custom_model_dir,
                                                model_file_name=model_file_name,
                                                model_status_fname=model_status_fname,
                                                model_result_fname=model_result_fname,
                                                txt_fn_list=txt_fn_list,
                                                is_doc_structure=False,
                                                # pylint: disable=line-too-long
                                                is_bespoke_mode=True)  # type: Tuple[Optional[spanannotator.SpanAnnotator], Dict[str, Any]]

        if eb_annotator_span:
            return eb_annotator_span.get_eval_status(), log_json_span

        # eb_annotator_span == None, or the training failed
        # return the error message stored in in log_json_span
        return log_json_span, {}


    # this function is here because it is a combination of both ML and rule-based annotator
    # pylint: disable=invalid-name
    def eval_mlxline_annotator(self,
                               provision: str,
                               test_doclist_fn: str,
                               work_dir: str = 'dir-work') -> Dict:
        # test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
        num_test_doc = 0
        tp, fn, fp, tn = 0, 0, 0, 0

        # need the ML threshold for evalutils.calc_doc_ant_confusion_mnatrix()
        threshold = self.provision_annotator_map[provision].threshold

        with open(test_doclist_fn, 'rt') as testin:
            for test_fn in testin:
                num_test_doc += 1
                test_fn = test_fn.strip()
                prov_labels_map, eb_antdoc = self.annotate_document(test_fn,
                                                                    provision_set=set([provision]),
                                                                    work_dir=work_dir)
                ant_list = prov_labels_map.get(provision, [])

                # print("\ntest_fn = {}".format(test_fn))
                # print("ant_list: {}".format(ant_list))

                print('ebantdoc.fileid = {}'.format(eb_antdoc.file_id))
                # print("ant_list: {}".format(ant_list))
                prov_human_ant_list = [hant for hant in eb_antdoc.prov_annotation_list
                                       if hant.label == provision]

                ant_list = self.recover_false_negatives(prov_human_ant_list,
                                                        eb_antdoc.get_text(),
                                                        provision,
                                                        ant_list)

                # print("\nfn: {}".format(ebantdoc.file_id))
                # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
                # pred_prob_start_end_list, txt)
                # currently, PROVISION_EVAL_ANYMATCH_SET only has 'title', not 'party' or 'date'
                if provision in ebannotator.PROVISION_EVAL_ANYMATCH_SET:
                    xtp, xfn, xfp, xtn, unused_json_log = \
                        evalutils.calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list,
                                                                         ant_list,
                                                                         eb_antdoc.file_id,
                                                                         eb_antdoc.get_text())
                else:
                    xtp, xfn, xfp, xtn, _, unused_json_log = \
                        evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                                ant_list,
                                                                eb_antdoc.file_id,
                                                                eb_antdoc.get_text(),
                                                                threshold,
                                                                is_raw_mode=False)
                tp += xtp
                fn += xfn
                fp += xfp
                tn += xtn

        title = 'annotate_status'
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)

        tmp_eval_status = {'ant_status': {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                               'fn': fn, 'tp': tp},
                                          'prec': prec, 'recall': recall, 'f1': f1}}

        print("len({}) = {}".format(test_doclist_fn, num_test_doc))

        return tmp_eval_status


    # This function is here because it is a combination of both ML and rule-based annotator
    # It is only used for debugging, in main.py
    # pylint: disable=invalid-name
    def eval_span_annotator(self,
                            provision: str,
                            doc_lang: str,
                            candidate_types: List[str],
                            test_doclist_fn: str,
                            work_dir: str = 'dir-work') -> Dict:
        # test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
        num_test_doc = 0
        tp, fn, fp, tn = 0, 0, 0, 0

        base_model_fname, unused_base_status_fname, unused_base_result_fname = \
            spanannotator.get_model_base_fnames(provision,
                                                doc_lang=doc_lang,
                                                candidate_types=candidate_types)
        model_file_name = '{}/{}'.format(self.custom_model_dir, base_model_fname)

        prov_model = joblib.load(model_file_name)
        self.provision_annotator_map[provision] = prov_model
        threshold = prov_model.threshold

        with open(test_doclist_fn, 'rt') as testin:
            for test_fn in testin:
                num_test_doc += 1
                test_fn = test_fn.strip()
                prov_labels_map, ebantdoc = self.annotate_document(test_fn,
                                                                   provision_set=set([provision]),
                                                                   work_dir=work_dir)
                ant_list = prov_labels_map.get(provision, [])

                # print("\ntest_fn = {}".format(test_fn))
                # print("ant_list: {}".format(ant_list))

                print('ebantdoc.fileid = {}'.format(ebantdoc.file_id))
                # print("ant_list: {}".format(ant_list))
                prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                                       if hant.label == provision]

                # print("\nfn: {}".format(ebantdoc.file_id))
                # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
                # pred_prob_start_end_list, txt)
                # currently, PROVISION_EVAL_ANYMATCH_SET only has 'title', not 'party' or 'date'
                if provision in ebannotator.PROVISION_EVAL_ANYMATCH_SET:
                    xtp, xfn, xfp, xtn, unused_json_log = \
                        evalutils.calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list,
                                                                         ant_list,
                                                                         ebantdoc.file_id,
                                                                         ebantdoc.get_text())
                else:
                    xtp, xfn, xfp, xtn, _, unused_json_log = \
                        evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                                ant_list,
                                                                ebantdoc.file_id,
                                                                ebantdoc.get_text(),
                                                                threshold,
                                                                is_raw_mode=False)
                tp += xtp
                fn += xfn
                fp += xfp
                tn += xtn

        title = 'annotate_status'
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)

        tmp_eval_status = {'ant_status': {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                               'fn': fn, 'tp': tp},
                                          'prec': prec, 'recall': recall, 'f1': f1}}

        print("len({}) = {}".format(test_doclist_fn, num_test_doc))

        return tmp_eval_status

    # pylint: disable=no-self-use
    def recover_false_negatives(self, prov_human_ant_list, doc_text, provision, ant_result):
        if not prov_human_ant_list:
            return ant_result
        for ant in prov_human_ant_list:
            if not evalutils.find_annotation_overlap_x2(ant.start, ant.end, ant_result):
                clean_text = strutils.sub_nltab_with_space(doc_text[ant.start:ant.end])
                fn_ant = ebpostproc.to_ant_result_dict(label=provision,
                                                       prob=0.0,
                                                       start=ant.start,
                                                       end=ant.end,
                                                       text=clean_text)
                ant_result.append(fn_ant)
        return ant_result


# pylint: disable=too-few-public-methods
class EbDocCatRunner:

    def __init__(self, model_dir: str) -> None:
        osutils.mkpath(model_dir)
        self.model_dir = model_dir

        # load the available classifiers from dir_model
        full_model_fn = '{}/{}'.format(self.model_dir, DOCCAT_MODEL_FILE_NAME)
        logger.info("model_fn = [%s]", full_model_fn)

        if os.path.exists(full_model_fn):
            self.doc_classifier = joblib.load(full_model_fn)
            logger.info("EbDocCatRunner loading %s, %s", full_model_fn,
                        str(self.doc_classifier.catname_list))
            self.is_initialized = True
        else:
            logger.info("EbDocCatRunner not running because %s is missing.", full_model_fn)
            self.is_initialized = False

    def classify_document(self, fname: str) -> List[str]:
        # logger.info("classifying document: '{}'".format(fname))
        with open(fname, 'rt') as fin:
            doc_text = fin.read()
            return self.doc_classifier.predict(doc_text)


class EbLangDetectRunner:

    def __init__(self) -> None:
        pass

    # pylint: disable=no-self-use
    def detect_lang(self, atext: str) -> Optional[str]:
        try:
            detect_lang = langdetect.detect(atext.lower())
            # Normalize Chinese lang names because
            # our existing model name convention expects
            # 2 letters for a language, except for English.
            #
            # CoreNLP only uses 'zh'.
            if detect_lang == 'zh-cn' or \
               detect_lang == 'zh-tw':
                detect_lang = 'zh'
        except LangDetectException:
            detect_lang = None
        # logger.info("detected language '{}'".format(detect_lang))
        return detect_lang

    # pylint: disable=no-self-use
    def detect_langs(self, atext: str) -> str:
        try:
            lang_probs = langdetect.detect_langs(atext.lower())
            if lang_probs is None:
                return ''
            detect_langs = ','.join(['{}={}'.format(lang.lang, lang.prob) for lang in lang_probs])
        except LangDetectException:
            detect_langs = ''
        # logger.info("detected languages '{}'".format(detect_langs))
        return detect_langs


def normalize_provision_set(cands: Set[str]):
    if 'CAND_DATE' in cands:
        cands.remove('CAND_DATE')
        cands.add('DATE')
    return cands


# this is in-place update
def update_ant_list_with_provision(alist: List[Dict],
                                   provision: str) \
                                   -> List[Dict]:
    for adict in alist:
        adict['label'] = provision
    return alist
