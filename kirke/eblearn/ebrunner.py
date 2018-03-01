from collections import defaultdict
import copy
import concurrent.futures
from datetime import datetime
import json
import logging
import os
import re
import time
# pylint: disable=unused-import
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple, Union

import langdetect
from langdetect.lang_detect_exception import LangDetectException
import psutil

from sklearn.externals import joblib

from kirke.docstruct import fromtomapper, htmltxtparser
from kirke.eblearn import (ebannotator, ebtrainer, lineannotator, provclassifier,
                           scutclassifier, spanannotator)
from kirke.ebrules import titles, parties, dates
from kirke.utils import osutils, strutils, evalutils, ebantdoc2, ebantdoc3

from kirke.utils.ebantdoc2 import EbDocFormat, prov_ants_cpoint_to_cunit

DEBUG_MODE = False

DOCCAT_MODEL_FILE_NAME = 'ebrevia_docclassifier.v1.pkl'

EBRUN_PROCESS = psutil.Process(os.getpid())


def annotate_provision(eb_annotator,
                       eb_antdoc: ebantdoc2.EbAnnotatedDoc2,
                       eb_antdoc3: ebantdoc3.EbAnnotatedDoc3) -> Tuple[List[Dict], float]:
    if isinstance(eb_annotator, spanannotator.SpanAnnotator):
        return eb_annotator.annotate_antdoc(eb_antdoc3)

    return eb_annotator.annotate_antdoc(eb_antdoc)


def test_provision(eb_annotator,
                   eb_antdoc_list,
                   threshold) -> Tuple[Dict[str, Dict],
                                       Dict[str, Dict]]:
    print("test_provision, type(eb_annotator) = {}".format(type(eb_annotator)))
    return eb_annotator.test_antdoc_list(eb_antdoc_list, threshold)


# now adjust the date using domain specific logic
# this operation is destructive
def update_dates_by_domain_rules(ant_result_dict):

    # special handling for dates, as in PythonDateOfAgreementClassifier.java
    date_annotations = ant_result_dict.get('date')
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
            sigdate_annotations = ant_result_dict.get('sigdate')
            if sigdate_annotations:
                # make a copy to preserve original list
                sigdate_annotations = copy.deepcopy(sigdate_annotations)
                for sig_ant in sigdate_annotations:
                    sig_ant['label'] = 'date'
                ant_result_dict['date'] = sigdate_annotations
    # user never want to see sigdate
    ant_result_dict['sigdate'] = []

    # if 'l_execution_date' is being annotated, replace it with 'date'
    l_execution_dates = ant_result_dict.get('l_execution_date')
    if l_execution_dates is not None:
        l_execution_date_annotations = copy.deepcopy(ant_result_dict.get('date', []))
        for date_ant in l_execution_date_annotations:
            date_ant['label'] = 'l_execution_date'
        ant_result_dict['l_execution_date'] = l_execution_date_annotations


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
        self.custom_model_timestamp_map = {}  # type: Dict[str, datetime]
        # Kirke can have different custom model versions for the same provision right
        # after training new ones. Need to keep them unique by remove the old one.
        # Must syncrhonize the update of self.provision_annotator_map and
        # self.provision_custom_model_fn_map.
        self.provision_custom_model_fn_map = {}  # type: Dict[str, str]

        # load the available classifiers from dir_model
        model_files = osutils.get_model_files(model_dir)
        provision_classifier_map = {}
        self.provisions = set([])  # type: Set[str]
        # pylint: disable=line-too-long
        self.provision_annotator_map = {}  # type: Dict[str, Union[ebannotator.ProvisionAnnotator, spanannotator.SpanAnnotator]]

        candg_model_files = osutils.get_candg_model_files(model_dir)

        orig_mem_usage = EBRUN_PROCESS.memory_info()[0] / 2**20
        logging.info('original memory use: %d Mbytes', orig_mem_usage)
        prev_mem_usage = orig_mem_usage
        num_model = 0

        for model_fn in model_files:
            full_model_fn = '{}/{}'.format(model_dir, model_fn)
            prov_classifier = joblib.load(full_model_fn)
            clf_provision = prov_classifier.provision
            if hasattr(prov_classifier, 'version'):
                prov_classifier_version = prov_classifier.version
            else:
                prov_classifier_version = '1.1'
            logging.info("ebrunner loading #%d: %s, ver=%s, model_fn=%s",
                         num_model, clf_provision, prov_classifier_version, full_model_fn)
            if clf_provision in self.provisions:
                logging.warning("*** WARNING ***  Replacing an existing provision: %s",
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

        # after loading regular models, now load candidate-generation models
        for model_fn in candg_model_files:
            full_model_fn = '{}/{}'.format(model_dir, model_fn)
            # this is spanannotator.SpanAnnotator
            prov_classifier = joblib.load(full_model_fn)
            clf_provision = prov_classifier.provision
            if hasattr(prov_classifier, 'version'):
                prov_classifier_version = prov_classifier.version
            else:
                prov_classifier_version = '0.9'
            logging.info("ebrunner loading candg #%d: %s, ver=%s, model_fn=%s",
                         num_model, clf_provision, prov_classifier_version, full_model_fn)
            if clf_provision in self.provisions:
                logging.warning("*** WARNING ***  Replacing an existing provision: %s",
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

        custom_model_files = osutils.get_model_files(custom_model_dir)
        for custom_model_fn in custom_model_files:
            # record the timestamp for update if needed in the future
            mtime = os.path.getmtime(os.path.join(self.custom_model_dir, custom_model_fn))
            last_modified_date = datetime.fromtimestamp(mtime)
            self.custom_model_timestamp_map[custom_model_fn] = last_modified_date

            full_custom_model_fn = '{}/{}'.format(custom_model_dir, custom_model_fn)
            prov_classifier = joblib.load(full_custom_model_fn)
            cust_id = re.match(r'(cust_\d+_\w\w)_scutclassifier', custom_model_fn)
            if cust_id:
                clf_provision = cust_id.group(1)
            else:
                clf_provision = prov_classifier.provision
            if hasattr(prov_classifier, 'version'):
                prov_classifier_version = prov_classifier.version
            else:
                prov_classifier_version = '1.1'
            self.provision_custom_model_fn_map[clf_provision] = full_custom_model_fn
            logging.info("ebrunner loading custom #%d: %s, ver=%s, model_fn=%s",
                         num_model, clf_provision, prov_classifier_version, full_custom_model_fn)
            if clf_provision in self.provisions:
                logging.warning("*** WARNING ***  Replacing an existing provision: %s",
                                clf_provision)
            provision_classifier_map[clf_provision] = prov_classifier
            self.provisions.add(clf_provision)
            if DEBUG_MODE:
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
            prov_threshold = provclassifier.get_provision_threshold(provision)
            if isinstance(pclassifier, spanannotator.SpanAnnotator):
                self.provision_annotator_map[provision] = pclassifier
            else:
                self.provision_annotator_map[provision] = \
                    ebannotator.ProvisionAnnotator(pclassifier,
                                                   self.work_dir,
                                                   threshold=prov_threshold)

        self.title_annotator = lineannotator.LineAnnotator('title', titles.TitleAnnotator('title'))
        self.party_annotator = lineannotator.LineAnnotator('party', parties.PartyAnnotator('party'))
        self.date_annotator = lineannotator.LineAnnotator('date', dates.DateAnnotator('date'))

        if num_model == 0:
            logging.error('No model is loaded from %s and %s.',
                          model_dir, custom_model_dir)
            logging.error('Please also verify model file names match the filter in osutils.get_model_files()')
            return

        total_mem_usage = EBRUN_PROCESS.memory_info()[0] / 2**20
        avg_model_mem = (total_mem_usage - orig_mem_usage) / num_model
        logging.info('total mem: %0.2f,  model mem: %0.2f,  avg: %0.2f',
                     total_mem_usage, total_mem_usage - orig_mem_usage, avg_model_mem)
        logging.info('EbRunner is initiated.')

    def run_annotators_in_parallel(self,
                                   eb_antdoc: ebantdoc2.EbAnnotatedDoc2,
                                   eb_antdoc3: ebantdoc3.EbAnnotatedDoc3,
                                   provision_set=None) -> Dict[str, List]:
        if not provision_set:
            provision_set = self.provisions
        #else:
        #    logging.info("user specified provision list: %s", provision_set)
        prov_antlist_map = defaultdict(list)  # type: DefaultDict[str, List[Dict]]
        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            future_to_provision = {executor.submit(annotate_provision,
                                                   self.provision_annotator_map[provision],
                                                   eb_antdoc,
                                                   eb_antdoc3):
                                   provision for provision in provision_set
                                   if provision in self.provision_annotator_map.keys()}
            for future in concurrent.futures.as_completed(future_to_provision):
                provision = future_to_provision[future]
                ant_list, unused_threshold = future.result()
                # want to collapse language-specific cust models to one provision
                if 'cust_' in provision and ant_list:
                    provision = ant_list[0]['label']
                # aggregates all prov_antlist_map across languages for cust models
                prov_antlist_map[provision].extend(ant_list)
        return prov_antlist_map

    def update_custom_models(self) -> None:
        provision_classifier_map = {}
        orig_mem_usage = EBRUN_PROCESS.memory_info()[0] / 2**20
        num_model = 0

        start_time_1 = time.time()
        for fname in osutils.get_model_files(self.custom_model_dir):
            mtime = os.path.getmtime(os.path.join(self.custom_model_dir, fname))
            last_modified_date = datetime.fromtimestamp(mtime)
            # print("hi {} {}".format(fn, last_modified_date))

            old_timestamp = self.custom_model_timestamp_map.get(fname)
            if old_timestamp and old_timestamp == last_modified_date:
                pass
            else:
                # for both out-of-date models and new models
                # prev_mem_usage = EBRUN_PROCESS.memory_info()[0] / 2**20
                # logging.info('current memory use: {} Mbytes'.format(prev_mem_usage))

                full_custom_model_fn = '{}/{}'.format(self.custom_model_dir, fname)
                prov_classifier = joblib.load(full_custom_model_fn)
                cust_id = re.match(r'(cust_\d+_\w\w)_scutclassifier', fname)
                if cust_id:
                    clf_provision = cust_id.group(1)
                else:
                    clf_provision = prov_classifier.provision

                #if clf_provision in self.provisions:
                #    logging.warning("*** WARNING ***  Replacing an existing provision: %s",
                #                    clf_provision)
                provision_classifier_map[clf_provision] = prov_classifier
                self.custom_model_timestamp_map[fname] = last_modified_date
                self.provisions.add(clf_provision)
                num_model += 1

                prev_custom_model_fn = self.provision_custom_model_fn_map.get(clf_provision)
                if not prev_custom_model_fn:  # doesnt exist before
                    self.provision_custom_model_fn_map[clf_provision] = full_custom_model_fn
                elif prev_custom_model_fn != full_custom_model_fn:  # must exist before
                    # check for any file name change due to version change
                    self.update_existing_provision_fn_map_aux(clf_provision, full_custom_model_fn)
                # if the same, don't do anything

        if provision_classifier_map:
            for provision in provision_classifier_map:
                logging.info("updating annotator: %s", provision)
                pclassifier = provision_classifier_map[provision]
                self.provision_annotator_map[provision] = \
                    ebannotator.ProvisionAnnotator(pclassifier, self.work_dir)

            total_mem_usage = EBRUN_PROCESS.memory_info()[0] / 2**20
            avg_model_mem = (total_mem_usage - orig_mem_usage) / num_model
            # pylint: disable=line-too-long
            logging.info('total mem: %.2f,  model mem: %.2f,  avg: %.2f',
                         total_mem_usage, total_mem_usage - orig_mem_usage, avg_model_mem)
            start_time_2 = time.time()
            logging.info('updating custom models took %.0f msec', (start_time_2 - start_time_1) * 1000)


    # pylint: disable=too-many-arguments
    def annotate_document(self,
                          file_name: str,
                          provision_set: Set[str] = None,
                          work_dir: str = None,
                          is_doc_structure: bool = True,
                          doc_lang: str = "en") \
                          -> Tuple[Dict[str, List], ebantdoc2.EbAnnotatedDoc2]:
        time1 = time.time()
        if not provision_set:
            provision_set = self.provisions
        #else:
        #    logging.info('user specified provision list: %s', provision_set)

        if not work_dir:
            work_dir = self.work_dir

        # update custom models if necessary by checking dir.
        # custom models can be update by other workers

        self.update_custom_models()

        eb_antdoc = ebantdoc2.text_to_ebantdoc2(file_name,
                                                work_dir=work_dir,
                                                is_doc_structure=is_doc_structure,
                                                doc_lang=doc_lang)
        eb_antdoc3 = ebantdoc3.text_to_ebantdoc3(file_name,
                                                 work_dir=work_dir,
                                                 is_doc_structure=is_doc_structure,
                                                 doc_lang=doc_lang)

        # if the file contains too few words, don't bother
        # otherwise, might cause classifier error if only have 1 error because of minmax
        if len(eb_antdoc.text) < 100:
            empty_result = {}  # type: Dict[str, List]
            for prov in provision_set:
                empty_result[prov] = []
            # we always return eb_antdoc, not eb_antdoc3
            return empty_result, eb_antdoc
        # this execute the annotators in parallel
        prov_labels_map = self.run_annotators_in_parallel(eb_antdoc, eb_antdoc3, provision_set)

        # this update the 'start_end_span_list' in each antx in-place
        # docutils.update_ants_gap_spans(prov_labels_map, eb_antdoc.gap_span_list, eb_antdoc.text)
        # update prov_labels_map based on rules
        self.apply_line_annotators(prov_labels_map,
                                   eb_antdoc,
                                   work_dir=work_dir)
        # since nobody is using rate-table classifier yet
        # we are disabling it for now.
        # pylint: disable=pointless-string-statement
        """
        if eb_antdoc.doc_format == EbDocFormat.pdf:
            # print("classify_table_list......................................")
            rate_tables = rateclassifier.classify_table_list(eb_antdoc.table_list,
                                                             eb_antdoc.nl_text)
            #for rate_table in rate_tables:
            #    print("rate_table: {}".format(rate_table))
            prov_labels_map['rate_table'] = rate_tables
        else:
            # HTML document has no table detection, so 'rate-table' annotation is an empty list
            prov_labels_map['rate_table'] = []
        """

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
        logging.info('annotate_document(%s) took %0.2f sec', file_name, (time2 - time1))
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
            paras_with_attrs, para_doc_text, _, _ = \
                    htmltxtparser.parse_document('{}/{}'.format(work_dir, paraline_fname),
                                                 work_dir=work_dir,
                                                 is_combine_line=False)

            origin_sx_lnpos_list, nlp_sx_lnpos_list = \
                fromtomapper.paras_to_fromto_lists(paras_with_attrs)

            # there is no offset map because paraline is the same
            self.apply_line_annotators_aux(prov_labels_map, paras_with_attrs, para_doc_text,
                                           nlp_sx_lnpos_list, origin_sx_lnpos_list,
                                           eb_antdoc.nl_text)
        else:
            self.apply_line_annotators_aux(prov_labels_map,
                                           eb_antdoc.paras_with_attrs,
                                           eb_antdoc.nlp_text,
                                           eb_antdoc.nlp_sx_lnpos_list,
                                           eb_antdoc.origin_sx_lnpos_list,
                                           eb_antdoc.nl_text)


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

        # title works on the para_doc_text, not original text. so the
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
            logging.info('user specified provision list: %s', provision_set)

        # in reality, we only use 1 provision
        if len(provision_set) == 1:
            provision = list(provision_set)[0]
            annotator2 = self.provision_annotator_map[provision]
            if isinstance(annotator2, spanannotator.SpanAnnotator):
                ebantdoc_list = ebantdoc3.doclist_to_ebantdoc_list(txt_fns_file_name,
                                                                   self.work_dir)
            else:
                ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(txt_fns_file_name,
                                                                   self.work_dir)
        else:
            ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(txt_fns_file_name,
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
                                            candidate_type,
                                            is_doc_structure=False,
                                            work_dir=None,
                                            doc_lang="en") \
                                            -> Tuple[Dict[str, Any], Dict[str, Dict]]:

        logging.info("txt_fn_list_fn: %s", txt_fn_list)
        if not work_dir:
            work_dir = self.work_dir
        full_model_fname = '{}/{}'.format(custom_model_dir, base_model_fname)

        logging.info("custom_mode_file: %s", full_model_fname)

        # SENTENCE runs the standard pipeline, if specified candidate type run candidate generation
        if candidate_type == 'SENTENCE':
            eb_classifier = scutclassifier.ShortcutClassifier(provision)
            # It is know that 'eb_annotator' is ProvisionAnnotator, mypy.
            # Conflicts with below.
            eb_annotator, log_json = \
                ebtrainer.train_eval_annotator(provision,
                                               txt_fn_list,
                                               work_dir,
                                               custom_model_dir,
                                               full_model_fname,
                                               eb_classifier,
                                               is_doc_structure=is_doc_structure,
                                               custom_training_mode=True)
        else:
            # It is know that 'eb_annotator' is SpanAnnotator, mypy.
            # Conflicts with above.
            eb_annotator, log_json = \
                ebtrainer.train_eval_span_annotator(provision,
                                                    txt_fn_list,
                                                    work_dir,
                                                    custom_model_dir,
                                                    candidate_type,
                                                    model_file_name=full_model_fname,
                                                    is_bespoke_mode=True)

        if doc_lang != "en":
            provision = "{}_{}".format(provision, doc_lang)
        # update maps of provision to model name, provision to annotator, and custom model to
        # modified date
        old_provision_annotator = self.provision_annotator_map.get(provision)
        if old_provision_annotator:
            logging.info("Updating annotator, '%s', %s.", provision, full_model_fname)
            self.update_existing_provision_fn_map_aux(provision, full_model_fname)
        else:
            logging.info("Adding annotator, '%s', %s.", provision, full_model_fname)
            self.provisions.add(provision)
        self.provision_annotator_map[provision] = eb_annotator
        self.provision_custom_model_fn_map[provision] = full_model_fname

        # update the model timestamp to reflect last time trained
        mtime = os.path.getmtime(os.path.join(self.custom_model_dir, base_model_fname))
        last_modified_date = datetime.fromtimestamp(mtime)
        self.custom_model_timestamp_map[base_model_fname] = last_modified_date
        return eb_annotator.get_eval_status(), log_json

    def update_existing_provision_fn_map_aux(self,
                                             provision: str,
                                             full_model_fname: str) -> None:

        # intentionally not using .get(), because the previous model file must exist.
        prev_provision_model_fname = self.provision_custom_model_fn_map[provision]

        # in case the model version is different
        if prev_provision_model_fname != full_model_fname:
            logging.info("removing old customized model file, '%s', %s.",
                         provision, prev_provision_model_fname)
            if os.path.isfile(prev_provision_model_fname):
                os.remove(prev_provision_model_fname)
                # the old timestamp for the removed file doesn't matter.
                self.provision_custom_model_fn_map[provision] = full_model_fname

    # this function is here because it is a combination of both ML and rule-based annotator
    # pylint: disable=invalid-name
    def eval_mlxline_annotator_with_trte(self,
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
                prov_labels_map, ebantdoc = self.annotate_document(test_fn,
                                                                   provision_set=set([provision]),
                                                                   work_dir=work_dir)
                ant_list = prov_labels_map.get(provision, [])

                print("\ntest_fn = {}".format(test_fn))
                print("ant_list: {}".format(ant_list))

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
                                                                         ebantdoc.get_text(),
                                                                         diagnose_mode=True)
                else:
                    xtp, xfn, xfp, xtn, unused_json_log = \
                        evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                                ant_list,
                                                                ebantdoc.file_id,
                                                                ebantdoc.get_text(),
                                                                threshold,
                                                                is_raw_mode=False,
                                                                diagnose_mode=True)
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




# pylint: disable=too-few-public-methods
class EbDocCatRunner:

    def __init__(self, model_dir: str) -> None:
        osutils.mkpath(model_dir)
        self.model_dir = model_dir

        # load the available classifiers from dir_model
        full_model_fn = '{}/{}'.format(self.model_dir, DOCCAT_MODEL_FILE_NAME)
        logging.info("model_fn = [%s]", full_model_fn)

        if os.path.exists(full_model_fn):
            self.doc_classifier = joblib.load(full_model_fn)
            logging.info("EbDocCatRunner loading %s, %s", full_model_fn,
                         str(self.doc_classifier.catname_list))
            self.is_initialized = True
        else:
            logging.info("EbDocCatRunner not running because %s is missing.", full_model_fn)
            self.is_initialized = False

    def classify_document(self, fname: str) -> List[str]:
        # logging.info("classifying document: '{}'".format(fname))
        with open(fname, 'rt') as fin:
            doc_text = fin.read()
            return self.doc_classifier.predict(doc_text)


class EbLangDetectRunner:

    def __init__(self) -> None:
        pass

    # pylint: disable=no-self-use
    def detect_lang(self, atext: str) -> Optional[str]:
        try:
            detect_lang = langdetect.detect(atext)
        except LangDetectException:
            detect_lang = None
        # logging.info("detected language '{}'".format(detect_lang))
        return detect_lang

    # pylint: disable=no-self-use
    def detect_langs(self, atext: str) -> str:
        try:
            lang_probs = langdetect.detect_langs(atext)
            if lang_probs is None:
                return ''
            detect_langs = ','.join(['{}={}'.format(lang.lang, lang.prob) for lang in lang_probs])
        except LangDetectException:
            detect_langs = ''
        # logging.info("detected languages '{}'".format(detect_langs))
        return detect_langs
