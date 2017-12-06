from collections import defaultdict
import copy
import concurrent.futures
from datetime import datetime
import json
import langdetect
import logging
import os
import psutil
import pprint
import re
import sys
import time
from typing import List

from sklearn.externals import joblib

from kirke.docstruct import docutils, fromtomapper, htmltxtparser
from kirke.eblearn import ebannotator, ebtrainer, lineannotator, provclassifier, scutclassifier
from kirke.ebrules import rateclassifier, titles, parties, dates
from kirke.utils import osutils, strutils, evalutils, ebantdoc2

from kirke.utils.ebantdoc2 import EbDocFormat

DEBUG_MODE = False

def annotate_provision(eb_annotator, eb_antdoc):
    return eb_annotator.annotate_antdoc(eb_antdoc)


def test_provision(eb_annotator, eb_antdoc_list, threshold):
    return eb_annotator.test_antdoc_list(eb_antdoc_list, threshold)

# def annotate_provision(eb_runner, file_name):
#     return eb_runner.annotate_document(file_name)

# def trainProvision(provision,train_file,test_file,bag_matrix,bag_matrix_te,Y,Y_te,prefix):
#  print ('STARTING TRAINING FOR ' + provision)
#  clf = EBClassifier(prefix,provision)
#  clf.train(train_file,test_file,bag_matrix,bag_matrix_te,Y,Y_te)
#  return clf

pid = os.getpid()
py = psutil.Process(pid)


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


class EbRunner:

    # pylint: disable=too-many-locals,too-many-statements
    def __init__(self, model_dir, work_dir, custom_model_dir):
        osutils.mkpath(model_dir)
        osutils.mkpath(work_dir)
        osutils.mkpath(custom_model_dir)
        self.model_dir = model_dir
        self.work_dir = work_dir
        self.custom_model_dir = custom_model_dir
        self.custom_model_timestamp_map = {}
        # Kirke can have different custom model versions for the same provision right
        # after training new ones. Need to keep them unique by remove the old one.
        # Must syncrhonize the update of self.provision_annotator_map and
        # self.provision_custom_model_fn_map.
        self.provision_custom_model_fn_map = {}

        # load the available classifiers from dir_model
        model_files = osutils.get_model_files(model_dir)
        provision_classifier_map = {}
        self.provisions = set([])
        self.provision_annotator_map = {}

        # print("megabyte = {}".format(2**20))
        orig_mem_usage = py.memory_info()[0] / 2**20
        logging.info('original memory use: %d Mbytes', orig_mem_usage)
        prev_mem_usage = orig_mem_usage
        num_model = 0

        for model_fn in model_files:
            full_model_fn = model_dir + "/" + model_fn
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
                memory_use = py.memory_info()[0] / 2**20
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
                memory_use = py.memory_info()[0] / 2**20
                # pylint: disable=line-too-long
                print('loading #{} {:<50}, mem = {:.2f}, diff {:.2f}'.format(num_model,
                                                                             full_model_fn,
                                                                             memory_use,
                                                                             memory_use - prev_mem_usage))
                prev_mem_usage = memory_use
            num_model += 1

        for provision in self.provisions:
            pclassifier = provision_classifier_map[provision]
            prov_threshold = provclassifier.get_provision_threshold(provision)  # in case we want to override
            self.provision_annotator_map[provision] = ebannotator.ProvisionAnnotator(pclassifier,
                                                                                     self.work_dir,
                                                                                     threshold=prov_threshold)

        self.title_annotator = lineannotator.LineAnnotator('title', titles.TitleAnnotator('title'))
        self.party_annotator = lineannotator.LineAnnotator('party', parties.PartyAnnotator('party'))
        self.date_annotator = lineannotator.LineAnnotator('date', dates.DateAnnotator('date'))

        if num_model == 0:
            logging.error('No model is loaded from {} and {}.'.format(model_dir, custom_model_dir))
            logging.error('Please also verify model file names match the filter in osutils.get_model_files()')
            return None

        total_mem_usage = py.memory_info()[0] / 2**20
        avg_model_mem = (total_mem_usage - orig_mem_usage) / num_model
        logging.info('total mem: {:.2f},  model mem: {:.2f},  avg: {:.2f}'.format(total_mem_usage,
                                                                                  total_mem_usage - orig_mem_usage,
                                                                                  avg_model_mem))
        logging.info('EbRunner is initiated.')

    def run_annotators_in_parallel(self, eb_antdoc, provision_set=None):
        if not provision_set:
            provision_set = self.provisions
        #else:
        #    logging.info("user specified provision list: %s", provision_set)

        annotations = defaultdict(list)
        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            future_to_provision = {executor.submit(annotate_provision,
                                                   self.provision_annotator_map[provision],
                                                   eb_antdoc):
                                   provision for provision in provision_set if provision in self.provision_annotator_map.keys()} 
            for future in concurrent.futures.as_completed(future_to_provision):
                provision = future_to_provision[future]
                data = future.result()
                # want to collapse language-specific cust models to one provision
                if 'cust_' in provision and data:
                    provision = data[0]['label']
                # aggregates all annotations across languages for cust models
                annotations[provision].extend(data)
        return annotations

    def update_custom_models(self):
        provision_classifier_map = {}
        orig_mem_usage = py.memory_info()[0] / 2**20
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
                # prev_mem_usage = py.memory_info()[0] / 2**20
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
                    self.update_existing_provision_fn_map_aux(clf_provision, full_custom_model_fname)
                # if the same, don't do anything

        if provision_classifier_map:
            for provision in provision_classifier_map:
                logging.info("updating annotator: %s", provision)
                pclassifier = provision_classifier_map[provision]
                self.provision_annotator_map[provision] = \
                    ebannotator.ProvisionAnnotator(pclassifier, self.work_dir)

            total_mem_usage = py.memory_info()[0] / 2**20
            avg_model_mem = (total_mem_usage - orig_mem_usage) / num_model
            # pylint: disable=line-too-long
            logging.info('total mem: {:.2f},  model mem: {:.2f},  avg: {:.2f}'.format(total_mem_usage,
                                                                                      total_mem_usage - orig_mem_usage,
                                                                                      avg_model_mem))
            start_time_2 = time.time()
            logging.info('updating custom models took %.0f msec', (start_time_2 - start_time_1) * 1000)


    def annotate_document(self, file_name, provision_set=None, work_dir=None, is_doc_structure=True, doc_lang="en"):
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

        # if the file contains too few words, don't bother
        # otherwise, might cause classifier error if only have 1 error because of minmax
        if len(eb_antdoc.text) < 100:
            empty_result = {}
            for prov in provision_set:
                empty_result[prov] = []
            return empty_result, eb_antdoc
        # this execute the annotators in parallel
        prov_labels_map = self.run_annotators_in_parallel(eb_antdoc, provision_set)

        # this update the 'start_end_span_list' in each antx in-place
        # docutils.update_ants_gap_spans(prov_labels_map, eb_antdoc.gap_span_list, eb_antdoc.text)

        # update prov_labels_map based on rules
        self.apply_line_annotators(prov_labels_map,
                                   eb_antdoc,
                                   work_dir=work_dir)

        if eb_antdoc.doc_format == EbDocFormat.pdf:
            # print("classify_table_list......................................")
            rate_tables = rateclassifier.classify_table_list(eb_antdoc.table_list, eb_antdoc.nl_text)
            #for rate_table in rate_tables:
            #    print("rate_table: {}".format(rate_table))
            prov_labels_map['rate_table'] = rate_tables
        else:
            # HTML document has no table detection, so 'rate-table' annotation is an empty list
            prov_labels_map['rate_table'] = []

        # jshaw. evalxxx, composite
        update_dates_by_domain_rules(prov_labels_map)

        # save the prov_labels_map
        prov_ants_fn = file_name.replace('.txt', '.prov.ants.json')
        prov_ants_st = json.dumps(prov_labels_map)
        strutils.dumps(prov_ants_st, prov_ants_fn)

        time2 = time.time()
        logging.info('annotate_document(%s) took %0.2f sec', file_name, (time2 - time1))
        return prov_labels_map, eb_antdoc


    def apply_line_annotators(self, prov_labels_map, eb_antdoc, work_dir):
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

            fromto_mapper = fromtomapper.paras_to_fromto_mapper_sorted_by_from(paras_with_attrs)
            origin_sx_lnpos_list, nlp_sx_lnpos_list = fromtomapper.paras_to_fromto_lists(paras_with_attrs)

            # there is no offset map because paraline is the same
            self.apply_line_annotators_aux(prov_labels_map, paras_with_attrs, para_doc_text,
                                           nlp_sx_lnpos_list, origin_sx_lnpos_list)
        else:
            self.apply_line_annotators_aux(prov_labels_map, eb_antdoc.paras_with_attrs, eb_antdoc.nlp_text,
                                           eb_antdoc.nlp_sx_lnpos_list, eb_antdoc.origin_sx_lnpos_list)


    # TODO, remove later, this is for html, original
    # NLP text doesn't always work for PDF because blanks lines sometime cause
    # whole page to be the same paragraph
    def apply_line_annotators_aux_orig(self, prov_labels_map, eb_antdoc, work_dir):

        fromto_mapper = fromtomapper.FromToMapper('an offset mapper', eb_antdoc.nlp_sx_lnpos_list, eb_antdoc.origin_sx_lnpos_list)

        # title works on the para_doc_text, not original text. so the
        # offsets needs to be adjusted, just like for text4nlp stuff.
        # The offsets here differs from above because of line break differs.
        # As a result, probably more page numbers are detected correctly and skipped.
        title_ant_list = self.title_annotator.annotate_antdoc(eb_antdoc.paras_with_attrs,
                                                              eb_antdoc.nlp_text)

        # we always replace the title using rules
        fromto_mapper.adjust_fromto_offsets(title_ant_list)
        prov_labels_map['title'] = title_ant_list

        party_ant_list = self.party_annotator.annotate_antdoc(eb_antdoc.paras_with_attrs,
                                                              eb_antdoc.nlp_text)

        # if rule found parties, replace it.  Otherwise, keep the old ones
        if party_ant_list:
            fromto_mapper.adjust_fromto_offsets(party_ant_list)
            prov_labels_map['party'] = party_ant_list

        # comment out all the date code below to disable applying date rule
        date_ant_list = self.date_annotator.annotate_antdoc(eb_antdoc.paras_with_attrs,
                                                            eb_antdoc.nlp_text)
        if date_ant_list:
            xx_effective_date_list = []
            xx_date_list = []
            fromto_mapper.adjust_fromto_offsets(date_ant_list)

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
                #    prov_labels_map['date'] = []  # let override later in update_dates_by_domain_rules()
                # prov_labels_map['effectivedate'] = xx_effective_date_list
            if xx_date_list:
                prov_labels_map['date'] = xx_date_list

                
    def annotate_text_document_too_new(self,
                               file_name,
                               provision_set=None,
                               work_dir=None,
                               is_called_by_pdfboxed=False,
                               is_doc_structure=False):
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

        eb_antdoc = ebtext2antdoc.doc_to_ebantdoc(file_name,
                                                  work_dir,
                                                  is_doc_structure=is_doc_structure)

        # if the file contains too few words, don't bother
        # otherwise, might cause classifier error if only have 1 error because of minmax
        if len(eb_antdoc.text) < 100:
            empty_result = {}
            for prov in provision_set:
                empty_result[prov] = []
            return empty_result, eb_antdoc.text

        # this execute the annotators in parallel
        ant_result_dict = self.run_annotators_in_parallel(eb_antdoc, provision_set)

        if not is_called_by_pdfboxed:
            # now adjust the date using domain specific logic
            # fix the issue with retired 'effectivedate'
            # first try to get effectivedate from rule-based approach
            # if none, then try get from ML approach.  The label is already correct.
            effectivedate_annotations = ant_result_dict.get('effectivedate_auto', [])
            if not effectivedate_annotations:
                effectivedate_annotations = ant_result_dict.get('effectivedate', [])
                if effectivedate_annotations:  # make a copy in 'effectivedate_auto'
                    ant_result_dict['effectivedate_auto'] = effectivedate_annotations
                    ant_result_dict['effectivedate'] = []

            # special handling for dates, as in PythonDateOfAgreementClassifier.java
            date_annotations = ant_result_dict.get('date')
            # print("-------------------------------------aaaaaaaaaaaaaaa")
            if not date_annotations:
                # print("-------------------------------------bbbbbbbbbbbbbbbbbb")
                effectivedate_annotations = ant_result_dict.get('effectivedate_auto', [])
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

            # save the prov_labels_map
            prov_ants_fn = file_name.replace('.txt', '.prov.ants.json')
            prov_ants_st = json.dumps(ant_result_dict)
            strutils.dumps(prov_ants_st, prov_ants_fn)

        time2 = time.time()
        logging.info('annotate_document(%s) took %0.2f sec', file_name, (time2 - time1))
        return ant_result_dict, eb_antdoc.text


    # this parses both originally text and html documents
    # It's main goal is to detect sechead
    # optionally pagenum, footer, toc, signature
    def annotate_htmled_document(self, file_name, provision_set=None, work_dir=None, doc_lang="en"):
        time1 = time.time()

        # base_fname = os.path.basename(file_name)

        prov_labels_map, orig_doc_text = self.annotate_text_document(file_name,
                                                                     provision_set=provision_set,
                                                                     work_dir=work_dir,
								     doc_lang=doc_lang)

    def apply_line_annotators_aux(self, prov_labels_map, paraline_with_attrs, paraline_text,
                                      paraline_sx_lnpos_list, origin_sx_lnpos_list):

        # update custom models if necessary by checking dir.
        # custom models can be update by other workers
        self.update_custom_models()

        eb_antdoc = ebtext2antdoc.doc_to_ebantdoc(file_name,
                                                  work_dir,
                                                  is_doc_structure=is_doc_structure)

        # if the file contains too few words, don't bother
        # otherwise, might cause classifier error if only have 1 error because of minmax
        if len(eb_antdoc.text) < 100:
            empty_result = {}
            for prov in provision_set:
                empty_result[prov] = []
            return empty_result, eb_antdoc.text

        # this execute the annotators in parallel
        ant_result_dict = self.run_annotators_in_parallel(eb_antdoc, provision_set)

        if not is_called_by_pdfboxed:
            # now adjust the date using domain specific logic
            # fix the issue with retired 'effectivedate'
            # first try to get effectivedate from rule-based approach
            # if none, then try get from ML approach.  The label is already correct.
            effectivedate_annotations = ant_result_dict.get('effectivedate_auto', [])
            if not effectivedate_annotations:
                effectivedate_annotations = ant_result_dict.get('effectivedate', [])
                if effectivedate_annotations:  # make a copy in 'effectivedate_auto'
                    ant_result_dict['effectivedate_auto'] = effectivedate_annotations
                    ant_result_dict['effectivedate'] = []

            # special handling for dates, as in PythonDateOfAgreementClassifier.java
            date_annotations = ant_result_dict.get('date')
            # print("-------------------------------------aaaaaaaaaaaaaaa")
            if not date_annotations:
                # print("-------------------------------------bbbbbbbbbbbbbbbbbb")
                effectivedate_annotations = ant_result_dict.get('effectivedate_auto', [])
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

            # save the prov_labels_map
            prov_ants_fn = file_name.replace('.txt', '.prov.ants.json')
            prov_ants_st = json.dumps(ant_result_dict)
            strutils.dumps(prov_ants_st, prov_ants_fn)

        time2 = time.time()
        logging.info('annotate_document(%s) took %0.2f sec', file_name, (time2 - time1))
        return ant_result_dict, eb_antdoc.text

    # this parses both originally text and html documents
    # It's main goal is to detect sechead
    # optionally pagenum, footer, toc, signature
    def annotate_htmled_document(self, file_name, provision_set=None, work_dir=None):
        time1 = time.time()

        # base_fname = os.path.basename(file_name)

        prov_labels_map, orig_doc_text = self.annotate_text_document(file_name,
                                                                     provision_set=provision_set,
                                                                     work_dir=work_dir)

        # because special case of 'effectivdate_auto'
        if not prov_labels_map.get('effectivedate_auto'):
            prov_labels_map['effectivedate_auto'] = prov_labels_map.get('effectivedate', [])

        # TODO, uncomment this below for production
        # this will updae prov_labels_map
        self.apply_line_annotators(prov_labels_map,
                                   file_name,
                                   work_dir=work_dir,
                                   # for HTML, we combine lines for sechead
                                   is_combine_line=True)


        # HTML document has no table detection, so 'rate-table' annotation is an empty list
        prov_labels_map['rate_table'] = []

        # jshaw. evalxxx, composite
        update_dates_by_domain_rules(prov_labels_map)

        # save the prov_labels_map
        prov_ants_fn = file_name.replace('.txt', '.prov.ants.json')
        prov_ants_st = json.dumps(prov_labels_map)
        strutils.dumps(prov_ants_st, prov_ants_fn)

        logging.info('annotate_htmled_document(%s) took %0.2f sec', file_name, (time2 - time1))
        return prov_labels_map, orig_doc_text

    
    # this parses both originally text and html documents
    # It's main goal is to detect sechead
    # optionally pagenum, footer, toc, signature
    def annotate_pdfboxed_document(self, file_name, offsets_file_name, provision_set=None, work_dir=None):
        time1 = time.time()

        orig_text, nl_text, paraline_text, nl_fname, paraline_fname = \
           doc_pdf_reader.to_nl_paraline_texts(file_name, offsets_file_name, work_dir=work_dir)
        
        # base_fname = os.path.basename(file_name)

        prov_labels_map, orig_doc_text = self.annotate_text_document(file_name,
                                                                     provision_set=provision_set,
                                                                     work_dir=work_dir)
        # because special case of 'effectivdate_auto'
        if not prov_labels_map.get('effectivedate_auto'):
            prov_labels_map['effectivedate_auto'] = prov_labels_map.get('effectivedate', [])

        # this will updae prov_labels_map
        self.apply_line_annotators(prov_labels_map,
                                   paraline_fname,
                                   work_dir=work_dir,
                                   # for PDF document, we do not combine lines for sechead
                                   is_combine_line=False)

        # this update the 'start_end_span_list' in each antx in-place
        # docreader.update_ant_spans(all_prov_ant_list, gap_span_list, orig_doc_text)

        # HTML document has no table detection, so 'rate-table' annotation is an empty list
        prov_labels_map['rate_table'] = []

        update_dates_by_domain_rules(prov_labels_map)

        # save the prov_labels_map
        prov_ants_fn = file_name.replace('.txt', '.prov.ants.json')
        prov_ants_st = json.dumps(prov_labels_map)
        strutils.dumps(prov_ants_st, prov_ants_fn)

        time2 = time.time()
        logging.info('annotate_pdfboxed_document(%s) took %0.2f sec', file_name, (time2 - time1))
        return prov_labels_map, orig_doc_text
        fromto_mapper = fromtomapper.FromToMapper('an offset mapper', paraline_sx_lnpos_list, origin_sx_lnpos_list)

        # title works on the para_doc_text, not original text. so the
        # offsets needs to be adjusted, just like for text4nlp stuff.
        # The offsets here differs from above because of line break differs.
        # As a result, probably more page numbers are detected correctly and skipped.
        title_ant_list = self.title_annotator.annotate_antdoc(paraline_with_attrs,
                                                              paraline_text)

        # we always replace the title using rules
        fromto_mapper.adjust_fromto_offsets(title_ant_list)
        prov_labels_map['title'] = title_ant_list

        party_ant_list = self.party_annotator.annotate_antdoc(paraline_with_attrs,
                                                              paraline_text)

        # if rule found parties, replace it.  Otherwise, keep the old ones
        if party_ant_list:
            fromto_mapper.adjust_fromto_offsets(party_ant_list)
            prov_labels_map['party'] = party_ant_list

        # comment out all the date code below to disable applying date rule
        date_ant_list = self.date_annotator.annotate_antdoc(paraline_with_attrs,
                                                            paraline_text)
        if date_ant_list:
            xx_effective_date_list = []
            xx_date_list = []
            fromto_mapper.adjust_fromto_offsets(date_ant_list)

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
                #    prov_labels_map['date'] = []  # let override later in update_dates_by_domain_rules()
                # prov_labels_map['effectivedate'] = xx_effective_date_list
            if xx_date_list:
                prov_labels_map['date'] = xx_date_list
    
    # this parses both originally text and html documents
    # It's main goal is to detect sechead
    # optionally pagenum, footer, toc, signature
    def annotate_pdfboxed_document(self, file_name, offsets_file_name, provision_set=None, work_dir=None, doc_lang="en"):
        time1 = time.time()

        orig_text, nl_text, paraline_text, nl_fname, paraline_fname = \
           doc_pdf_reader.to_nl_paraline_texts(file_name, offsets_file_name, work_dir=work_dir)
        
        # base_fname = os.path.basename(file_name)

        prov_labels_map, orig_doc_text = self.annotate_text_document(file_name,
                                                                     provision_set=provision_set,
                                                                     work_dir=work_dir,
								     doc_lang=doc_lang)
        # because special case of 'effectivdate_auto'
        if not prov_labels_map.get('effectivedate_auto'):
            prov_labels_map['effectivedate_auto'] = prov_labels_map.get('effectivedate', [])

        # this will updae prov_labels_map
        self.apply_line_annotators(prov_labels_map,
                                   paraline_fname,
                                   work_dir=work_dir,
                                   # for PDF document, we do not combine lines for sechead
                                   is_combine_line=False)

        # this update the 'start_end_span_list' in each antx in-place
        # docreader.update_ant_spans(all_prov_ant_list, gap_span_list, orig_doc_text)

        # HTML document has no table detection, so 'rate-table' annotation is an empty list
        prov_labels_map['rate_table'] = []

        update_dates_by_domain_rules(prov_labels_map)

        # save the prov_labels_map
        prov_ants_fn = file_name.replace('.txt', '.prov.ants.json')
        prov_ants_st = json.dumps(prov_labels_map)
        strutils.dumps(prov_ants_st, prov_ants_fn)

        time2 = time.time()
        logging.info('annotate_pdfboxed_document(%s) took %0.2f sec', file_name, (time2 - time1))
        return prov_labels_map, orig_doc_text
        fromto_mapper = fromtomapper.FromToMapper('an offset mapper', paraline_sx_lnpos_list, origin_sx_lnpos_list)

        # title works on the para_doc_text, not original text. so the
        # offsets needs to be adjusted, just like for text4nlp stuff.
        # The offsets here differs from above because of line break differs.
        # As a result, probably more page numbers are detected correctly and skipped.
        title_ant_list = self.title_annotator.annotate_antdoc(paraline_with_attrs,
                                                              paraline_text)

        # we always replace the title using rules
        fromto_mapper.adjust_fromto_offsets(title_ant_list)
        prov_labels_map['title'] = title_ant_list

        party_ant_list = self.party_annotator.annotate_antdoc(paraline_with_attrs,
                                                              paraline_text)

        # if rule found parties, replace it.  Otherwise, keep the old ones
        if party_ant_list:
            fromto_mapper.adjust_fromto_offsets(party_ant_list)
            prov_labels_map['party'] = party_ant_list

        # comment out all the date code below to disable applying date rule
        date_ant_list = self.date_annotator.annotate_antdoc(paraline_with_attrs,
                                                            paraline_text)
        if date_ant_list:
            xx_effective_date_list = []
            xx_date_list = []
            fromto_mapper.adjust_fromto_offsets(date_ant_list)

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
                #    prov_labels_map['date'] = []  # let override later in update_dates_by_domain_rules()
                # prov_labels_map['effectivedate'] = xx_effective_date_list
            if xx_date_list:
                prov_labels_map['date'] = xx_date_list


    def annotate_provision_in_document(self, file_name, provision: str):
        provision_set = set([provision])
        ant_result_dict, doc_text = self.annotate_text_document(file_name, provision_set)
        prov_list = ant_result_dict[provision]
        for i, prov in enumerate(prov_list, 1):
            start = prov['start']
            end = prov['end']
            prob = prov['prob']
            print('{}\t{}\t{}\t{}\t{}\t{}\t{:.4f}'.format(file_name, i, provision,
                                                          doc_text[start:end], start, end, prob))


    def test_annotators(self, txt_fns_file_name, provision_set, threshold=None):
        if not provision_set:
            provision_set = self.provisions
        else:
            logging.info('user specified provision list: %s', provision_set)

        ebantdoc_list = ebantdoc2.doclist_to_ebantdoc_list(txt_fns_file_name,
                                                           self.work_dir)

        annotations = {}
        logs = {}
        print("<<<<<<", threshold)
        with concurrent.futures.ThreadPoolExecutor(8) as executor:
            future_to_provision = {executor.submit(test_provision,
                                                   self.provision_annotator_map[provision],
                                                   ebantdoc_list,
                                                   threshold):
                                   provision for provision in provision_set}
            for future in concurrent.futures.as_completed(future_to_provision):
                provision = future_to_provision[future]
                data, log_json = future.result()
                logs[provision] = log_json
                annotations[provision] = data
        return annotations, logs

    #
    # custom_train_provision_and_evaluate
    #
    # pylint: disable=C0103
    def custom_train_provision_and_evaluate(self,
                                            txt_fn_list,
                                            provision,
                                            custom_model_dir,
                                            base_model_fname,
                                            is_doc_structure=False,
                                            work_dir=None,
                                            doc_lang="en"):

        logging.info("txt_fn_list_fn: %s", txt_fn_list)
        
        if not work_dir:
            work_dir = self.work_dir

        full_model_fname = '{}/{}'.format(custom_model_dir, base_model_fname)

        logging.info("custom_mode_file: %s", full_model_fname)

        # eb_classifier = scutclassifier.ShortcutClassifier(provision)
        eb_classifier = scutclassifier.ShortcutClassifier(provision)
        eb_annotator = ebtrainer.train_eval_annotator(provision,
                                                      txt_fn_list,
                                                      work_dir,
                                                      custom_model_dir,
                                                      full_model_fname,
                                                      eb_classifier,
                                                      is_doc_structure=is_doc_structure,
                                                      custom_training_mode=True,
                                                      doc_lang=doc_lang)

        # update the hashmap of classifier
        if doc_lang != "en":
            provision = "{}_{}".format(provision, doc_lang)
        old_provision_annotator = self.provision_annotator_map.get(provision)
        if old_provision_annotator:
            logging.info("Updating annotator, '%s', %s.", provision, full_model_fname)
            self.update_existing_provision_fn_map_aux(provision, full_model_fname)
        else:
            logging.info("Adding annotator, '%s', %s.", provision, full_model_fname)
            self.provisions.add(provision)
        self.provision_annotator_map[provision] = eb_annotator
        self.provision_custom_model_fn_map[provision] = full_model_fname

        # updating the model timestamp, for update purpose
        mtime = os.path.getmtime(os.path.join(self.custom_model_dir, base_model_fname))
        last_modified_date = datetime.fromtimestamp(mtime)
        self.custom_model_timestamp_map[base_model_fname] = last_modified_date

        return eb_annotator.get_eval_status()

    def update_existing_provision_fn_map_aux(self, provision: str, full_model_fname: str) -> None:
        # intentioanlly not using .get(), because the previous model file must exist.
        prev_provision_model_fname = self.provision_custom_model_fn_map[provision]
        # in case the model version is different
        if prev_provision_model_fname != full_model_fname:
            logging.info("removing old customized model file, '%s', %s.", provision, prev_provision_model_fname)
            if os.path.isfile(prev_provision_model_fname):
                os.remove(prev_provision_model_fname)
                # the old timestamp for the removed file doesn't matter.
                # tmp_base_fname = os.path.basename(prev_provision_model_fname)
                # del self.custom_model_timestamp_map[tmp_base_fname]
                self.provision_custom_model_fn_map[provision] = full_model_fname

    def eval_ml_rule_annotator_with_trte(self,
                                         provision,
                                         model_dir='dir-scut-model',
                                         work_dir='dir-work',
                                         is_doc_structure=False):

        test_doclist_fn = "{}/{}_test_doclist.txt".format(model_dir, provision)
        num_test_doc = 0
        # pylint: disable=C0103
        tp, fn, fp, tn = 0, 0, 0, 0

        with open(test_doclist_fn, 'rt') as testin:
            for test_fn in testin:
                num_test_doc += 1
                test_fn = test_fn.strip()
                prov_labels_map, ebantdoc = self.annotate_document(test_fn,
                                                                   provision_set=set([provision]),
                                                                   work_dir=work_dir)
                ant_list = prov_labels_map.get(provision)

                print("\ntest_fn = {}".format(test_fn))
                print("ant_list: {}".format(ant_list))

                print('ebantdoc.fileid = {}'.format(ebantdoc.file_id))
                # print("ant_list: {}".format(ant_list))
                prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                                       if hant.label == provision]

                # print("\nfn: {}".format(ebantdoc.file_id))
                # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
                # pred_prob_start_end_list, txt)
                if provision in ebannotator.PROVISION_EVAL_ANYMATCH_SET:
                    xtp, xfn, xfp, xtn = \
                        evalutils.calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list,
                                                                         ant_list,
                                                                         ebantdoc.get_text(),
                                                                         diagnose_mode=True)
                else:
                    xtp, xfn, xfp, xtn = \
                        evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                                ant_list,
                                                                ebantdoc.get_text(),
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

    def __init__(self, model_dir):
        osutils.mkpath(model_dir)
        self.model_dir = model_dir

        # load the available classifiers from dir_model
        full_model_fn = self.model_dir + '/ebrevia_docclassifier.pkl'
        print("model_fn = [{}]".format(full_model_fn))

        self.doc_classifier = joblib.load(full_model_fn)
        logging.info("EbDocCatRunner loading %s, %s", full_model_fn,
                     str(self.doc_classifier.catname_list))

    def classify_document(self, fname):
        # logging.info("classifying document: '{}'".format(fname))
        with open(fname, 'rt') as fin:
            doc_text = fin.read()
            return self.doc_classifier.predict(doc_text)


class EbLangDetectRunner:

    def __init__(self):
        pass

    def detect_lang(self, atext):
        try:
            detect_lang = langdetect.detect(atext) or 'unknown'
        except:
            detect_lang = 'unknown'
        # logging.info("detected language '{}'".format(detect_lang))
        return detect_lang

    def detect_langs(self, atext):
        try:
            lang_probs = langdetect.detect_langs(atext)
            detect_langs = ','.join(['{}={}'.format(lang.lang, lang.prob) for lang in lang_probs])
        except:
            detect_langs = 'unknown=0.00001'
        # logging.info("detected languages '{}'".format(detect_langs))
        return detect_langs
