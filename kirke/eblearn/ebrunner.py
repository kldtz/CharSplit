
import concurrent.futures
import copy
import json
import logging
import os
import psutil
import time
from datetime import datetime

from sklearn.externals import joblib

from kirke.eblearn import ebannotator, ebtext2antdoc, ebtrainer, scutclassifier, lineannotator
from kirke.utils import osutils, strutils, txtreader, evalutils

from kirke.docstruct import docreader, htmltxtparser, doc_pdf_reader

from kirke.ebrules import rateclassifier, titles, parties, dates

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
    # fix the issue with retired 'effectivedate'
    # first try to get effectivedate from rule-based approach
    # if none, then try get from ML approach.  The label is already correct.
    effectivedate_annotations = ant_result_dict.get('effectivedate_auto')
    if not effectivedate_annotations:
        effectivedate_annotations = ant_result_dict.get('effectivedate')
        if effectivedate_annotations:  # make a copy in 'effectivedate_auto'
            ant_result_dict['effectivedate_auto'] = effectivedate_annotations
            ant_result_dict['effectivedate'] = []

    # special handling for dates, as in PythonDateOfAgreementClassifier.java
    date_annotations = ant_result_dict.get('date')
    if not date_annotations:
        effectivedate_annotations = ant_result_dict.get('effectivedate_auto')
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


class EbRunner:

    def __init__(self, model_dir, work_dir, custom_model_dir):
        osutils.mkpath(model_dir)
        osutils.mkpath(work_dir)
        osutils.mkpath(custom_model_dir)
        self.model_dir = model_dir
        self.work_dir = work_dir
        self.custom_model_dir = custom_model_dir
        self.custom_model_timestamp_map = {}

        # load the available classifiers from dir_model
        model_files = osutils.get_model_files(model_dir)
        provision_classifier_map = {}
        self.provisions = set([])
        self.provision_annotator_map = {}

        # print("megabyte = {}".format(2**20))
        orig_mem_usage = py.memory_info()[0] / 2**20
        logging.info('original memory use: {} Mbytes'.format(orig_mem_usage))
        prev_mem_usage = orig_mem_usage
        num_model = 0

        for model_fn in model_files:
            full_model_fn = model_dir + "/" + model_fn
            prov_classifier = joblib.load(full_model_fn)
            clf_provision = prov_classifier.provision
            logging.info("ebrunner loading #%d: %s, %s", num_model, clf_provision, full_model_fn)
            if clf_provision in self.provisions:
                logging.warning("*** WARNING ***  Replacing an existing provision: %s",
                                clf_provision)
            provision_classifier_map[clf_provision] = prov_classifier
            self.provisions.add(clf_provision)
            if DEBUG_MODE:
                # print out memory usage info
                memoryUse = py.memory_info()[0] / 2**20
                print('loading #{} {:<50}, mem = {:.2f}, diff {:.2f}'.format(num_model,
                                                                             full_model_fn,
                                                                             memoryUse,
                                                                             memoryUse - prev_mem_usage))
                prev_mem_usage = memoryUse
            num_model += 1

        custom_model_files = osutils.get_model_files(custom_model_dir)
        for custom_model_fn in custom_model_files:
            # record the timestamp for update if needed in the future
            mtime = os.path.getmtime(os.path.join(self.custom_model_dir, custom_model_fn))
            last_modified_date = datetime.fromtimestamp(mtime)
            self.custom_model_timestamp_map[custom_model_fn] = last_modified_date

            full_custom_model_fn = custom_model_dir + "/" + custom_model_fn
            prov_classifier = joblib.load(full_custom_model_fn)
            clf_provision = prov_classifier.provision
            logging.info("ebrunner loading custom #%d: %s, %s", num_model, clf_provision, full_custom_model_fn)
            if clf_provision in self.provisions:
                logging.warning("*** WARNING ***  Replacing an existing provision: %s",
                                clf_provision)
            provision_classifier_map[clf_provision] = prov_classifier
            self.provisions.add(clf_provision)
            if DEBUG_MODE:
                memoryUse = py.memory_info()[0] / 2**20
                print('loading #{} {:<50}, mem = {:.2f}, diff {:.2f}'.format(num_model,
                                                                             full_model_fn,
                                                                             memoryUse,
                                                                             memoryUse - prev_mem_usage))
                prev_mem_usage = memoryUse
            num_model += 1

        for provision in self.provisions:
            pclassifier = provision_classifier_map[provision]
            self.provision_annotator_map[provision] = ebannotator.ProvisionAnnotator(pclassifier,
                                                                                     self.work_dir)

        self.title_annotator = lineannotator.LineAnnotator('title', titles.TitleAnnotator('title'))
        self.party_annotator = lineannotator.LineAnnotator('party', parties.PartyAnnotator('party'))
        self.date_annotator = lineannotator.LineAnnotator('date', dates.DateAnnotator('date'))

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

        annotations = {}
        with concurrent.futures.ThreadPoolExecutor(8) as executor:
            future_to_provision = {executor.submit(annotate_provision,
                                                   self.provision_annotator_map[provision],
                                                   eb_antdoc):
                                   provision for provision in provision_set}
            for future in concurrent.futures.as_completed(future_to_provision):
                provision = future_to_provision[future]
                data = future.result()
                annotations[provision] = data
        return annotations

    def update_custom_models(self):
        provision_classifier_map = {}
        orig_mem_usage = py.memory_info()[0] / 2**20
        num_model = 0

        start_time_1 = time.time()
        for fn in osutils.get_model_files(self.custom_model_dir):
            mtime = os.path.getmtime(os.path.join(self.custom_model_dir, fn))
            last_modified_date = datetime.fromtimestamp(mtime)
            # print("hi {} {}".format(fn, last_modified_date))

            old_timestamp = self.custom_model_timestamp_map.get(fn)
            if old_timestamp and old_timestamp == last_modified_date:
                pass
            else:
                prev_mem_usage = py.memory_info()[0] / 2**20
                # logging.info('current memory use: {} Mbytes'.format(prev_mem_usage))

                full_custom_model_fn = self.custom_model_dir + "/" + fn
                prov_classifier = joblib.load(full_custom_model_fn)
                clf_provision = prov_classifier.provision
                #if clf_provision in self.provisions:
                #    logging.warning("*** WARNING ***  Replacing an existing provision: %s",
                #                    clf_provision)
                provision_classifier_map[clf_provision] = prov_classifier
                self.custom_model_timestamp_map[fn] = last_modified_date
                self.provisions.add(clf_provision)
                num_model += 1

        if provision_classifier_map:
            for provision in provision_classifier_map.keys():
                logging.info("updating annotator: {}".format(provision))
                pclassifier = provision_classifier_map[provision]
                self.provision_annotator_map[provision] = ebannotator.ProvisionAnnotator(pclassifier,
                                                                                         self.work_dir)

            total_mem_usage = py.memory_info()[0] / 2**20
            avg_model_mem = (total_mem_usage - orig_mem_usage) / num_model
            logging.info('total mem: {:.2f},  model mem: {:.2f},  avg: {:.2f}'.format(total_mem_usage,
                                                                                      total_mem_usage - orig_mem_usage,
                                                                                      avg_model_mem))
            start_time_2 = time.time()
            logging.info('updating custom models took %.0f msec', (start_time_2 - start_time_1) * 1000)


    def annotate_text_document(self, file_name, provision_set=None, work_dir=None):
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

        eb_antdoc = ebtext2antdoc.doc_to_ebantdoc(file_name, work_dir)

        # if the file contains too few words, don't bother
        # otherwise, might cause classifier error if only have 1 error because of minmax
        if len(eb_antdoc.text) < 100:
            empty_result = {}
            for prov in provision_set:
                empty_result[prov] = []
            return empty_result, eb_antdoc.text

        # this execute the annotators in parallel
        ant_result_dict = self.run_annotators_in_parallel(eb_antdoc, provision_set)

        time2 = time.time()
        logging.info('annotate_text_document(%s) took %0.2f sec', file_name, (time2 - time1))
        return ant_result_dict, eb_antdoc.text

    def apply_line_annotators(self,
                              prov_labels_map,
                              file_name,
                              work_dir,
                              is_combine_line=False):
        # title works on the para_doc_text, not original text. so the
        # offsets needs to be adjusted, just like for text4nlp stuff.
        # The offsets here differs from above because of line break differs.
        # As a result, probably more page numbers are detected correctly and skipped.
        nl_paras_with_attrs, nl_para_doc_text, nl_gap_span_list, nl_orig_doc_text = \
            htmltxtparser.parse_document(file_name,
                                         work_dir=work_dir, is_combine_line=is_combine_line)
        nl_to_list, nl_from_list = htmltxtparser.paras_to_fromto_lists(nl_paras_with_attrs)

        title_ant_list = self.title_annotator.annotate_antdoc(nl_paras_with_attrs, nl_para_doc_text)
        # print('title_start, end = {}'.format(title_ant_list))
        # TODO, be careful about override title!
        # now we override.
        if title_ant_list:
            for antx in title_ant_list:
                # print("ant start = {}, end = {}".format(antx['start'], antx['end']))
                xstart = antx['start']
                xend = antx['end']
                antx['corenlp_start'] = xstart
                antx['corenlp_end'] = xend
                antx['start'] = docreader.find_offset_to(xstart, nl_from_list, nl_to_list)
                antx['end'] = docreader.find_offset_to(xend, nl_from_list, nl_to_list)
        prov_labels_map['title'] = title_ant_list

        party_ant_list = self.party_annotator.annotate_antdoc(nl_paras_with_attrs, nl_para_doc_text)
        # print('title_start, end = {}'.format(title_ant_list))
        # TODO, be careful about override title!
        # now we override.
        if party_ant_list:
            for antx in party_ant_list:
                # print("ant start = {}, end = {}".format(antx['start'], antx['end']))
                xstart = antx['start']
                xend = antx['end']
                antx['corenlp_start'] = xstart
                antx['corenlp_end'] = xend
                antx['start'] = docreader.find_offset_to(xstart, nl_from_list, nl_to_list)
                antx['end'] = docreader.find_offset_to(xend, nl_from_list, nl_to_list)
            prov_labels_map['party'] = party_ant_list


        date_ant_list = self.date_annotator.annotate_antdoc(nl_paras_with_attrs, nl_para_doc_text)
        # logging.info("running date_annotator()------2222222222222222--------- {}".format(len(date_ant_list)))
        if date_ant_list:
            xx_effective_date_list = []
            xx_date_list = []
            for antx in date_ant_list:
                # print("ant start = {}, end = {}".format(antx['start'], antx['end']))
                xstart = antx['start']
                xend = antx['end']
                antx['corenlp_start'] = xstart
                antx['corenlp_end'] = xend
                antx['start'] = docreader.find_offset_to(xstart, nl_from_list, nl_to_list)
                antx['end'] = docreader.find_offset_to(xend, nl_from_list, nl_to_list)

                if antx['label'] == 'effectivedate_auto':
                    xx_effective_date_list.append(antx)
                else:
                    xx_date_list.append(antx)
            if xx_effective_date_list:
                prov_labels_map['effectivedate_auto'] = xx_effective_date_list
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
            effectivedate_annotations = ant_result_dict.get('effectivedate_auto')
            if not effectivedate_annotations:
                effectivedate_annotations = ant_result_dict.get('effectivedate')
                if effectivedate_annotations:  # make a copy in 'effectivedate_auto'
                    ant_result_dict['effectivedate_auto'] = effectivedate_annotations
                    ant_result_dict['effectivedate'] = []

            # special handling for dates, as in PythonDateOfAgreementClassifier.java
            date_annotations = ant_result_dict.get('date')
            # print("-------------------------------------aaaaaaaaaaaaaaa")
            if not date_annotations:
                # print("-------------------------------------bbbbbbbbbbbbbbbbbb")
                effectivedate_annotations = ant_result_dict.get('effectivedate_auto')
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
            prov_labels_map['effectivedate_auto'] = prov_labels_map.get('effectivedate')

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

        time2 = time.time()
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
            prov_labels_map['effectivedate_auto'] = prov_labels_map.get('effectivedate')

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


    # TODO, this is the same as main.annotate_pdfboxed_document?
    # this calls annotate_text_document()
    def annotate_pdfboxed_document_for_table(self, file_name, linfo_file_name, provision_set=None, work_dir=None):
        time1 = time.time()

        base_fname = os.path.basename(file_name)

        # gap_span_list is for sentv2.txt or xxx.txt?
        # the offsets in para_list is for doc_text
        doc_text, gap_span_list, text4nlp_fn, text4nlp_offsets_fn, para_list = \
             docreader.parse_document(file_name, linfo_file_name, work_dir=work_dir)

        # now file_name.nlp.txt and file_name.nlp.offsets.json are created
        # text4nlp_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nlp.txt'))
        # text4nlp_offsets_fn = '{}/{}'.format(work_dir, base_fname.replace('.txt', '.nlp.offsets.json'))

        prov_labels_map, text4nlp = self.annotate_text_document(text4nlp_fn,
                                                                provision_set=provision_set,
                                                                work_dir=work_dir,
                                                                is_called_by_pdfboxed=True)
        # prov_labels_map, doc_text = eb_runner.annotate_document(file_name, set(['choiceoflaw','change_control', 'indemnify', 'jurisdiction', 'party', 'warranty', 'termination', 'term']))

        # translate the offsets
        from_list, to_list = docreader.read_fromto_json(text4nlp_offsets_fn)
        all_prov_ant_list = []
        for provision, ant_list in prov_labels_map.items():
            for antx in ant_list:
                # print("ant start = {}, end = {}".format(antx['start'], antx['end']))
                xstart = antx['start']
                xend = antx['end']
                antx['corenlp_start'] = xstart
                antx['corenlp_end'] = xend
                antx['start'] = docreader.find_offset_to(xstart, from_list, to_list)
                antx['end'] = docreader.find_offset_to(xend, from_list, to_list)

                all_prov_ant_list.append(antx)

        # this update the 'start_end_span_list' in each antx in-place
        docreader.update_ant_spans(all_prov_ant_list, gap_span_list, doc_text)

        # apply rule-based classification system
        # the offsets in para_list is for doc_text, so update the annotations after
        # adjustment were made
        table_list = docreader.extract_table_list(para_list)
        #if table_list:
        #    for i, atable in enumerate(table_list, 1):
        #        print("table #{}".format(i))
        #        for sentV4 in atable:
        #            print("\t{}".format(sentV4.text))
        prov_labels_map['rate_table'] = rateclassifier.classify_table_list(table_list, doc_text)
        # print("rate_table = {}".format(rateclassifier.classify_table_list(table_list, doc_text)))

        # save the prov_labels_map
        prov_ants_fn = file_name.replace('.txt', '.prov.ants.json')
        prov_ants_st = json.dumps(prov_labels_map)
        strutils.dumps(prov_ants_st, prov_ants_fn)

        time2 = time.time()
        logging.info('annotate_pdfboxed_document(%s) took %0.2f sec', file_name, (time2 - time1))
        return prov_labels_map, doc_text


    def annotate_provision_in_document(self, file_name, provision: str):
        provision_set = set([provision])
        ant_result_dict, doc_text = self.annotate_text_document(file_name, provision_set)
        prov_list = ant_result_dict[provision]
        for i, prov in enumerate(prov_list, 1):
            start = prov['start']
            end = prov['end']
            prob = prov['prob']
            print('{}\t{}\t{}\t{}\t{}\t{}\t{:.4f}'.format(file_name, i, provision, doc_text[start:end], start, end, prob))


    def test_annotators(self, txt_fns_file_name, provision_set, threshold=None):
        if not provision_set:
            provision_set = self.provisions
        else:
            logging.info('user specified provision list: %s', provision_set)

        ebantdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fns_file_name,
                                                               self.work_dir)

        annotations = {}
        with concurrent.futures.ThreadPoolExecutor(8) as executor:
            future_to_provision = {executor.submit(test_provision,
                                                   self.provision_annotator_map[provision],
                                                   ebantdoc_list,
                                                   threshold):
                                   provision for provision in provision_set}
            for future in concurrent.futures.as_completed(future_to_provision):
                provision = future_to_provision[future]
                data = future.result()
                annotations[provision] = data
        return annotations

    #
    # custom_train_provision_and_evaluate
    #
    # pylint: disable=C0103
    def custom_train_provision_and_evaluate(self, txt_fn_list, provision,
                                            custom_model_dir,
                                            is_doc_structure=False,
                                            work_dir=None):

        logging.info("txt_fn_list_fn: %s", txt_fn_list)

        model_file_name = '{}/{}_scutclassifier.pkl'.format(custom_model_dir,
                                                            provision)
        logging.info("custom_mode_file: %s", model_file_name)
        if not work_dir:
            work_dir = self.work_dir

        eb_classifier = scutclassifier.ShortcutClassifier(provision)
        eb_annotator = ebtrainer.train_eval_annotator(provision,
                                                      txt_fn_list,
                                                      work_dir,
                                                      custom_model_dir,
                                                      model_file_name,
                                                      eb_classifier,
                                                      is_doc_structure=is_doc_structure,
                                                      custom_training_mode=True)

        # update the hashmap of classifier
        old_provision_annotator = self.provision_annotator_map.get(provision)
        if old_provision_annotator:
            logging.info("Updating annotator, '%s', %s.", provision, model_file_name)
        else:
            logging.info("Adding annotator, '%s', %s.", provision, model_file_name)
            self.provisions.add(provision)
        self.provision_annotator_map[provision] = eb_annotator

        # updating the model timestamp, for update purpose
        local_custom_model_fn = "{}_scutclassifier.pkl".format(provision)
        mtime = os.path.getmtime(os.path.join(self.custom_model_dir, local_custom_model_fn))
        last_modified_date = datetime.fromtimestamp(mtime)
        self.custom_model_timestamp_map[local_custom_model_fn] = last_modified_date

        return eb_annotator.get_eval_status()


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
                prov_labels_map, doc_text = self.annotate_htmled_document(test_fn,
                                                                          provision_set=set([provision]),
                                                                          work_dir=work_dir)
                # special treatment for effectivedate, which is really effectivedate_auto
                if provision == 'effectivedate':
                    ant_list = prov_labels_map.get('effectivedate_auto', [])
                else:
                    ant_list = prov_labels_map.get(provision)

                print("\ntest_fn = {}".format(test_fn))
                print("ant_list: {}".format(ant_list))
                ebantdoc, ebant_fn = ebtext2antdoc.load_cached_ebantdoc(test_fn,
                                                                        is_bespoke_mode=False,
                                                                        work_dir=work_dir,
                                                                        is_cache_enabled=True)

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
