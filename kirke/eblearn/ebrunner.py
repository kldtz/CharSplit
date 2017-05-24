
import concurrent.futures
import copy
import logging
import os
import psutil
import time
from datetime import datetime

from sklearn.externals import joblib

from kirke.eblearn import ebannotator, ebtext2antdoc, ebtrainer, scutclassifier
from kirke.utils import osutils

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

        total_mem_usage = py.memory_info()[0] / 2**20
        avg_model_mem = (total_mem_usage - orig_mem_usage) / num_model
        print('\ntotal mem: {:.2f},  model mem: {:.2f},  avg: {:.2f}'.format(total_mem_usage,
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

    def annotate_document(self, file_name, provision_set=None, work_dir=None):
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

        # this execute the annotators in parallel
        ant_result_dict = self.run_annotators_in_parallel(eb_antdoc, provision_set)

        # special handling for dates, as in PythonDateOfAgreementClassifier.java
        date_annotations = ant_result_dict.get('date')
        if not date_annotations:
            effectivedate_annotations = ant_result_dict.get('effectivedate')
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

        time2 = time.time()
        logging.info('annotate_document(%s) took %0.2f sec', file_name, (time2 - time1))
        return ant_result_dict, eb_antdoc.text

    
    def annotate_provision_in_document(self, file_name, provision: str):
        provision_set = set([provision])
        ant_result_dict, doc_text = self.annotate_document(file_name, provision_set)
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
                                            custom_model_dir, work_dir=None):

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
