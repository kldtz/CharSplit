import concurrent.futures
import logging


from sklearn.externals import joblib

from eblearn import ebtext2antdoc
from eblearn import ebannotator, ebtrainer
from eblearn import scutclassifier
from utils import osutils


def annotate_provision(eb_annotator, eb_antdoc):
    return eb_annotator.annotate_antdoc(eb_antdoc)

def test_provision(eb_annotator, eb_antdoc_list):
    return eb_annotator.test_antdoc_list(eb_antdoc_list)

# def annotate_provision(eb_runner, file_name):
#     return eb_runner.annotate_document(file_name)

#def trainProvision(provision,train_file,test_file,bag_matrix,bag_matrix_te,Y,Y_te,prefix):
#  print ('STARTING TRAINING FOR ' + provision)
#  clf = EBClassifier(prefix,provision)
#  clf.train(train_file,test_file,bag_matrix,bag_matrix_te,Y,Y_te)
#  return clf


class EbRunner:
    
    def __init__(self, model_dir, work_dir, custom_model_dir):
        osutils.mkpath(model_dir)
        osutils.mkpath(work_dir)
        osutils.mkpath(custom_model_dir)
        self.model_dir = model_dir
        self.work_dir = work_dir
        self.custom_model_dir = custom_model_dir

        # load the available classifiers from dir_model
        # xxx
        model_files = osutils.get_model_files(model_dir)
        provision_classifier_map = {}
        self.provisions = set([])
        self.provision_annotator_map = {}
        
        for model_fn in model_files:
            full_model_fn = model_dir + "/" + model_fn
            prov_classifier = joblib.load(full_model_fn)
            logging.info("ebrunner loading: {}, {}".format(prov_classifier.provision, full_model_fn))
            if prov_classifier.provision in self.provisions:
                logging.warning("*** WARNING ***  Replacing an existing provision: '{}'".format(prov_classifier.provision))            
            provision_classifier_map[prov_classifier.provision] = prov_classifier
            self.provisions.add(prov_classifier.provision)

        custom_model_files = osutils.get_model_files(custom_model_dir)
        for custom_model_fn in custom_model_files:
            full_custom_model_fn = custom_model_dir + "/" + custom_model_fn
            prov_classifier = joblib.load(full_custom_model_fn)
            logging.info("ebrunner loading: {}, {}".format(prov_classifier.provision, full_custom_model_fn))
            if prov_classifier.provision in self.provisions:
                logging.warning("*** WARNING ***  Replacing an existing provision: '{}'".format(prov_classifier.provision))
            provision_classifier_map[prov_classifier.provision] = prov_classifier
            self.provisions.add(prov_classifier.provision)

        for provision in self.provisions:
            pclassifier = provision_classifier_map[provision]
            self.provision_annotator_map[provision] = ebannotator.ProvisionAnnotator(pclassifier, self.work_dir)
        logging.info('EbRunner is initiated.')


    def run_annotators_in_parallel(self, eb_antdoc, provision_set=None):
        if not provision_set:
            provision_set = self.provisions

        # TODO, jshaw
        # hacking for now
        logging.info("reset provision list: {}".format(provision_set))
        provision_set = set(['party', 'change_control', 'date'])
        # end hacking

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

    def annotate_document(self, file_name, provision_set=None):
        if not provision_set:
            provision_set = self.provisions
            
        eb_antdoc = ebtext2antdoc.doc_to_ebantdoc(file_name, self.work_dir)

        # this execute the annotators in parallel
        ant_result_dict = self.run_annotators_in_parallel(eb_antdoc, provision_set)
        return ant_result_dict

    def test_annotators(self, txt_fns_file_name, provision_set):
        if not provision_set:
            provision_set = self.provisions
            logging.info('reset provision list: {}'.format(provision_set))

        ebantdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fns_file_name,
                                                               self.work_dir)

        annotations = {}
        with concurrent.futures.ThreadPoolExecutor(8) as executor:
            future_to_provision = {executor.submit(test_provision,
                                                   self.provision_annotator_map[provision],
                                                   ebantdoc_list): 
                                   provision for provision in provision_set}
            for future in concurrent.futures.as_completed(future_to_provision):
                provision = future_to_provision[future]
                data = future.result()
                annotations[provision] = data
        return annotations
        
        
    #
    # custom_train_provision_and_evaluate
    #
    def custom_train_provision_and_evaluate(self, txt_fn_list, provision,
                                            custom_model_dir):

        logging.info("txt_fn_list: '{}'".format(txt_fn_list))

        model_file_name = '{}/{}_scutclassifier.pkl'.format(custom_model_dir,
                                                            provision)
        logging.info("custom_mode_file: '{}'".format(model_file_name))
        
        eb_classifier = scutclassifier.ShortcutClassifier(provision)
        eb_annotator = ebtrainer.train_eval_annotator(provision,
                                                      txt_fn_list,
                                                      self.work_dir,
                                                      custom_model_dir,
                                                      model_file_name,
                                                      eb_classifier,
                                                      custom_training_mode=True)

        # update the hashmap of classifier
        old_provision_annotator = self.provision_annotator_map.get(provision)
        if old_provision_annotator:
            logging.info("Updating annotator, '{}', {}.".format(provision, model_file_name))
        else:
            logging.info("Adding annotator, '{}', {}.".format(provision, model_file_name))
            self.provisions.add(provision)
        self.provision_annotator_map[provision] = eb_annotator

        return eb_annotator.get_eval_status()
