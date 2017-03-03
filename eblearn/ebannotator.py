import logging

from eblearn import ebpostproc
from utils import evalutils


class ProvisionAnnotator:

    def __init__(self, prov_classifier, work_dir):
        self.provision_classifier = prov_classifier
        self.provision = prov_classifier.provision
        self.threshold = prov_classifier.threshold
        self.work_dir = work_dir
        self.eval_status = {}  # this is set after training

    def get_eval_status(self):
        return self.eval_status

    # ProvisionAnnotator does not train, it only predict
    # Training is available only for classifiers
    # def train(self):
    #    pass

    def test_antdoc_list(self, ebantdoc_list, diagnose_mode=False):
        logging.info('test_document_list')

        tp, fn, fp, tn = 0, 0, 0, 0
        
        for ebantdoc in ebantdoc_list:
            ant_list = self.annotate_antdoc(ebantdoc)
            # print("ant_list: {}".format(ant_list))
            prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list if hant.label == self.provision]

            # print("\nfn: {}".format(ebantdoc.file_id))
            # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list, pred_prob_start_end_list, txt)
            xtp, xfn, xfp, xtn = evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                                         ant_list,
                                                                         ebantdoc.get_text())

            tp += xtp
            fn += xfn
            fp += xfp
            tn += xtn

        title = "annotate_status, threshold = {}".format(self.threshold)
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)

        tmp_eval_status = {'ant_status': {'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
                                          'threshold': self.threshold,
                                          'prec': prec, 'recall': recall, 'f1': f1}}

        return tmp_eval_status

    def test_antdoc(self, ebantdoc, diagnose_mode=False):
        logging.info('test_document')

        ant_list = self.annotate_antdoc(ebantdoc)
        # print("ant_list: {}".format(ant_list))
        prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list if hant.label == self.provision]
        # print("human_list: {}".format(prov_human_ant_list))

        # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list, pred_prob_start_end_list, txt)
        tp, fn, fp, tn = evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                                 ant_list,
                                                                 ebantdoc.get_text())

        title = "annotate_status, threshold = {}".format(self.threshold)
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)

        tmp_eval_status = {'ant_status': {'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
                                      'threshold': self.threshold,
                                      'prec': prec, 'recall': recall, 'f1': f1}}

        return tmp_eval_status
    

    def annotate_antdoc(self, eb_antdoc):
        # attrvec_list = eb_antdoc.get_attrvec_list()
        # ebsent_list = eb_antdoc.get_ebsent_list()
        # print("txt_fn = '{}', vec_size= {}, ant_list = {}".format(txt_fn, len(instance_list), ant_list))
        ebsent_list = eb_antdoc.get_ebsent_list()
        
        prob_list = self.provision_classifier.predict_antdoc(eb_antdoc, self.work_dir)


        prov = self.provision
        prob_ebsent_list = list(zip(prob_list, ebsent_list))
        prov_annotations = ebpostproc.obtain_postproc(prov).post_process(prob_ebsent_list,
                                                                         self.threshold,
                                                                         provision=prov)

        return prov_annotations

