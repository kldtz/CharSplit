import logging
import time
from typing import Dict, List, Tuple

from kirke.docstruct import docutils, fromtomapper
from kirke.eblearn import ebpostproc
from kirke.utils import evalutils, strutils

PROVISION_EVAL_ANYMATCH_SET = set(['title'])

class ProvisionAnnotator:

    def __init__(self, prov_classifier, work_dir, threshold=None):
        self.provision_classifier = prov_classifier
        self.provision = prov_classifier.provision
        if threshold is not None:  # allow overrides from provclassifier.py
            self.threshold = threshold
        else:
            self.threshold = prov_classifier.threshold
        self.work_dir = work_dir
        self.eval_status = {}  # this is set after training

    def get_eval_status(self):
        return self.eval_status

    # ProvisionAnnotator does not train, it only predict
    # Training is available only for classifiers
    # def train(self):
    #    pass
    # pylint: disable=R0914
    def test_antdoc_list(self, ebantdoc_list, threshold=None) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        logging.debug('test_document_list')
        if not threshold:
            threshold = self.threshold
        # pylint: disable=C0103
        tp, fn, fp, tn = 0, 0, 0, 0
        log_json = {}
        for ebantdoc in ebantdoc_list:
            #print('ebantdoc.fileid = {}'.format(ebantdoc.file_id))
            # print("ant_list: {}".format(ant_list))
            prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
            # prov_human_ant_list = [hant for hant in ebantdoc.para_prov_ant_list
                                   if hant.label == self.provision]
            ant_list, threshold = self.annotate_antdoc(ebantdoc, threshold=threshold, prov_human_ant_list=prov_human_ant_list)
            # print("\nfname: {}".format(ebantdoc.file_id))
            # print("\nant_list: {}".format(ant_list))
            # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
            # pred_prob_start_end_list, txt)
            if self.provision in PROVISION_EVAL_ANYMATCH_SET:
                xtp, xfn, xfp, xtn, json_return = \
                    evalutils.calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list,
                                                                     ant_list,
                                                                     ebantdoc,
                                                                     # threshold,
                                                                     diagnose_mode=True)
            else:
                xtp, xfn, xfp, xtn, json_return = \
                    evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                            ant_list,
                                                            ebantdoc,
                                                            threshold,
                                                            diagnose_mode=True)
            tp += xtp
            fn += xfn
            fp += xfp
            tn += xtn
            log_json[ebantdoc.file_id] = json_return

        title = "annotate_status, threshold = {}".format(threshold)
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)

        tmp_eval_status = {'ant_status': {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                               'fn': fn, 'tp': tp},
                                          'threshold': threshold,
                                          'prec': prec, 'recall': recall, 'f1': f1}}

        return tmp_eval_status, log_json

    def recover_false_negatives(self, prov_human_ant_list, doc_text, provision, ant_result):
        if not prov_human_ant_list:
            return ant_result
        for ant in prov_human_ant_list:
            if not evalutils.find_annotation_overlap_x2(ant.start, ant.end, ant_result):
                fn_ant = ebpostproc.to_ant_result_dict(label=provision,
                                                       prob=0.0,
                                                       start=ant.start,
                                                       end=ant.end,
                                                       text=strutils.remove_nltab(doc_text[ant.start:ant.end]))
                ant_result.append(fn_ant)
        return ant_result

    def annotate_antdoc(self, eb_antdoc, threshold=None, prov_human_ant_list=None):
        # attrvec_list = eb_antdoc.get_attrvec_list()
        # ebsent_list = eb_antdoc.get_ebsent_list()
        # print("txt_fn = '{}', vec_size= {}".format(eb_antdoc.file_id,
        # len(eb_antdoc.get_attrvec_list())))

        attrvec_list = eb_antdoc.get_attrvec_list()
        # manually set the threshold
        # self.provision_classifier.threshold = 0.5
        if threshold != None:
            self.threshold = threshold
        else:
            threshold = self.threshold
        start_time = time.time()
        prob_list = self.provision_classifier.predict_antdoc(eb_antdoc, self.work_dir)
        end_time = time.time()
        logging.debug("annotate_antdoc(%s, %s) took %.0f msec",
                      self.provision, eb_antdoc.file_id, (end_time - start_time) * 1000)

        prov = self.provision
        prob_attrvec_list = list(zip(prob_list, attrvec_list))
        prov_annotations, threshold = ebpostproc.obtain_postproc(prov).post_process(eb_antdoc.text,
                                                                                    prob_attrvec_list,
                                                                                    threshold,
                                                                                    provision=prov,
                                                                                    prov_human_ant_list=prov_human_ant_list)

        # print("eb_antdoc.from_list: {}".format(eb_antdoc.from_list))
        # print("eb_antdoc.to_list: {}".format(eb_antdoc.to_list))
        # for fr_sxlnpos, to_sxlnpos in zip(eb_antdoc.origin_sx_lnpos_list, eb_antdoc.nlp_sx_lnpos_list):
        #    print("35234 origin: {}, nlp: {}".format(fr_sxlnpos, to_sxlnpos))

        fromto_mapper = fromtomapper.FromToMapper('an offset mapper', eb_antdoc.nlp_sx_lnpos_list, eb_antdoc.origin_sx_lnpos_list)
        # this is an in-place modification
        fromto_mapper.adjust_fromto_offsets(prov_annotations)
        update_text_with_span_list(prov_annotations, eb_antdoc.text)

        prov_annotations = self.recover_false_negatives(prov_human_ant_list, eb_antdoc.text, prov, prov_annotations)

        return prov_annotations, threshold

# this is destructive
def update_text_with_span_list(prov_annotations, doc_text):
    # print("prov_annotations: {}".format(prov_annotations))
    for ant in prov_annotations:
        tmp_span_text_list = []
        for span in ant['span_list']:
            start = span['start']
            end = span['end']
            tmp_span_text_list.append(doc_text[start:end])
        ant['text'] = ' '.join(tmp_span_text_list)
