import logging
import time

from kirke.docstruct import docutils, fromtomapper
from kirke.eblearn import ebpostproc
from kirke.utils import evalutils

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
    def test_antdoc_list(self,
                         ebantdoc_list: List[ebantdoc2.EbAnnotatedDoc2],
                         threshold: Optional[float] =None) -> Tuple[Dict[str, Any],
                                                                    Dict[str, Dict]]:    
        logging.debug('test_document_list')
        if not threshold:
            threshold = self.threshold
        # pylint: disable=C0103
        tp, fn, fp, tn = 0, 0, 0, 0

        for ebantdoc in ebantdoc_list:
            #print('ebantdoc.fileid = {}'.format(ebantdoc.file_id))
            # print("ant_list: {}".format(ant_list))
            prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                                   if hant.label == self.provision]
            ant_list = self.annotate_antdoc(ebantdoc, threshold=self.threshold, prov_human_ant_list=prov_human_ant_list)
            # print("\nfn: {}".format(ebantdoc.file_id))
            # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
            # pred_prob_start_end_list, txt)
            if self.provision in PROVISION_EVAL_ANYMATCH_SET:
                xtp, xfn, xfp, xtn = \
                    evalutils.calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list,
                                                                     ant_list,
                                                                     ebantdoc,
                                                                     threshold,
                                                                     diagnose_mode=True)
            else:
                xtp, xfn, xfp, xtn = \
                    evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                            ant_list,
                                                            ebantdoc,
                                                            threshold,
                                                            diagnose_mode=True)
            tp += xtp
            fn += xfn
            fp += xfp
            tn += xtn

        title = "annotate_status, threshold = {}".format(self.threshold)
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)

        tmp_eval_status = {'ant_status': {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                               'fn': fn, 'tp': tp},
                                          'threshold': self.threshold,
                                          'prec': prec, 'recall': recall, 'f1': f1}}

        return tmp_eval_status

    def test_antdoc(self, ebantdoc, threshold=None):
        logging.debug('test_document')

        ant_list = self.annotate_antdoc(ebantdoc, threshold)
        # print("ant_list: {}".format(ant_list))
        prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                               if hant.label == self.provision]
        # print("human_list: {}".format(prov_human_ant_list))

        # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
        # pred_prob_start_end_list, txt)
        # pylint: disable=C0103
        tp, fn, fp, tn = evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                                 ant_list,
                                                                 ebantdoc.get_text())

        title = "annotate_status, threshold = {}".format(self.threshold)
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)

        tmp_eval_status = {'ant_status': {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                               'fn': fn, 'tp': tp},
                                          'threshold': self.threshold,
                                          'prec': prec, 'recall': recall, 'f1': f1}}

        return tmp_eval_status

     # pylint: disable=no-self-use
    def recover_false_negatives(self, prov_human_ant_list, doc_text, provision, ant_result):
        if not prov_human_ant_list:
            return ant_result
        for ant in prov_human_ant_list:
            if not evalutils.find_annotation_overlap_x2(ant.start, ant.end, ant_result):
                clean_text = strutils.remove_nltab(doc_text[ant.start:ant.end])
                fn_ant = ebpostproc.to_ant_result_dict(label=provision,
                                                       prob=0.0,
                                                       start=ant.start,
                                                       end=ant.end,
                                                       text=clean_text)
                ant_result.append(fn_ant)
        return ant_result

    def annotate_antdoc(self, eb_antdoc, threshold=None, prov_human_ant_list=None) \
        -> Tuple[List[Dict], float]:
        # attrvec_list = eb_antdoc.get_attrvec_list()
        # ebsent_list = eb_antdoc.get_ebsent_list()
        # print("txt_fn = '{}', vec_size= {}".format(eb_antdoc.file_id,
        # len(eb_antdoc.get_attrvec_list())))
        if prov_human_ant_list is None:
            prov_human_ant_list = []
        attrvec_list = eb_antdoc.get_attrvec_list()

        # manually set the threshold
        # self.provision_classifier.threshold = 0.5
        if threshold != None:
            self.threshold = threshold

        start_time = time.time()
        prob_list = self.provision_classifier.predict_antdoc(eb_antdoc, self.work_dir)
        end_time = time.time()
        logging.debug("annotate_antdoc(%s, %s) took %.0f msec",
                      self.provision, eb_antdoc.file_id, (end_time - start_time) * 1000)
        try:
            # mapping the offsets in prov_human_ant_list from raw_text to nlp_text
            fromto_mapper = fromtomapper.FromToMapper('raw_text to nlp_text offset mapper',
                                                      eb_antdoc.origin_sx_lnpos_list,
                                                      eb_antdoc.nlp_sx_lnpos_list)
            adj_prov_human_ant_list = \
                fromto_mapper.adjust_provants_fromto_offsets(prov_human_ant_list)
        except IndexError:
            error = traceback.format_exc()
            logging.warning("IndexError, adj_prov_human_ant_list, %s", eb_antdoc.file_id)
            logging.warning(error)
            # move on, probably because there is no input
            adj_prov_human_ant_list = prov_human_ant_list
        prov = self.provision
        prob_attrvec_list = list(zip(prob_list, attrvec_list))
        prov_annotations, threshold = \
            ebpostproc.obtain_postproc(prov).post_process(eb_antdoc.nlp_text,
                                                          prob_attrvec_list,
                                                          threshold,
                                                          provision=prov,
                                                          prov_human_ant_list=\
                                                              adj_prov_human_ant_list)
        try:
            fromto_mapper = fromtomapper.FromToMapper('an offset mapper',
                                                      eb_antdoc.nlp_sx_lnpos_list,
                                                      eb_antdoc.origin_sx_lnpos_list)
            # this is an in-place modification
            fromto_mapper.adjust_fromto_offsets(prov_annotations)
        except IndexError:
            error = traceback.format_exc()
            logging.warning("IndexError, adj_fromto_offsets, %s", eb_antdoc.file_id)
            logging.warning(error)
            # move on, probably because there is no input
        update_text_with_span_list(prov_annotations, eb_antdoc.text)
        prov_annotations = self.recover_false_negatives(prov_human_ant_list,
                                                        eb_antdoc.text,
                                                        prov,
                                                        prov_annotations)
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
