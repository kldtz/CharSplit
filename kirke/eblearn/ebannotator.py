import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import traceback

from kirke.docstruct import fromtomapper
from kirke.eblearn import ebpostproc
from kirke.utils import ebantdoc4, evalutils, strutils
from kirke.utils.ebsentutils import ProvisionAnnotation

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


PROVISION_EVAL_ANYMATCH_SET = set(['title'])

class ProvisionAnnotator:

    def __init__(self,
                 prov_classifier,
                 work_dir: str,
                 threshold: Optional[float] = None,
                 nbest: int = -1) -> None:
        self.provision_classifier = prov_classifier
        self.provision = prov_classifier.provision
        self.nbest = nbest
        if threshold is not None:  # allow overrides from provclassifier.py
            self.threshold = threshold
        else:
            self.threshold = prov_classifier.threshold
        self.work_dir = work_dir
        # this is set after training
        self.eval_status = {}  # type: Dict

    def get_eval_status(self):
        return self.eval_status


    def call_confusion_matrix(self,
                              prov_human_ant_list: List[ProvisionAnnotation],
                              ant_list: List[Dict],
                              ebantdoc: ebantdoc4.EbAnnotatedDoc4,
                              threshold: float) \
                              -> Tuple[int, int, int, int,
                                       Dict[str, List[Tuple[int, int, str, float, str]]]]:

        if self.provision in PROVISION_EVAL_ANYMATCH_SET:
            xtp, xfn, xfp, xtn, json_return = \
                evalutils.calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list,
                                                                 ant_list,
                                                                 ebantdoc.file_id,
                                                                 ebantdoc.get_text())
        else:
            xtp, xfn, xfp, xtn, _, json_return = \
                evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                        ant_list,
                                                        ebantdoc.file_id,
                                                        ebantdoc.get_text(),
                                                        threshold,
                                                        is_raw_mode=False)
        return xtp, xfn, xfp, xtn, json_return

    # ProvisionAnnotator does not train, it only predict
    # Training is available only for classifiers
    # def train(self):
    #    pass
    # pylint: disable=R0914
    def test_antdoc_list(self,
                         ebantdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                         specified_threshold: Optional[float] = None) \
                         -> Tuple[Dict[str, Any],
                                  Dict[str, Dict]]:
        logger.debug('test_document_list')
        if specified_threshold is None:
            threshold = self.threshold
        else:
            threshold = specified_threshold

        # pylint: disable=C0103
        tp, fn, fp, tn = 0, 0, 0, 0
        log_json = dict()
        for ebantdoc in ebantdoc_list:
            prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                                   if hant.label == self.provision]
            try:
                pred_list, full_list = self.annotate_antdoc(ebantdoc,
                                                            specified_threshold=threshold,
                                                            prov_human_ant_list=prov_human_ant_list)
                ant_list = self.recover_false_negatives(prov_human_ant_list,
                                                        ebantdoc.get_text(),
                                                        self.provision,
                                                        full_list)
            # pylint: disable=broad-except, unused-variable
            except Exception as e:
                logger.warning('Faile to annotat_antdoc(%s) in test_antdoc_list.',
                               ebantdoc.file_id)
                raise
            # pylint: disable=unreachable, pointless-string-statement

            xtp, xfn, xfp, xtn, json_return = self.call_confusion_matrix(prov_human_ant_list,
                                                                         ant_list,
                                                                         ebantdoc,
                                                                         threshold)
            if self.get_nbest() > 0:
                ant_list = self.recover_false_negatives(prov_human_ant_list,
                                                        ebantdoc.get_text(),
                                                        self.provision,
                                                        pred_list)
                xtp, xfn, xfp, xtn, _ = self.call_confusion_matrix(prov_human_ant_list,
                                                                   ant_list,
                                                                   ebantdoc,
                                                                   threshold)
                # adjust for top n extractions
                nbest_diff = max(xtp + xfn - self.get_nbest(), 0)
                xfn = max(xfn - nbest_diff, 0)

            tp += xtp
            fn += xfn
            fp += xfp
            tn += xtn
            log_json[ebantdoc.get_document_id()] = json_return

        title = "annotate_status, threshold = {}".format(threshold)
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)

        tmp_eval_status = {'ant_status': {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                               'fn': fn, 'tp': tp},
                                          'threshold': threshold,
                                          'prec': prec, 'recall': recall, 'f1': f1}}
        return tmp_eval_status, log_json


    # pylint: disable=no-self-use
    def recover_false_negatives(self,
                                prov_human_ant_list: List[ProvisionAnnotation],
                                doc_text: str,
                                provision: str,
                                ant_result: List[Dict]) \
        -> List[Dict]:
        if not prov_human_ant_list:
            return ant_result
        for ant in prov_human_ant_list:
            if not evalutils.find_annotation_overlap_x2(ant.start, ant.end, ant_result):
                clean_text = strutils.sub_nltab_with_space(doc_text[ant.start:ant.end])
                fn_ant = ebpostproc.to_ant_result_dict(label=provision,
                                                       prob=-1.0,
                                                       start=ant.start,
                                                       end=ant.end,
                                                       text=clean_text)
                ant_result.append(fn_ant)
        return ant_result

    def get_nbest(self) -> int:
        """Return nbest.

        If old bespoke model, this field won't be there.  We will create one then.
        """
        if not hasattr(self.provision_classifier, 'nbest'):
            self.provision_classifier.nbest = -1
        return self.provision_classifier.nbest


    def annotate_antdoc(self,
                        eb_antdoc,
                        specified_threshold: Optional[float] = None,
                        prov_human_ant_list: Optional[List[ProvisionAnnotation]] = None) \
        -> Tuple[List[Dict], List[Dict]]:

        if prov_human_ant_list is None:
            prov_human_ant_list = []

        attrvec_list = eb_antdoc.get_attrvec_list()

        # manually set the threshold
        if specified_threshold is None:
            threshold = self.threshold
        else:
            threshold = specified_threshold

        start_time = time.time()
        prob_list = self.provision_classifier.predict_antdoc(eb_antdoc, self.work_dir)
        end_time = time.time()
        logger.debug('annotate_antdoc(%s, %s) took %.0f msec, eb_antr',
                     self.provision, eb_antdoc.file_id, (end_time - start_time) * 1000)

        try:
            # mapping the offsets in prov_human_ant_list from raw_text to nlp_text
            fromto_mapper = fromtomapper.FromToMapper('raw_text to nlp_text offset mapper',
                                                      eb_antdoc.get_origin_sx_lnpos_list(),
                                                      eb_antdoc.get_nlp_sx_lnpos_list())
            adj_prov_human_ant_list = \
                fromto_mapper.adjust_provants_fromto_offsets(prov_human_ant_list)
        except IndexError:
            error = traceback.format_exc()
            logger.warning("IndexError, adj_prov_human_ant_list, %s", eb_antdoc.file_id)
            logger.warning(error)
            # move on, probably because there is no input
            adj_prov_human_ant_list = prov_human_ant_list
        prov = self.provision
        prob_attrvec_list = list(zip(prob_list, attrvec_list))
        prov_annotations, unused_threshold = \
            ebpostproc.obtain_postproc(prov).post_process(eb_antdoc.get_nlp_text(),
                                                          prob_attrvec_list,
                                                          threshold,
                                                          nbest=self.get_nbest(),
                                                          provision=prov,
                                                          # pylint: disable=line-too-long
                                                          prov_human_ant_list=adj_prov_human_ant_list)

        if self.nbest > 0:
            full_annotations, _ = \
                ebpostproc.obtain_postproc(prov).post_process(eb_antdoc.get_nlp_text(),
                                                              prob_attrvec_list,
                                                              threshold,
                                                              nbest=-1,
                                                              provision=prov,
                                                              # pylint: disable=line-too-long
                                                              prov_human_ant_list=adj_prov_human_ant_list)
        else:
            full_annotations = copy.deepcopy(prov_annotations)

        try:
            fromto_mapper = fromtomapper.FromToMapper('an offset mapper',
                                                      eb_antdoc.get_nlp_sx_lnpos_list(),
                                                      eb_antdoc.get_origin_sx_lnpos_list())
            # this is an in-place modification
            # This add span list.  If there is a gap, one will be inserted into span list.
            fromto_mapper.adjust_fromto_offsets(prov_annotations)
            fromto_mapper.adjust_fromto_offsets(full_annotations)
        except IndexError:
            error = traceback.format_exc()
            logger.warning("IndexError, adj_fromto_offsets, %s", eb_antdoc.file_id)
            logger.warning(error)
            # move on, probably because there is no input

        update_text_with_span_list(prov_annotations, eb_antdoc.text)
        update_text_with_span_list(full_annotations, eb_antdoc.text)
        return prov_annotations, full_annotations

# this is destructive
def update_text_with_span_list(prov_annotations, doc_text):
    # print("prov_annotations: {}".format(prov_annotations))
    for ant in prov_annotations:
        if ant.get('span_list'):  # check if the cached version is very old
            tmp_span_text_list = []
            for span in ant['span_list']:
                start = span['start']
                end = span['end']
                tmp_span_text_list.append(doc_text[start:end])
            ant['text'] = ' '.join(tmp_span_text_list)
