from datetime import datetime
import json
import logging
import pprint
import time

from typing import Any, Dict, List, Tuple

# from kirke.eblearn import baseannotator, ebpostproc
from kirke.utils import ebantdoc4, evalutils, strutils

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


PROVISION_EVAL_ANYMATCH_SET = set(['title'])

def adapt_pipeline_params(best_params):
    result = {}
    for param_name, param_val in best_params.items():
        if param_name.startswith('clf__'):
            result[param_name[5:]] = param_val
        else:
            pass  # skip eb_transformer_* and others
    return result


# class RuleAnnotator(baseannotator.BaseAnnotator):
# Because this is rule-based, differ from BaseAnnotator,
# there is no predict-and_evaluate(), predict_antdo(),
# or tran_antdoc_list()
# pylint: disable=too-many-instance-attributes
class RuleAnnotator:

    def __init__(self,
                 label: str,
                 version: str,
                 *,
                 doclist_to_antdoc_list,
                 is_use_corenlp: bool,
                 doc_to_candidates,
                 rule_engine,
                 post_process) -> None:
        # super().__init__(label, 'no description')
        self.label = label
        self.provision = label  # to be consistent with ProvAnnotator in ebannotator.py
        self.version = version

        self.threshold = 0.5

        # used for training
        self.doclist_to_antdoc_list = doclist_to_antdoc_list
        self.is_use_corenlp = is_use_corenlp
        self.doc_to_candidates = doc_to_candidates
        self.rule_engine = rule_engine
        self.post_process_list = post_process

        self.ant_status = {'label': label}  # type: Dict[str, Any]


    def get_is_use_corenlp(self):
        if not hasattr(self, 'is_use_corenlp'):
            self.is_use_corenlp = False
        return self.is_use_corenlp


    # pylint: disable=too-many-locals
    def test_antdoc_list(self,
                         ebantdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                         threshold: float = 0.5) -> Tuple[Dict[str, Any],
                                                          Dict[str, Dict]]:
        logger.debug('RuleAnnotator.test_antdoc_list(), len= %d', len(ebantdoc_list))

        # pylint: disable=C0103
        tp, fn, fp, tn = 0, 0, 0, 0
        log_json = dict()

        for ebantdoc in ebantdoc_list:

            prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                                   if hant.label == self.label]

            ant_list = self.annotate_antdoc(ebantdoc,
                                            prov_human_ant_list=prov_human_ant_list)

            # print("\nfn: {}".format(ebantdoc.file_id))
            # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
            # pred_prob_start_end_list, txt)
            if self.label in PROVISION_EVAL_ANYMATCH_SET:
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
                                                            is_raw_mode=True)
            tp += xtp
            fn += xfn
            fp += xfp
            tn += xtn
            log_json[ebantdoc.get_document_id()] = json_return

        title = "annotate_status, threshold = {}".format(self.threshold)
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)

        self.ant_status['eval_status'] = {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                               'fn': fn, 'tp': tp},
                                          'threshold': self.threshold,
                                          'prec': prec, 'recall': recall, 'f1': f1}

        return self.ant_status, log_json


    # returns candidates, label_list, group_id_list
    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                label: str):
        return self.doc_to_candidates.documents_to_candidates(antdoc_list, label)


    def annotate_antdoc(self,
                        antdoc: ebantdoc4.EbAnnotatedDoc4,
                        *,
                        # pylint: disable=unused-argument
                        prov_human_ant_list=None,
                        # pylint: disable=unused-argument
                        work_dir: str = 'dir-work') -> List[Dict]:
        logger.info('ruleannotator.annotate_antdoc(%s)', antdoc.file_id)

        start_time = time.time()
        # label_list, group_id_list are ignored
        candidates, _, _ = self.doc_to_candidates.documents_to_candidates([antdoc])

        if not candidates:
            return []

        prob_candidates = self.rule_engine.apply_rules(candidates)

        # perform merging operations, such as adjacent positive lines
        # this can also filter out negative candidates
        for post_process_x in self.post_process_list:
            prob_candidates = post_process_x.apply_post_process(prob_candidates)

        prov_annotations = []
        for prob, candidate in prob_candidates:
            if prob >= self.threshold:
                start = candidate['start']
                end = candidate['end']
                prov_annotations.append({'label': self.label,
                                         'start': start,
                                         'end': end,
                                         'span_list': [{'start': start,
                                                        'end': end}],
                                         'prob': prob,
                                         'text': candidate['text']})

        end_time = time.time()
        logger.debug("annotate_antdoc(%s, %s) took %.0f msec",
                     self.label, antdoc.file_id, (end_time - start_time) * 1000)

        return prov_annotations


    def print_eval_status(self, model_dir):

        eval_status = {'label': self.label}
        # eval_status['classifier_status'] = self.classifier_status['eval_status']
        # there is no classifier_status for ruleannotator
        eval_status['pred_status'] = self.ant_status['eval_status']
        eval_status['ant_status'] = self.ant_status['eval_status']
        pprint.pprint(eval_status)

        model_status_fn = model_dir + '/' +  self.label + ".status"
        strutils.dumps(json.dumps(eval_status), model_status_fn)

        with open('label_model_stat.tsv', 'a') as pmout:
            # cls_status = self.classifier_status['eval_status']
            # there is no classifier_status for ruleannotator
            cls_status = self.ant_status['eval_status']
            cls_cfmtx = cls_status['confusion_matrix']
            ant_status = self.ant_status['eval_status']
            ant_cfmtx = ant_status['confusion_matrix']

            timestamp = int(time.time())
            aline = [datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                     str(timestamp),
                     self.label,
                     cls_cfmtx['tp'], cls_cfmtx['fn'], cls_cfmtx['fp'], cls_cfmtx['tn'],
                     cls_status['prec'], cls_status['recall'], cls_status['f1'],
                     ant_cfmtx['tp'], ant_cfmtx['fn'], ant_cfmtx['fp'], ant_cfmtx['tn'],
                     ant_status['threshold'],
                     ant_status['prec'], ant_status['recall'], ant_status['f1']]
            print('\t'.join([str(x) for x in aline]), file=pmout)
