from collections import defaultdict
from datetime import datetime
import json
import logging
import pprint
import time

from typing import Dict, List, Tuple

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from kirke.docstruct import docutils, fromtomapper
from kirke.eblearn import baseannotator, ebpostproc
from kirke.utils import evalutils, strutils


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
class RuleAnnotator:

    def __init__(self,
                 label,
                 *,
                 doclist_to_antdoc_list,
                 docs_to_samples,
                 rule_engine,
                 post_process):
        self.label = label
        self.threshold = 0.5
        
        # used for training
        self.doclist_to_antdoc_list = doclist_to_antdoc_list
        self.docs_to_samples = docs_to_samples
        self.rule_engine = rule_engine
        self.post_process_list = post_process

        self.annotator_status = {'label': label}


    def test_antdoc_list(self, ebantdoc_list, threshold=0.5):
        logging.debug('RuleAnnotator.test_antdoc_list')

        # pylint: disable=C0103
        tp, fn, fp, tn = 0, 0, 0, 0

        for ebantdoc in ebantdoc_list:

            prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                                   if hant.label == self.label]

            ant_list = self.annotate_antdoc(ebantdoc,
                                            prov_human_ant_list=prov_human_ant_list)

            # print("\nfn: {}".format(ebantdoc.file_id))
            # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
            # pred_prob_start_end_list, txt)
            if self.label in PROVISION_EVAL_ANYMATCH_SET:
                xtp, xfn, xfp, xtn = \
                    evalutils.calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list,
                                                                     ant_list,
                                                                     ebantdoc,
                                                                     # threshold,
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

        self.annotator_status['eval_status'] = {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                               'fn': fn, 'tp': tp},
                                                'threshold': self.threshold,
                                                'prec': prec, 'recall': recall, 'f1': f1}


        return self.annotator_status

    def test_antdoc(self, ebantdoc, dir_work='dir-work'):
        logging.debug('RuleAnnotator.test_antdoc')

        ant_list = self.annotate_antdoc(ebantdoc, dir_work=dir_work)
        # print("ant_list: {}".format(ant_list))
        prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                               if hant.label == self.label]
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

    # returns samples, label_list, group_id_list
    def documents_to_samples(self, antdoc_list, label: str):
        return self.docs_to_samples.documents_to_samples(antdoc_list, label)


    def annotate_antdoc(self,
                        antdoc,
                        *,
                        prov_human_ant_list=None) -> List[Dict]:
        logging.info('annotate_antdoc({})...'.format(antdoc.file_id))
        
        start_time = time.time()
        # label_list, group_id_list are ignored
        samples, _, _ = self.docs_to_samples.documents_to_samples([antdoc])

        if not samples:
            return []
        
        prob_samples = self.rule_engine.apply_rules(samples)

        # perform merging operations, such as adjacent positive lines
        # this can also filter out negative samples
        for post_process_x in self.post_process_list:
            prob_samples = post_process_x.apply_post_process(prob_samples)

        prov_annotations = []
        for prob, sample in prob_samples:
            if prob >= self.threshold:
                start = sample['start']
                end = sample['end']
                prov_annotations.append({'label': self.label,
                                         'start': start,
                                         'end': end,
                                         'span_list': [{'start': start,
                                                        'end': end}],
                                         'prob': prob,
                                         'text': sample['text']})

        end_time = time.time()
        logging.debug("annotate_antdoc(%s, %s) took %.0f msec",
                      self.label, antdoc.file_id, (end_time - start_time) * 1000)

        return prov_annotations
    

    def print_eval_status(self, model_dir):
        
        eval_status = {'label': self.label}
        # eval_status['classifier_status'] = self.classifier_status['eval_status']
        # there is no classifier_status for ruleannotator
        eval_status['classifier_status'] = self.annotator_status['eval_status']
        eval_status['annotator_status'] = self.annotator_status['eval_status']
        pprint.pprint(eval_status)

        model_status_fn = model_dir + '/' +  self.label + ".status"
        strutils.dumps(json.dumps(eval_status), model_status_fn)

        with open('label_model_stat.tsv', 'a') as pmout:
            # cls_status = self.classifier_status['eval_status']
            # there is no classifier_status for ruleannotator
            cls_status = self.annotator_status['eval_status']
            cls_cfmtx = cls_status['confusion_matrix']
            ant_status = self.annotator_status['eval_status']
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
