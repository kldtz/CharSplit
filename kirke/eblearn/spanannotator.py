from collections import defaultdict
import configparser
from datetime import datetime
import json
import logging
import pprint
import time

# pylint: disable=unused-import
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

# pylint: disable=import-error
from sklearn.model_selection import GridSearchCV, GroupKFold
# pylint: disable=import-error
from sklearn.pipeline import Pipeline

from kirke.eblearn import baseannotator, ebpostproc
from kirke.utils import ebantdoc4, evalutils, strutils

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')

CANDG_CLF_VERSION = config['ebrevia.com']['CANDG_CLF_VERSION']

PROVISION_EVAL_ANYMATCH_SET = set(['title'])

def adapt_pipeline_params(best_params):
    result = {}
    for param_name, param_val in best_params.items():
        if param_name.startswith('clf__'):
            result[param_name[5:]] = param_val
        else:
            pass  # skip eb_transformer_* and others
    return result


def get_model_file_name(provision: str,
                        candidate_type: str,
                        model_dir: str):
    base_model_fname = '{}_{}_annotator.v{}.pkl'.format(provision,
                                                        candidate_type,
                                                        CANDG_CLF_VERSION)
    return "{}/{}".format(model_dir, base_model_fname)


def recover_false_negatives(prov_human_ant_list,
                            doc_text: str,
                            provision: str,
                            ant_result):
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


# pylint: disable=line-too-long
def antdoc_candidatex_list_to_candidatex(antdoc_candidatex_list: List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                                                            List[Dict],
                                                                            List[bool],
                                                                            List[int]]]) \
        -> Tuple[List[Dict], List[bool], List[int]]:
    all_candidates = []  # type: List[Dict]
    all_candidate_labels = []  # type: List[bool]
    all_group_ids = []  # type: List[int]
    #change to list of lists to preserve antdocs?
    for ant_candidatex in antdoc_candidatex_list:
        unused_antdoc, candidates, candidate_labels, group_ids = ant_candidatex
        all_candidates.extend(candidates)
        all_candidate_labels.extend(candidate_labels)
        all_group_ids.extend(group_ids)

    return all_candidates, all_candidate_labels, all_group_ids


class SpanAnnotator(baseannotator.BaseAnnotator):

    # pylint: disable=too-many-instance-attributes, too-many-locals
    def __init__(self,
                 provision: str,
                 candidate_type: str,
                 version: str,
                 nbest: int,
                 *,
                 doclist_to_antdoc_list,
                 is_use_corenlp: bool,
                 doc_to_candidates,
                 candidate_transformers,
                 pipeline,
                 doc_postproc_list: Optional[List] = None,
                 gridsearch_parameters,
                 # prefer recall over precision
                 threshold: float = 0.2,
                 kfold: int = 3,
                 text_type: str = '') -> None:
        super().__init__(provision, 'no description')
        self.provision = provision
        self.candidate_type = candidate_type
        self.version = version
        self.nbest = nbest
        self.text_type = text_type

        # used for training
        self.doclist_to_antdoc_list = doclist_to_antdoc_list
        self.is_use_corenlp = is_use_corenlp
        self.doc_to_candidates = doc_to_candidates
        self.candidate_transformers = candidate_transformers
        self.pipeline = pipeline
        self.gridsearch_parameters = gridsearch_parameters
        self.threshold = threshold
        self.kfold = kfold
        self.doc_postproc_list = doc_postproc_list if doc_postproc_list else []

        self.best_parameters = {}  # type: Dict[str, Any]
        self.estimator = None

        # these are set after training
        self.classifier_status = {'label': provision}  # type: Dict[str, Any]
        self.ant_status = {'label': provision}  # type: Dict[str, Any]

    def make_bare_copy(self):
        return SpanAnnotator(self.provision,
                             self.candidate_type,
                             self.version,
                             self.nbest,
                             doclist_to_antdoc_list=self.doclist_to_antdoc_list,
                             is_use_corenlp=self.is_use_corenlp,
                             doc_to_candidates=self.doc_to_candidates,
                             candidate_transformers=self.candidate_transformers,
                             pipeline=self.pipeline,
                             doc_postproc_list=self.doc_postproc_list,
                             gridsearch_parameters=self.gridsearch_parameters,
                             # prefer recall over precision
                             threshold=self.threshold,
                             kfold=self.kfold)

    def get_is_use_corenlp(self):
        if not hasattr(self, 'is_use_corenlp'):
            self.is_use_corenlp = False
        return self.is_use_corenlp

    # pylint: disable=too-many-arguments
    def train_candidates(self,
                         candidates: List[Dict],
                         label_list: List[bool],
                         group_id_list: List[int],
                         pipeline: Pipeline,
                         parameters: Dict,
                         work_dir: str) -> None:
        logger.info('spanannotator.train_candidates()...')

        logger.info("Performing grid search...")
        logger.info("parameters: %r", parameters)
        pos_neg_map = defaultdict(int)  # type: DefaultDict[bool, int]
        for label in label_list:
            pos_neg_map[label] += 1
        for label, count in pos_neg_map.items():
            logger.info("train_candidates(), pos_neg_map[%s] = %d", label, count)

        group_kfold = list(GroupKFold(n_splits=self.kfold).split(candidates,
                                                                 label_list,
                                                                 groups=group_id_list))
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, scoring='f1',
                                   verbose=1, cv=group_kfold)

        time_0 = time.time()
        grid_search.fit(candidates, label_list)
        logger.info("done in %0.3fs", (time.time() - time_0))

        logger.info("Best score: %0.3f", grid_search.best_score_)
        logger.info("Best parameters set:")
        self.best_parameters = adapt_pipeline_params(grid_search.best_estimator_.get_params())
        for param_name in sorted(self.best_parameters.keys()):
            logger.info("\t%s: %r", param_name, self.best_parameters[param_name])

        self.estimator = grid_search.best_estimator_


    # ProvisionAnnotator does not train, it only predict
    # Training is available only for classifiers
    # def train(self):
    #    pass
    # pylint: disable=R0914
    def test_antdoc_list(self,
                         ebantdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                         threshold: float,
                         work_dir: str = 'work_dir')  -> Tuple[Dict[str, Any],
                                                               Dict[str, Dict]]:
        logger.debug('spanannotator.test_antdoc_list(), len= %d', len(ebantdoc_list))
        if not threshold:
            threshold = self.threshold
        # pylint: disable=C0103
        fallout, tp, fn, fp, tn = 0, 0, 0, 0, 0
        log_json = dict()

        for ebantdoc in ebantdoc_list:
            prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                                   if hant.label == self.provision]

            ant_list, threshold = self.annotate_antdoc(ebantdoc,
                                                       threshold=threshold,
                                                       prov_human_ant_list=prov_human_ant_list,
                                                       work_dir=work_dir)
            # print("\nfn: {}".format(ebantdoc.file_id))
            # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
            # pred_prob_start_end_list, txt)
            if self.provision in PROVISION_EVAL_ANYMATCH_SET:
                xtp, xfn, xfp, xtn, json_return = \
                    evalutils.calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list,
                                                                     ant_list,
                                                                     ebantdoc.file_id,
                                                                     ebantdoc.get_text())
            else:
                xtp, xfn, xfp, xtn, xfallout, json_return = \
                    evalutils.calc_doc_ant_confusion_matrix(prov_human_ant_list,
                                                            ant_list,
                                                            ebantdoc.file_id,
                                                            ebantdoc.get_text(),
                                                            threshold,
                                                            is_raw_mode=False)
            tp += xtp
            fn += xfn
            fp += xfp
            tn += xtn
            fallout += xfallout
            log_json[ebantdoc.get_document_id()] = json_return

        title = "annotate_status, threshold = {}".format(self.threshold)
        prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp, title)
        max_recall = (tp + fn - fallout) / (tp + fn)
        print("MAX RECALL =", max_recall, "FALLOUT =", fallout)
        self.ant_status['eval_status'] = {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                               'fn': fn, 'tp': tp},
                                          'threshold': self.threshold,
                                          'prec': prec, 'recall': recall, 'f1': f1}


        return self.ant_status, log_json


    # returns candidates, label_list, group_id_list
    # this also enriches candidates using additional self.candidate_transformers
    def documents_to_candidates(self,
                                antdoc_list: List[ebantdoc4.EbAnnotatedDoc4],
                                label: Optional[str] = None) \
                                -> List[Tuple[ebantdoc4.EbAnnotatedDoc4,
                                              List[Dict],
                                              List[bool],
                                              List[int]]]:
        result = self.doc_to_candidates.documents_to_candidates(antdoc_list, label)
        return result

    def annotate_antdoc(self,
                        eb_antdoc: ebantdoc4.EbAnnotatedDoc4,
                        *,
                        threshold: Optional[float] = None,
                        prov_human_ant_list: Optional[List] = None,
                        work_dir: str = 'dir-work') -> Tuple[List[Dict], float]:
        """Annotate a document.

        Will always run recover_false_negatives() if there is human annotation.
        """

        # manually set the threshold
        # self.provision_classifier.threshold = 0.5
        if threshold is None:
            threshold = self.threshold
        nbest = self.nbest
        start_time = time.time()
        candidates, unused_prob_list = self.predict_antdoc(eb_antdoc, work_dir)
        end_time = time.time()
        logger.info('annotate_antdoc(%s, %s) took %.0f msec, span_antr',
                    self.provision, eb_antdoc.file_id, (end_time - start_time) * 1000)

        prov_annotations = candidates
        x_threshold = threshold

        prov_annotations = recover_false_negatives(prov_human_ant_list,
                                                   eb_antdoc.get_text(),
                                                   self.provision,
                                                   prov_annotations)
        # If there is no human annotation, must be normal annotation.
        # Remove anything below threshold
        if not prov_human_ant_list:
            prov_annotations = [ant for ant in prov_annotations if ant['prob'] >= x_threshold]
        if nbest > 0:
            return prov_annotations[:nbest], x_threshold
        return prov_annotations, x_threshold

    def get_eval_status(self):
        eval_status = {'label': self.provision}
        # eval_status['pred_status'] = self.classifier_status['eval_status']
        eval_status['ant_status'] = self.ant_status['eval_status']
        return eval_status

    def print_eval_status(self, model_dir: str, model_num: int):

        eval_status = {'label': self.provision}
        eval_status['ant_status'] = self.ant_status['eval_status']
        pprint.pprint(eval_status)

        model_status_fn = '{}/{}.{}.status'.format(model_dir, self.provision, model_num)
        strutils.dumps(json.dumps(eval_status), model_status_fn)

        with open('label_model_stat.tsv', 'a') as pmout:
            cls_status = self.ant_status['eval_status']
            cls_cfmtx = cls_status['confusion_matrix']
            ant_status = self.ant_status['eval_status']
            ant_cfmtx = ant_status['confusion_matrix']

            timestamp = int(time.time())
            aline = [datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
                     str(timestamp),
                     self.provision,
                     cls_cfmtx['tp'], cls_cfmtx['fn'], cls_cfmtx['fp'], cls_cfmtx['tn'],
                     cls_status['prec'], cls_status['recall'], cls_status['f1'],
                     ant_cfmtx['tp'], ant_cfmtx['fn'], ant_cfmtx['fp'], ant_cfmtx['tn'],
                     ant_status['threshold'],
                     ant_status['prec'], ant_status['recall'], ant_status['f1']]
            print('\t'.join([str(x) for x in aline]), file=pmout)

    # return a list of candidates, a list of labels
    def predict_antdoc(self,
                       eb_antdoc: ebantdoc4.EbAnnotatedDoc4,
                       work_dir: str) -> Tuple[List[Dict[str, Any]], List[float]]:
        # logger.info('prov = %s, predict_antdoc(%s)', self.provision, eb_antdoc.file_id)
        text = eb_antdoc.get_text()
        # label_list, group_id_list are ignored
        antdoc_candidatex_list = self.documents_to_candidates([eb_antdoc])
        candidates, unused_label_list, unused_group_ids = \
                antdoc_candidatex_list_to_candidatex(antdoc_candidatex_list)

        if not candidates:
            return [], []

        probs = [] # type: List[float]
        if self.estimator:
            probs = self.estimator.predict_proba(candidates)[:, 1]

        # to indicate which type of annotation this is
        for unused_i, (candidate, prob) in enumerate(zip(candidates, probs)):
            candidate['label'] = self.provision
            candidate['prob'] = prob
            candidate['text'] = text[candidate['start']:candidate['end']]

        # apply post processing, such as date normalization
        # in case there is any bad apple, with 'reject' == True
        for post_proc in self.doc_postproc_list:
            candidates = post_proc.doc_postproc(candidates, self.nbest)

        return candidates, probs

    def predict_and_evaluate(self,
                             candidates: List[Dict],
                             label_list: List[bool],
                             work_dir: str,
                             is_debug: bool = False):
        logger.info('spanannotator.predict_and_evaluate()...')
        pos_neg_map = defaultdict(int)  # type: DefaultDict[bool, int]
        for label in label_list:
            pos_neg_map[label] += 1
        for label, count in pos_neg_map.items():
            print("predict_and_evaluate(), pos_neg_map[{}] = {}".format(label, count))

        # y_te has to be a list of int, with True == 1
        y_te = [strutils.bool_to_int(label) for label in label_list]  # type: List[int]

        probs = [] # type: List[float]
        if self.estimator:
            probs = self.estimator.predict_proba(candidates)[:, 1]

        self.classifier_status['classifer_type'] = 'spanclassifier'
        self.classifier_status['eval_status'] = evalutils.calc_pred_status_with_prob(probs, y_te)
        self.classifier_status['best_params_'] = self.best_parameters
        self.ant_status['eval_status'] = self.classifier_status['eval_status']

        return self.classifier_status
