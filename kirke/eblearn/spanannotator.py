from collections import defaultdict
from datetime import datetime
import json
import logging
import pprint
import time

from typing import Any, Dict, List, Optional, Tuple

# pylint: disable=import-error
from sklearn.model_selection import GroupKFold
# pylint: disable=import-error
from sklearn.model_selection import GridSearchCV

from kirke.eblearn import baseannotator, ebpostproc
from kirke.utils import ebantdoc3, evalutils, strutils


PROVISION_EVAL_ANYMATCH_SET = set(['title'])

def adapt_pipeline_params(best_params):
    result = {}
    for param_name, param_val in best_params.items():
        if param_name.startswith('clf__'):
            result[param_name[5:]] = param_val
        else:
            pass  # skip eb_transformer_* and others
    return result


class SpanAnnotator(baseannotator.BaseAnnotator):

    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 label: str,
                 version: str,
                 *,
                 doclist_to_antdoc_list,
                 docs_to_samples,
                 sample_transformers,
                 pipeline,
                 gridsearch_parameters,
                 # prefer recall over precision
                 threshold: float = 0.2,
                 kfold: int = 3) -> None:
        super().__init__(label, 'no description')
        self.provision = label  # to be consistent with ProvAnnotator in ebannotator.py
        self.version = version

        # used for training
        self.doclist_to_antdoc_list = doclist_to_antdoc_list
        self.docs_to_samples = docs_to_samples
        self.sample_transformers = sample_transformers
        self.pipeline = pipeline
        self.gridsearch_parameters = gridsearch_parameters
        self.threshold = threshold
        self.kfold = kfold

        self.best_parameters = {}  # type: Dict[str, Any]
        self.estimator = None

        # these are set after training
        self.classifier_status = {'label': label}  # type: Dict[str, Any]
        self.ant_status = {'label': label}  # type: Dict[str, Any]

    # pylint: disable=too-many-arguments
    def train_antdoc_list(self,
                          samples,
                          label_list,
                          group_id_list,
                          pipeline,
                          parameters,
                          work_dir):
        logging.info('spanannotator.train_antdoc_list()...')

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint.pprint(parameters)

        pos_neg_map = defaultdict(int)
        for label in label_list:
            pos_neg_map[label] += 1
        for label, count in pos_neg_map.items():
            logging.info("train_antdoc_list(), pos_neg_map[%s] = %d", label, count)

        group_kfold = list(GroupKFold(n_splits=self.kfold).split(samples,
                                                                 label_list,
                                                                 groups=group_id_list))
        # grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, scoring='roc_auc',
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, scoring='f1',
                                   verbose=1, cv=group_kfold)

        time_0 = time.time()
        grid_search.fit(samples, label_list)
        print("done in %0.3fs" % (time.time() - time_0))

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        self.best_parameters = adapt_pipeline_params(grid_search.best_estimator_.get_params())
        for param_name in sorted(self.best_parameters.keys()):
            print("\t%s: %r" % (param_name, self.best_parameters[param_name]))
        print()

        self.estimator = grid_search.best_estimator_


    # ProvisionAnnotator does not train, it only predict
    # Training is available only for classifiers
    # def train(self):
    #    pass
    # pylint: disable=R0914
    def test_antdoc_list(self,
                         ebantdoc_list: List[ebantdoc3.EbAnnotatedDoc3],
                         threshold: float,
                         work_dir: str = 'work_dir')  -> Tuple[Dict[str, Any],
                                                               Dict[str, Dict]]:
        logging.debug('spanannotator.test_antdoc_list(), len= %d', len(ebantdoc_list))
        if not threshold:
            threshold = self.threshold
        # pylint: disable=C0103
        tp, fn, fp, tn = 0, 0, 0, 0
        log_json = dict()

        for ebantdoc in ebantdoc_list:
            prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
                                   if hant.label == self.label]

            ant_list, threshold = self.annotate_antdoc(ebantdoc,
                                                       threshold=threshold,
                                                       prov_human_ant_list=prov_human_ant_list,
                                                       work_dir=work_dir)
            # print("\nfn: {}".format(ebantdoc.file_id))
            # tp, fn, fp, tn = self.calc_doc_confusion_matrix(prov_ant_list,
            # pred_prob_start_end_list, txt)
            if self.label in PROVISION_EVAL_ANYMATCH_SET:
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
                                                            is_raw_mode=True,
                                                            diagnose_mode=True)
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

    # nobody is calling this
    """
    def test_antdoc(self,
                    ebantdoc: ebantdoc3.EbAnnotatedDoc3,
                    threshold: float = None,
                    dir_work: str = 'dir-work'):
        logging.debug('test_document')

        ant_list = self.annotate_antdoc(ebantdoc, threshold=threshold, dir_work=dir_work)
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
    """

    # returns samples, label_list, group_id_list
    # this also enriches samples using additional self.sample_transformers
    def documents_to_samples(self,
                             antdoc_list: List[ebantdoc3.EbAnnotatedDoc3],
                             label: Optional[str] = None) -> Tuple[List[Dict],
                                                                   List[bool],
                                                                   List[int]]:
        samples, label_list, group_id_list = \
            self.docs_to_samples.documents_to_samples(antdoc_list, label)
        for sample_transformer in self.sample_transformers:
            samples = sample_transformer.enrich(samples)
        return samples, label_list, group_id_list

    def annotate_antdoc(self,
                        eb_antdoc: ebantdoc3.EbAnnotatedDoc3,
                        *,
                        threshold: Optional[float] = None,
                        prov_human_ant_list: Optional[List] = None,
                        work_dir: str = 'dir-work') -> Tuple[List[Dict], float]:

        # manually set the threshold
        # self.provision_classifier.threshold = 0.5
        if threshold is None:
            threshold = self.threshold

        start_time = time.time()
        samples, prob_list = self.predict_antdoc(eb_antdoc, work_dir)
        end_time = time.time()
        logging.debug("annotate_antdoc(%s, %s) took %.0f msec",
                      self.label, eb_antdoc.file_id, (end_time - start_time) * 1000)

        # print('threshold = {}'.format(threshold))
        # for ci, (ss, pp) in enumerate(zip(samples, prob_list)):
        #    if pp >= threshold:
        #        print('#{}, prov= {}, prob = {}, sample = {}'.format(ci, self.provision, pp, ss))

        post_processor = ebpostproc.obtain_postproc('span_default')
        # change to x_threshold to pass "mypy" type checking
        prov_annotations, x_threshold = post_processor.post_process(eb_antdoc.text,
                                                                    list(zip(samples, prob_list)),
                                                                    threshold,
                                                                    provision=self.label,
                                                                    # pylint: disable=line-too-long
                                                                    prov_human_ant_list=prov_human_ant_list)
        return prov_annotations, x_threshold

    def print_eval_status(self, model_dir):

        eval_status = {'label': self.label}
        eval_status['pred_status'] = self.classifier_status['eval_status']
        eval_status['ant_status'] = self.ant_status['eval_status']
        pprint.pprint(eval_status)

        model_status_fn = model_dir + '/' +  self.label + ".status"
        strutils.dumps(json.dumps(eval_status), model_status_fn)

        with open('label_model_stat.tsv', 'a') as pmout:
            cls_status = self.classifier_status['eval_status']
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


    # return a list of samples, a list of labels
    def predict_antdoc(self,
                       antdoc: ebantdoc3.EbAnnotatedDoc3,
                       work_dir: str) -> Tuple[List[Dict[str, Any]], List[float]]:
        logging.info('prov = %s, predict_antdoc(%s)', self.provision, antdoc.file_id)

        # label_list, group_id_list are ignored
        samples, unused_label_list, unused_group_id_list = \
            self.documents_to_samples([antdoc])

        if not samples:
            return [], []

        # to indicate which type of annotation this is
        for sample in samples:
            sample['label'] = self.label

        """
        doc_text = antdoc.text
        span_st_list = [doc_text[sample['start']:sample['end']]
                        for sample in samples]
        overrides = ebpostproc.gen_provision_overrides(self.provision,
                                                       sent_st_list)
        """

        probs = [] # type: List[float]
        if self.estimator:
            probs = self.estimator.predict_proba(samples)[:, 1]

        # do the override
        """
        for i, override in enumerate(overrides):
            if override != 0.0:
                probs[i] += override
                probs[i] = min(probs[i], 1.0)
        """

        return samples, probs

    def predict_and_evaluate(self, antdoc_list, work_dir, is_debug=False):
        logging.info('predict_and_evaluate()...')

        # label_list, group_id_list are ignored
        samples, label_list, unused_group_id_list = self.documents_to_samples(antdoc_list,
                                                                              self.label)

        pos_neg_map = defaultdict(int)
        for label in label_list:
            pos_neg_map[label] += 1
        for label, count in pos_neg_map.items():
            print("predict_and_evaluate(), pos_neg_map[{}] = {}".format(label, count))

        # print("attrvec_list.size = ", len(attrvec_list))
        # print("label_list.size = ", len(label_list))
        # print("full_txt_fn_list.size = ", len(full_txt_fn_list))

        y_te = label_list
        """
        # num_positive = np.count_nonzero(y_te)
        # logging.debug('num true positives in testing = {}'.format(num_positive))
        sent_st_list = [attrvec.bag_of_words for attrvec in attrvec_list]
        overrides = ebpostproc.gen_provision_overrides(self.provision, sent_st_list)
        """

        # pylint: disable=fixme
        # TODO, jshaw
        # can remove sgd_preds and use 0.5 as the filter
        # sgd_preds = self.eb_grid_search.predict(attrvec_list)
        probs = [] # type: List[float]
        if self.estimator:
            probs = self.estimator.predict_proba(samples)[:, 1]

        self.classifier_status['classifer_type'] = 'spanclassifier'
        self.classifier_status['eval_status'] = evalutils.calc_pred_status_with_prob(probs, y_te)
        # self.classifier['pos_threshold_status'] = (
        #     evalutils.calc_pos_threshold_prob_status(probs, y_te, self.pos_threshold))
        # self.classifier['threshold_status'] = (
        #     evalutils.calc_threshold_prob_status(probs, y_te, self.threshold))
        # self.classifier['override_status'] = (
        #    evalutils.calc_prob_override_status(probs, y_te, self.threshold, overrides))
        self.classifier_status['best_params_'] = self.best_parameters

        return self.classifier_status
