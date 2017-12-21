from collections import defaultdict
from datetime import datetime
import json
import logging
import pprint
import time

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


class SpanAnnotator(baseannotator.BaseAnnotator):

    def __init__(self,
                 label,
                 *,
                 doclist_to_antdoc_list,
                 docs_to_samples,
                 pipeline,
                 gridsearch_parameters,
                 threshold=0.5):
        self.label = label
        
        # used for training
        self.doclist_to_antdoc_list = doclist_to_antdoc_list
        self.docs_to_samples = docs_to_samples
        self.pipeline = pipeline
        self.gridsearch_parameters = gridsearch_parameters
        self.threshold = threshold

        self.best_parameters = {}
        self.estimator = None

        # these are set after training        
        self.classifier_status = {'label': label}
        self.annotator_status = {'label': label}

    def train_antdoc_list(self,
                          samples,
                          label_list,
                          group_id_list,
                          pipeline,
                          parameters,
                          work_dir):
        logging.info('train_antdoc_list()...')

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint.pprint(parameters)

        pos_neg_map = defaultdict(int)
        for label in label_list:
            pos_neg_map[label] += 1
        for label, count in pos_neg_map.items():
            print("train_antdoc_list(), pos_neg_map[{}] = {}".format(label, count))

        group_kfold = list(GroupKFold().split(samples,
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
    def test_antdoc_list(self, ebantdoc_list, threshold=None, work_dir='work_dir'):
        logging.debug('test_document_list')
        if not threshold:
            threshold = self.threshold
        # pylint: disable=C0103
        tp, fn, fp, tn = 0, 0, 0, 0

        for ebantdoc in ebantdoc_list:
            #print('ebantdoc.fileid = {}'.format(ebantdoc.file_id))
            # print("ant_list: {}".format(ant_list))
            prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
            # prov_human_ant_list = [hant for hant in ebantdoc.para_prov_ant_list
                                   if hant.label == self.label]

            ant_list = self.annotate_antdoc(ebantdoc,
                                            threshold=threshold,
                                            prov_human_ant_list=prov_human_ant_list,
                                            work_dir=work_dir)
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

    def test_antdoc(self, ebantdoc, threshold=None, dir_work='dir-work'):
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

    # returns samples, label_list, group_id_list
    def documents_to_samples(self, antdoc_list, label: str):
        return self.docs_to_samples.documents_to_samples(antdoc_list, label)


    def annotate_antdoc(self,
                        eb_antdoc,
                        *,
                        threshold=None,
                        prov_human_ant_list=None,
                        work_dir='dir-work'):
        
        # manually set the threshold
        # self.provision_classifier.threshold = 0.5
        if threshold is None:
            threshold = self.threshold

        start_time = time.time()
        samples, prob_list = self.predict_antdoc(eb_antdoc, work_dir)
        end_time = time.time()
        logging.debug("annotate_antdoc(%s, %s) took %.0f msec",
                      self.label, eb_antdoc.file_id, (end_time - start_time) * 1000)

        post_processor = ebpostproc.obtain_postproc('span_default')
        prov_annotations = post_processor.post_process(eb_antdoc.text,
                                                       list(zip(samples, prob_list)),
                                                       threshold,
                                                       label=self.label,
                                                       prov_human_ant_list=prov_human_ant_list)

        """
        fromto_mapper = fromtomapper.FromToMapper('an offset mapper',
                                                  eb_antdoc.nlp_sx_lnpos_list,
                                                  eb_antdoc.origin_sx_lnpos_list)
        # this is an in-place modification
        fromto_mapper.adjust_fromto_offsets(prov_annotations)
        update_text_with_span_list(prov_annotations, eb_antdoc.text)
        """

        return prov_annotations

    def print_eval_status(self, model_dir):
        
        eval_status = {'label': self.label}
        eval_status['classifier_status'] = self.classifier_status['eval_status']
        eval_status['annotator_status'] = self.annotator_status['eval_status']
        pprint.pprint(eval_status)

        model_status_fn = model_dir + '/' +  self.label + ".status"
        strutils.dumps(json.dumps(eval_status), model_status_fn)

        with open('label_model_stat.tsv', 'a') as pmout:
            cls_status = self.classifier_status['eval_status']
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


    # return a list of samples, a list of labels
    def predict_antdoc(self, antdoc, work_dir):
        logging.info('predict_antdoc({})...'.format(antdoc.file_id))

        # label_list, group_id_list are ignored
        samples, _, _ = self.docs_to_samples.documents_to_samples([antdoc])

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
        samples, label_list, group_id = self.docs_to_samples.documents_to_samples(antdoc_list, self.label)

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
