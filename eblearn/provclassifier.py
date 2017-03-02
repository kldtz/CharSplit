#!/usr/bin/env python

import logging
from pprint import pprint
from time import time

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from eblearn import ebpostproc, evalutils
from eblearn.ebtransformer import EbTransformer
from eblearn.ebclassifier import EbClassifier

# based on http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html#sphx-glr-auto-examples-hetero-feature-union-py


global_threshold = 0.12

# The value in this provision_threshold_map is manually 
# set by inspecting the result.  Using 0.06 in general
# produces too many false positives.
provision_threshold_map = {'change_control': 0.24,
                           'confidentiality': 0.24,
                           'equitable_relief': 0.24,
                           'events_default': 0.18,
                           'sublicense': 0.24,
                           'survival': 0.24,
                           'termination': 0.36}
                           

class ProvisionClassifier(EbClassifier):

    def __init__(self, provision):
        EbClassifier.__init__(self, provision)
        self.eb_grid_search = None

        self.pos_threshold = 0.5   # default threshold for sklearn classifier
        self.threshold = provision_threshold_map.get(provision, global_threshold)        

    def train_antdoc_list(self, ebantdoc_list, work_dir, model_file_name):
        logging.info('train_antdoc_list()...')
        
        attrvec_list = []
        ebsent_list = []
        group_id_list = []
        for group_id, eb_antdoc in enumerate(ebantdoc_list):
            attrvec_list.extend(eb_antdoc.get_attrvec_list())
            tmp_ebsent_list = eb_antdoc.get_ebsent_list()
            ebsent_list.extend(tmp_ebsent_list)
            group_id_list.extend([group_id] * len(tmp_ebsent_list))
            
        label_list = [self.provision in ebsent.labels for ebsent in ebsent_list]
        attrvec_ebsent_list = list(zip(attrvec_list, ebsent_list))
        # print("attrvec_ebsent_list.size = ", len(attrvec_ebsent_list))

        # TODO, jshaw, explore this more in future.
        # iterations = 50  (for 10 iteration, f1=0.91; for 50 iterations, f1=0.90,
        #                   for 5 iterations, f1=0,89),  So 10 iterations wins for now.
        # This shows that the "iteration" parameter probably needs tuning.
        iterations = 10
        pipeline = Pipeline([
            ('eb_transformer', EbTransformer(provision=self.provision)),
            ('clf', SGDClassifier(loss='log', penalty='l2', n_iter=iterations, shuffle=True, random_state=42, class_weight={True: 10, False: 1})),
        ])            

        # TODO, jshaw, uncomment in real code
        # parameters = {'clf__alpha': 10.0 ** -np.arange(1, 5)}
        parameters = {'clf__alpha': 10.0 ** -np.arange(3, 4)}
        #    parameters = {'C': [.01,.1,1,10,100]}
        #    sgd_clf = LogisticRegression()

        group_kfold = list(GroupKFold().split(attrvec_ebsent_list, label_list, groups=group_id_list))
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, scoring='roc_auc', verbose=1, cv=group_kfold)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        pprint(parameters)
        t0 = time()
        grid_search.fit(attrvec_ebsent_list, label_list)
        print("done in %0.3fs" % (time() - t0))

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        print()

        self.eb_grid_search = grid_search
        self.save(model_file_name)

        return grid_search


    def predict_antdoc(self, eb_antdoc, work_dir):
        # logging.info('predict_antdoc()...')

        attrvec_list = eb_antdoc.get_attrvec_list()
        ebsent_list = eb_antdoc.get_ebsent_list()
        # print("attrvec_list.size = ", len(attrvec_list))
        # print("ebsent_list.size = ", len(ebsent_list))        
        
        sent_st_list = [ebsent.get_tokens_text() for ebsent in ebsent_list]        
        overrides = ebpostproc.gen_provision_overrides(self.provision, sent_st_list)
        attrvec_ebsent_list = list(zip(attrvec_list, ebsent_list))
        probs = self.eb_grid_search.predict_proba(attrvec_ebsent_list)[:, 1]

        return probs

    
# override should not be in the normal sklean pipeline
# first during search for parameters, we do have access to
# which sentences are training or testing
# overrides = self.gen_provision_overrides(sent_st_list)
# print('size of overrides = {}'.format(len(overrides)))
# self.print_prob_override_status(probs, y_te, overrides)
# self.print_with_threshold(probs, y_te, overrides)
        

    # this is mainly used for the outer testing (real hold out)
    def predict_and_evaluate(self, ebantdoc_list, work_dir, diagnose_mode=False):
        logging.info('predict_and_evaluate()...')        

        attrvec_list = []
        ebsent_list = []
        full_txt_fn_list = []
        for eb_antdoc in ebantdoc_list:
            tmp_attrvec_list = eb_antdoc.get_attrvec_list()
            num_sent = len(tmp_attrvec_list)
            txt_fn = eb_antdoc.get_file_id()
            
            attrvec_list.extend(tmp_attrvec_list)
            ebsent_list.extend(eb_antdoc.get_ebsent_list())
            # for diagnosis purpose
            full_txt_fn_list.extend([txt_fn] * num_sent)
        label_list = [self.provision in ebsent.labels for ebsent in ebsent_list]
        attrvec_ebsent_list = list(zip(attrvec_list, ebsent_list))
            
        # print("attrvec_list.size = ", len(attrvec_list))
        # print("ebsent_list.size = ", len(ebsent_list))        
        # print("label_list.size = ", len(label_list))
        # print("full_txt_fn_list.size = ", len(full_txt_fn_list))
        
        y_te = label_list
        num_positive = np.count_nonzero(y_te)
        # logging.debug('num true positives in testing = {}'.format(num_positive))
        sent_st_list = [ebsent.get_tokens_text() for ebsent in ebsent_list]        
        overrides = ebpostproc.gen_provision_overrides(self.provision, sent_st_list)

        # TODO, jshaw
        # can remove sgd_preds and use 0.5 as the filter                     
        # sgd_preds = self.eb_grid_search.predict(attrvec_ebsent_list)
        probs = self.eb_grid_search.predict_proba(attrvec_ebsent_list)[:, 1]

        #print("Grid scores on development set:")
        #print()
        #means = gs_clf.cv_results_['mean_test_score']
        #stds = gs_clf.cv_results_['std_test_score']
        #for mean, std, params in zip(means, stds, gs_clf.cv_results_['params']):
        #    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        # y_actual, y_pred = y_te, sgd_preds
        # print("classification report2")
        # print(classification_report(y_actual, y_pred))
        # print()

        # now error analysis
        if diagnose_mode:
            quintet_list = zip(full_txt_fn_list, sent_st_list, probs, overrides, y_te)
            num_fp, num_fn, num_adj_fp, num_adj_fn = 0, 0, 0, 0 
            for txtfn, sent, prob, override, actual in quintet_list:
                pred = prob >= 0.5
                if pred and not actual:
                    num_fp += 1
                    print("\n=====FP, txt_fn = {}".format(txtfn))
                    print("\nprob = {}, sgd_preds = {}, threshold = {}".format(prob, pred, self.threshold)) 
                    print("sent= [{}]".format(sent))
                elif not pred and actual:
                    num_fn += 1
                    print("\n=====FN, txt_fn = {}".format(txtfn))
                    print("\nprob = {}, sgd_preds = {}, threshold = {}".format(prob, pred, self.threshold)) 
                    print("sent= [{}]".format(sent))                
                adjusted_pred = prob >= self.threshold or override
                if adjusted_pred and not actual:
                    num_adj_fp += 1
                    print("\n=====Adjusted FP, txt_fn = {}".format(txtfn))
                    print("\nprob= {}, sgd_preds= {}, threshold= {}, override= {}".format(prob, pred, self.threshold, override)) 
                    print("sent= [{}]".format(sent))                                
                elif not adjusted_pred and actual:
                    num_adj_fn += 1
                    print("\n=====Adjusted FN, txt_fn = {}".format(txtfn))
                    print("\nprob= {}, sgd_preds= {}, threshold= {}, override= {}".format(prob, pred, self.threshold, override)) 
                    print("sent= [{}]".format(sent))                                                
            print("num_fp = {}".format(num_fp))
            print("num_fn = {}".format(num_fn))
            print("num_adj_fp = {}".format(num_adj_fp))
            print("num_adj_fn = {}".format(num_adj_fn))

        # print("probs: {}".format(sorted(probs, reverse=True)))
        evalutils.print_with_threshold(probs, y_te, overrides)

        self.pred_status['classifer_type'] = 'provclassifier'
        self.pred_status['pred_status'] =  evalutils.calc_pred_status(probs, y_te)
        self.pred_status['pos_threshold_status'] =  evalutils.calc_pos_threshold_prob_status(probs, y_te,
                                                                                             self.pos_threshold)
        self.pred_status['threshold_status'] = evalutils.calc_threshold_prob_status(probs, y_te, self.threshold)
        self.pred_status['override_status'] = evalutils.calc_prob_override_status(probs, y_te,
                                                                                  self.threshold, overrides)
        self.pred_status['best_params_'] = self.eb_grid_search.best_params_

        return self.pred_status

