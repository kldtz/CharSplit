from collections import defaultdict, namedtuple
import re
from kirke.utils import mathutils

# pylint: disable=C0103

# label_start_end_list is of type prov_annotation_list
def find_annotation_overlap(start, end, label_start_end_list):
    result_list = []
    if not label_start_end_list:
      return result_list
    for ant in label_start_end_list:
        if mathutils.start_end_overlap((start, end), (ant.start, ant.end)):
            result_list.append(ant)
    return result_list


def calc_precision_recall_f1(tn, fp, fn, tp, title):
    print("\n" + title)
    actual_true = fn + tp
    actual_false = tn + fp
    pred_true = tp + fp
    pred_false = tn + fn

    print("actual_true= {}, actual_false= {}".format(actual_true, actual_false))
    print("  pred_true= {},   pred_false= {}".format(pred_true, pred_false))
    print("[[tn={}, fp={}], [fn={}, tp={}]]".format(tn, fp, fn, tp))

    if tp + fp == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if prec + recall == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    print("prec= {}, recall= {}, f1= {}".format(prec, recall, f1))

    return prec, recall, f1


AnnotationWithProb = namedtuple('AnnotationWithProb', ['label', 'start', 'end', 'prob'])


# pylint: disable=R0914
def calc_doc_ant_confusion_matrix(prov_human_ant_list, ant_list, ebantdoc, threshold, diagnose_mode=False):
    txt = ebantdoc.get_text()
    tp, fp, tn, fn = 0, 0, 0, 0

    pred_ant_list = []
    for adict in ant_list:
        pred_ant_list.append(AnnotationWithProb(adict['label'],
                                                adict['start'],
                                                adict['end'],
                                                adict['prob']))
    linebreaks = re.compile("[\n\r]")
    tp_inst_map = defaultdict(list)
    fp_inst_list = []
    fn_inst_map = defaultdict(list)
    tp_fn_set = set([])

    for hant in prov_human_ant_list:
        pred_overlap_list = find_annotation_overlap(hant.start, hant.end, pred_ant_list)
        if len(pred_overlap_list) > 0:
            prob = max([x.prob for x in pred_overlap_list])
            if prob >= threshold:
                tp_inst_map[(hant.start, hant.end, hant.label)] = pred_overlap_list
                tp += 1
            else:
                fn_inst_map[(hant.start, hant.end, hant.label)] = pred_overlap_list
                fn += 1
        tp_fn_set |= set(pred_overlap_list)
    
    for pant in pred_ant_list:
        if pant in tp_fn_set:
            continue
        if pant.prob > threshold:
          fp_inst_list.append(pant)
          fp += 1 
    

    # there is no tn, because we deal with only annotations
    if diagnose_mode:
        for i, hant in enumerate(sorted(tp_inst_map.keys())):
            hstart, hend, _ = hant
            tp_inst_list = tp_inst_map[hant]
            tp_txt = " ".join([txt[x.start:x.end] for x in tp_inst_list])
            prob = max([x.prob for x in tp_inst_list])
            print("tp\t{}\t{}\t{}".format(ebantdoc.file_id, linebreaks.sub(" ", tp_txt), str(prob)))

        for i, hant in enumerate(sorted(fn_inst_map.keys())):
            hstart, hend, _ = hant
            fn_inst_list = fn_inst_map[hant]
            fn_txt = " ".join([txt[x.start:x.end] for x in fn_inst_list])
            prob = max([x.prob for x in fn_inst_list])
            print("fn\t{}\t{}\t{}".format(ebantdoc.file_id, linebreaks.sub(" ", fn_txt), str(prob)))

        for i, pred_ant in enumerate(fp_inst_list):
            print("fp\t{}\t{}\t{}".format(ebantdoc.file_id, linebreaks.sub(" ", txt[pred_ant.start:pred_ant.end]), str(pred_ant.prob)))

    return tp, fn, fp, tn

# for 'title', we want to match any title annotation
# if any matched, we passed.  Don't care about any other.
# pylint: disable=R0914
def calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list, ant_list, txt, diagnose_mode=False):
    tp, fp, tn, fn = 0, 0, 0, 0
    # print("calc_doc_ant_confusion_matrix:")

    pred_ant_list = []
    for adict in ant_list:
        pred_ant_list.append(AnnotationWithProb(adict['label'],
                                                adict['start'],
                                                adict['end'],
                                                adict['prob']))
    # print("prov_human_ant_list: {}".format(prov_human_ant_list))
    # print("pred_ant_list: {}".format(pred_ant_list))

    tp_inst_map = defaultdict(list)
    fp_inst_list = []
    fn_inst_list = []
    tp_fn_set = set([])
    for hant in prov_human_ant_list:

        pred_overlap_list = find_annotation_overlap(hant.start, hant.end, pred_ant_list)
        if pred_overlap_list:
            # This handles the case there a predicted annotation overlap with one
            # or more human annotations.
            tp_inst_map[(hant.start, hant.end, hant.label)] = pred_overlap_list
            tp += 1
        else:
            fn_inst_list.append(hant)
            fn += 1

        tp_fn_set |= set(pred_overlap_list)

    for pant in pred_ant_list:
        # skip if in TP or FP before
        if pant in tp_fn_set:
            continue
        # if pred:
        fp_inst_list.append(pant)
        fp += 1

    # we only care about any match
    if tp:
        fn = 0
        fp = 0
        return tp, fn, fp, tn

    if fp:
        fp = 1

    if fn:
        fn = 1

    # print("tp= {}, fn= {}, fp = {}, tn = {}".format(tp, fn, fp, tn))

    # there is no tn, because we deal with only annotations
    if diagnose_mode:
        print("tp = {}".format(tp))
        for i, hant in enumerate(sorted(tp_inst_map.keys())):
            hstart, hend, _ = hant
            print("\ntp #{}, start= {}, end= {}".format(i+1, hstart, hend))
            print(txt[hstart:hend])
            tp_inst_list = tp_inst_map[hant]
            for j, pred_ant in enumerate(tp_inst_list):
                print("     inst #%d, start2= %d, end2= %d, prob= %.6f" %
                      (j+1, pred_ant.start, pred_ant.end, pred_ant.prob))
                print("     ", end='')
                print("[[" + txt[pred_ant.start:pred_ant.end] + "]]")

        print("\n\nfn = {}".format(fn))
        for i, hant in enumerate(fn_inst_list):
            print("\nfn #%d, start= %d, end= %d, label = %s" %
                  (i+1, hant.start, hant.end, hant.label))
            print("     ", end='')
            print("[[" + txt[hant.start:hant.end] + "]]")

        print("\n\nfp = {}".format(fp))
        for i, pred_ant in enumerate(fp_inst_list):
            print("\nfp #%d, start= %d, end= %d, prob= %.6f" %
                  (i, pred_ant.start, pred_ant.end, pred_ant.prob))
            print("[[" + txt[pred_ant.start:pred_ant.end] + "]]")

    return tp, fn, fp, tn



# pylint: disable=R0914
def calc_doc_ant_confusion_matrix_precx2(prov_human_ant_list, ant_list, txt, diagnose_mode=False):
    tp, fp, tn, fn = 0, 0, 0, 0
    # print("calc_doc_ant_confusion_matrix:")

    pred_ant_list = []
    for adict in ant_list:
        pred_ant_list.append(AnnotationWithProb(adict.label, adict.start,
                                                adict.end, adict.prob))
    # print("prov_human_ant_list: {}".format(prov_human_ant_list))
    # print("pred_ant_list: {}".format(pred_ant_list))

    tp_inst_map = defaultdict(list)
    fp_inst_list = []
    fn_inst_list = []
    tp_fn_set = set([])
    for hant in prov_human_ant_list:

        pred_overlap_list = find_annotation_overlap(hant.start, hant.end, pred_ant_list)
        if pred_overlap_list:
            # This handles the case there a predicted annotation overlap with one
            # or more human annotations.
            tp_inst_map[(hant.start, hant.end, hant.label)] = pred_overlap_list
            tp += len(pred_overlap_list)   # here is the difference from before
        else:
            fn_inst_list.append(hant)
            fn += 1

        tp_fn_set |= set(pred_overlap_list)

    for pant in pred_ant_list:
        # skip if in TP or FP before
        if pant in tp_fn_set:
            continue
        # if pred:
        fp_inst_list.append(pant)
        fp += 1

    # print("tp= {}, fn= {}, fp = {}, tn = {}".format(tp, fn, fp, tn))

    # there is no tn, because we deal with only annotations
    if diagnose_mode:
        print("tp = {}".format(tp))
        for i, hant in enumerate(sorted(tp_inst_map.keys())):
            hstart, hend, _ = hant
            print("\ntp #{}, start= {}, end= {}".format(i+1, hstart, hend))
            print(txt[hstart:hend])
            tp_inst_list = tp_inst_map[hant]
            for j, pred_ant in enumerate(tp_inst_list):
                print("     inst #%d, start2= %d, end2= %d, prob= %.6f",
                      j+1, pred_ant.start, pred_ant.end, pred_ant.prob)
                print("     ", end='')
                print(txt[pred_ant.start:pred_ant.end])

        print("\n\nfn = {}".format(fn))
        for i, hant in enumerate(fn_inst_list):
            print("\nfn #%d, start= %d, end= %d, label = %s",
                  i+1, hant.start, hant.end, hant.label)
            print("     ", end='')
            print(txt[hant.start:hant.end])

        print("\n\nfp = {}".format(fp))
        for i, pred_ant in enumerate(fp_inst_list):
            print("fp #%d, start= %d, end= %d, prob= %.6f",
                  i, pred_ant.start, pred_ant.end, pred_ant.prob)
            print(txt[pred_ant.start:pred_ant.end])

    return tp, fn, fp, tn


#
# utilities
#


# pylint: disable=W0105
"""
    # compute thresholded recall/precision
    THRESHOLD = .06
    rec_den = 0
    prec_den = 0
    tp = 0
    o_prec_den = 0
    o_tp = 0

    for i, pred in enumerate(list(probs)):
        if y_te[i] == 1:
            rec_den += 1    # actual_true
        if pred >= THRESHOLD:
            prec_den += 1   # predict true
            if y_te[i] == 1:
                tp += 1
        if (pred >= THRESHOLD and not overrides[i]) or overrides[i]:
            o_prec_den += 1  # override true
            if y_te[i] == 1:
                o_tp += 1    # override true positive

    print("THRESHOLD= {}".format(THRESHOLD))
    print("rec_den: ", rec_den)
    print("pred_den: ", prec_den)
    print("tp: ", tp)
    print("o_prec_den: ", o_prec_den)
    print("o_tp: ", o_tp)

    if (rec_den > 0):
        adj_rec = tp / rec_den
        o_rec = o_tp / rec_den
    else:
        adj_rec = 0
        o_rec = 0

    if (prec_den > 0):
        adj_prec = tp / prec_den
    else:
        adj_prec = 0

    if (o_prec_den > 0):
        o_prec = o_tp / o_prec_den
    else:
        o_prec = 0

    print("TH:", THRESHOLD)
    print("adjusted recall: ", adj_rec)
    print("adjusted prec: ", adj_prec)
    print("overridden recall: ", o_rec)
    print("overridden prec: ", o_prec)
    """


def calc_pred_status_with_prob(probs, y_te):
    tn, fp, fn, tp = 0, 0, 0, 0

    for i, prob in enumerate(list(probs)):
        pred = prob >= 0.5
        if y_te[i] == 1:
            if pred:
                tp += 1
            else:
                fn += 1
        else:
            if pred:
                fp += 1
            else:
                tn += 1
    title = 'pred status'
    prec, recall, f1 = calc_precision_recall_f1(tn, fp, fn, tp, title)
    status = {'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
              'pred_threshold': 0.5,
              'prec': prec, 'recall': recall, 'f1': f1}
    return status


def calc_pred_status(preds, y_te):
    tn, fp, fn, tp = 0, 0, 0, 0

    for i, pred in enumerate(list(preds)):
        if y_te[i] == 1:
            if pred:
                tp += 1
            else:
                fn += 1
        else:
            if pred:
                fp += 1
            else:
                tn += 1
    title = 'pred status'
    prec, recall, f1 = calc_precision_recall_f1(tn, fp, fn, tp, title)
    status = {'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
              'pred_threshold': 0.5,
              'prec': prec, 'recall': recall, 'f1': f1}
    return status


def calc_pos_threshold_prob_status(probs, y_te, pos_threshold):
    tn, fp, fn, tp = 0, 0, 0, 0

    for i, prob in enumerate(list(probs)):
        # this is strictly greater, not >=
        pred = prob > pos_threshold
        if y_te[i] == 1:
            if pred:
                tp += 1
            else:
                fn += 1
        else:
            if pred:
                fp += 1
            else:
                tn += 1

    title = "pos threshold status, pos threshold = {}".format(pos_threshold)
    prec, recall, f1 = calc_precision_recall_f1(tn, fp, fn, tp, title)
    status = {'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
              'pos_threshold': pos_threshold,
              'prec': prec, 'recall': recall, 'f1': f1}
    return status


def calc_threshold_prob_status(probs, y_te, threshold):
    tn, fp, fn, tp = 0, 0, 0, 0

    for i, prob in enumerate(list(probs)):
        pred = prob >= threshold
        if y_te[i] == 1:
            if pred:
                tp += 1
            else:
                fn += 1
        else:
            if pred:
                fp += 1
            else:
                tn += 1

    title = "threshold status, threshold = {}".format(threshold)
    prec, recall, f1 = calc_precision_recall_f1(tn, fp, fn, tp, title)
    status = {'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
              'threshold': threshold,
              'prec': prec, 'recall': recall, 'f1': f1}
    return status


def calc_prob_override_status(probs, y_te, threshold, overrides):
    tn, fp, fn, tp = 0, 0, 0, 0

    for i, prob in enumerate(list(probs)):
        pred = (prob + overrides[i]) >= threshold
        if y_te[i] == 1:
            if pred:
                tp += 1
            else:
                fn += 1
        else:
            if pred:
                fp += 1
            else:
                tn += 1

    title = "threshold + override, threshold = {}".format(threshold)
    prec, recall, f1 = calc_precision_recall_f1(tn, fp, fn, tp, title)
    status = {'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
              'threshold': threshold,
              'prec': prec, 'recall': recall, 'f1': f1}
    return status


def calc_pred_override_status(preds, y_te, overrides):
    tn, fp, fn, tp = 0, 0, 0, 0

    for i, our_pred in enumerate(list(preds)):
        override = overrides[i]
        pred = our_pred or override
        if y_te[i] == 1:
            if pred:
                tp += 1
            else:
                fn += 1
        else:
            if pred:
                fp += 1
            else:
                tn += 1

    title = "pred + override"
    prec, recall, f1 = calc_precision_recall_f1(tn, fp, fn, tp, title)
    status = {'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
              'pred_threshold': 0.5,
              'prec': prec, 'recall': recall, 'f1': f1}
    return status
