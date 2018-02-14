from collections import defaultdict, namedtuple
import re
# pylint: disable=unused-import
from typing import DefaultDict, Dict, List, Set, Tuple

from kirke.utils import mathutils, wordutils
from kirke.utils.ebsentutils import ProvisionAnnotation


AnnotationWithProb = namedtuple('AnnotationWithProb', ['label', 'start', 'end', 'prob'])

# pylint: disable=C0103
# label_start_end_list is of type prov_annotation_list
def find_annotation_overlap(start: int,
                            end: int,
                            label_start_end_list: List[AnnotationWithProb]) \
                            -> List[AnnotationWithProb]:
    """Find annotations that overlaps with 'start' and 'end'.

    The annotation list is expected to have minimally 'start' and 'end' attributes.

    Args:
        obvious

    Returns:
        The list of annotations that overlaps with 'start' and 'end'.
    """

    result_list = []  # type: List[AnnotationWithProb]
    if not label_start_end_list:
        return result_list
    for ant in label_start_end_list:
        if mathutils.start_end_overlap((start, end), (ant.start, ant.end)):
            result_list.append(ant)
    return result_list


def calc_precision_recall_f1(tn: int,
                             fp: int,
                             fn: int,
                             tp: int,
                             title: str = None) -> Tuple[float, float, float]:
    if title:
        print("\n" + title)
    actual_true = fn + tp
    actual_false = tn + fp
    pred_true = tp + fp
    pred_false = tn + fn

    if tp + fp == 0:
        prec = 0.0
    else:
        prec = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    if prec + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * recall / (prec + recall)

    if title:
        print("actual_true= {}, actual_false= {}".format(actual_true, actual_false))
        print("  pred_true= {},   pred_false= {}".format(pred_true, pred_false))
        print("[[tn={}, fp={}], [fn={}, tp={}]]".format(tn, fp, fn, tp))
        print("prec= {}, recall= {}, f1= {}".format(prec, recall, f1))

    return prec, recall, f1


# pylint: disable=R0914
def calc_doc_ant_confusion_matrix(prov_human_ant_list: List[ProvisionAnnotation],
                                  ant_list: List[Dict],
                                  ebantdoc,
                                  threshold: float,
                                  diagnose_mode: bool = False) -> Tuple[float, float, float, float]:
    """Calculate the confusion matrix for one document only, based only on offsets.

    Args:
        prov_human_ant_list: human annotation list, only for 1 provision
        ant_list: annotation predicted by machine
        ebantdoc: document; kirke.utils.ebantdoc2 right now.
        threshold: the threshold to decide if a provision is positive
        diagnose_mode: whether to print debug info

    Returns:
        tp, fn, fp, tn
    """

    txt = ebantdoc.get_text()
    tp, fp, tn, fn = 0, 0, 0, 0

    pred_ant_list = []
    for adict in ant_list:
        pred_ant_list.append(AnnotationWithProb(adict['label'],
                                                adict['start'],
                                                adict['end'],
                                                adict['prob']))
    linebreaks = re.compile("[\n\r]")
    # pylint: disable=line-too-long
    tp_inst_map = defaultdict(list)  # type: DefaultDict[Tuple[int, int, str], List[AnnotationWithProb]]
    fp_inst_list = []
    # pylint: disable=line-too-long
    fn_inst_map = defaultdict(list)  # type: DefaultDict[Tuple[int, int, str], List[AnnotationWithProb]]
    tp_fn_set = set([])  # type: Set[AnnotationWithProb]

    for hant in prov_human_ant_list:
        pred_overlap_list = find_annotation_overlap(hant.start, hant.end, pred_ant_list)
        if pred_overlap_list:
            prob = max([x.prob for x in pred_overlap_list])
            if prob >= threshold:
                tp_inst_map[(hant.start, hant.end, hant.label)] = pred_overlap_list
                tp += 1
            else:
                fn_inst_map[(hant.start, hant.end, hant.label)] = pred_overlap_list
                fn += 1
        else:  # in case people are using this without providing FN beforehand
            antprob = AnnotationWithProb(hant.label, hant.start, hant.end, -1.0)
            fn_inst_map[(hant.start, hant.end, hant.label)].append(antprob)
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
        for hant_key in sorted(tp_inst_map.keys()):
            tp_inst_list = tp_inst_map[hant_key]
            tp_txt = " ".join([txt[x.start:x.end] for x in tp_inst_list])
            prob = max([x.prob for x in tp_inst_list])
            print("tp\t{}\t{}\t{}".format(ebantdoc.file_id, linebreaks.sub(" ", tp_txt), str(prob)))

        for hant_key in sorted(fn_inst_map.keys()):
            fn_inst_list = fn_inst_map[hant_key]
            fn_txt = " ".join([txt[x.start:x.end] for x in fn_inst_list])
            prob = -1.0
            if fn_inst_list:
                prob = max([x.prob for x in fn_inst_list])
            print("fn\t{}\t{}\t{}".format(ebantdoc.file_id, linebreaks.sub(" ", fn_txt), str(prob)))

        for pred_ant in fp_inst_list:
            print("fp\t{}\t{}\t{}".format(ebantdoc.file_id,
                                          linebreaks.sub(" ",
                                                         txt[pred_ant.start:pred_ant.end]),
                                          str(pred_ant.prob)))

    return tp, fn, fp, tn

# for 'title', we want to match any title annotation
# if any matched, we passed.  Don't care about any other.
# pylint: disable=too-many-branches
def calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list: List[ProvisionAnnotation],
                                           ant_list: List[Dict],  # this is machine annotation
                                           ebantdoc,
                                           # threshold,
                                           diagnose_mode: bool = False) \
                                           -> Tuple[float, float, float, float]:
    """Calculate the confusion matrix for one document, if there is any match by offset or string.

    This differs from calc_doc_ant_confusion_matrix() because if there is any annotation match,
    either by offsets or surface form, TP will be max 1.  FP and FN can also maximally be 1.

    There is no threshold in the Args because there is no prob.

    Args:
        prov_human_ant_list: human annotation list, only for 1 provision
        ant_list: annotation predicted by machine
        ebantdoc: document
        diagnose_mode: whether to print debug info

    Returns:
        tp, fn, fp, tn
    """
    txt = ebantdoc.get_text()
    tp, fp, tn, fn = 0, 0, 0, 0
    # print("calc_doc_ant_confusion_matrix:")

    pred_ant_list = []  # this is machine annotations
    # For doing matching with all human annotation by surface form,
    # not by offsets.
    pred_ant_st_list = []  # type: List[str]
    for adict in ant_list:
        pred_ant_list.append(AnnotationWithProb(adict['label'],
                                                adict['start'],
                                                adict['end'],
                                                adict['prob']))
        pred_ant_st_list.append(txt[adict['start']:adict['end']])
    # print("prov_human_ant_list: {}".format(prov_human_ant_list))
    # print("pred_ant_list: {}".format(pred_ant_list))

    # pylint: disable=line-too-long
    tp_inst_map = defaultdict(list)  # type: DefaultDict[Tuple[int, int, str], List[AnnotationWithProb]]
    fp_inst_list = []  # type: List[AnnotationWithProb]
    fn_inst_list = []  # type: List[ProvisionAnnotation]
    tp_fn_set = set([])  # type: Set[AnnotationWithProb]

    # this is to using string to test overlap, in case human annotation is not exhaustic for a doc
    human_ant_st_list = []  # type: List[str]
    for hant in prov_human_ant_list:
        human_ant_st_list.append(txt[hant.start:hant.end])
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

    for pant in pred_ant_list:  # looping through machine annotations
        # skip if in TP or FN before
        if pant in tp_fn_set:
            continue
        # if reach here, the pant must be false positive
        fp_inst_list.append(pant)
        fp += 1

    # If no tp, maybe human annotation is not exhaustic.
    # Try to match any machine annotation with any human annotation with >= 0.66 word overlap
    if tp == 0:
        for pred_ant_st in pred_ant_st_list:
            for human_ant_st in human_ant_st_list:
                if wordutils.is_word_overlap_ge_66p(pred_ant_st, human_ant_st):
                    tp = 1
                    break
            if tp > 0:
                break

    # we only care about any match
    # This also assume that all the titles in the document are human annotated.
    # This evaluation is only for one particular provision in the doc, so max
    # tp is 1.
    if tp:
        tp = 1
        fn = 0
        fp = 0
        return tp, fn, fp, tn

    # Since this is "anymatch", only max 1
    if fp:
        fp = 1

    # Since this is "anymatch", only max 1
    if fn:
        fn = 1

    # print("tp= {}, fn= {}, fp = {}, tn = {}".format(tp, fn, fp, tn))

    # there is no tn, because we deal with only annotations
    if diagnose_mode:
        print("tp = {}".format(tp))
        for i, hant_key in enumerate(sorted(tp_inst_map.keys())):
            hstart, hend, _ = hant_key
            print("\ntp #{}, start= {}, end= {}".format(i+1, hstart, hend))
            print(txt[hstart:hend])
            tp_inst_list = tp_inst_map[hant_key]
            for j, pred_ant in enumerate(tp_inst_list):
                print("     inst #%d, start2= %d, end2= %d, prob= %.6f" %
                      (j+1, pred_ant.start, pred_ant.end, pred_ant.prob))
                print("     ", end='')
                print("[[" + txt[pred_ant.start:pred_ant.end] + "]]")

        print("\n\nfn = {}".format(fn))
        for i, hant_x in enumerate(fn_inst_list):
            print("\nfn #%d, start= %d, end= %d, label = %s" %
                  (i+1, hant_x.start, hant_x.end, hant_x.label))
            print("     ", end='')
            print("[[" + txt[hant_x.start:hant_x.end] + "]]")

        print("\n\nfp = {}".format(fp))
        for i, pred_ant in enumerate(fp_inst_list):
            print("\nfp #%d, start= %d, end= %d, prob= %.6f" %
                  (i, pred_ant.start, pred_ant.end, pred_ant.prob))
            print("[[" + txt[pred_ant.start:pred_ant.end] + "]]")

    return tp, fn, fp, tn

# pylint: disable=pointless-string-statement
"""
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
"""

#
# utilities
#


# pylint: disable=pointless-string-statement
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


def calc_pred_status_with_prob(probs: List[float],
                               y_te: List[int]) -> Dict:
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


# nobody calls this
"""
def calc_pred_status(preds: List[float],
                     y_te: List[int]) -> Dict:
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
"""


def calc_pos_threshold_prob_status(probs: List[float],
                                   y_te: List[int],
                                   pos_threshold: float) -> Dict:
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


def calc_threshold_prob_status(probs: List[float],
                               y_te: List[int],
                               threshold: float) -> Dict:
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


def calc_prob_override_status(probs: List[float],
                              y_te: List[int],
                              threshold: float,
                              overrides: List[float]) -> Dict:
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


# the reference in kirke/eblearn/ebtrainer.y is removed already
"""
def calc_pred_override_status(preds,
                              y_te: List[int],
                              overrides: List[int]):
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
"""
