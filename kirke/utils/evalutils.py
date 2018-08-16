from collections import defaultdict, namedtuple
import logging
import re
# pylint: disable=unused-import
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

from kirke.utils import mathutils, wordutils
from kirke.utils.ebsentutils import ProvisionAnnotation

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# pylint: disable=fixme
# TODO, verify IS_DIAGNOSE_MODE is used for logformat?
IS_DIAGNOSE_MODE = True
IS_DEBUG = False


AnnotationWithProb = namedtuple('AnnotationWithProb', ['label', 'start', 'end', 'prob'])

# pylint: disable=C0103
# label_start_end_list is of type prov_annotation_list

# Any = Union[AnnotationWithProb, ProvisionAnnotation]
def find_annotation_overlap(start: int,
                            end: int,
                            label_start_end_list: Optional[List]) \
                            -> List:
    """Find annotations that overlaps with 'start' and 'end'.

    The annotation list is expected to have minimally 'start' and 'end' attributes.

    Args:
        obvious

    Returns:
        The list of annotations that overlaps with 'start' and 'end'.
    """

    result_list = []  # type: List
    if not label_start_end_list:
        return result_list
    for ant in label_start_end_list:
        if mathutils.start_end_overlap((start, end), (ant.start, ant.end)):
            result_list.append(ant)
    return result_list


# label_start_end are dict
def find_annotation_overlap_x2(start: int,
                               end: int,
                               label_start_end_list: Optional[List]) \
                               -> List:
    result_list = []  # type: List
    if not label_start_end_list:
        return result_list
    for ant in label_start_end_list:
        if mathutils.start_end_overlap((start, end), (ant['start'], ant['end'])):
            result_list.append(ant)
    return result_list


def calc_precision_recall_f1(tn: int,
                             fp: int,
                             fn: int,
                             tp: int,
                             title: Optional[str] = None) \
                             -> Tuple[float, float, float]:
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


def aggregate_ant_status_list(alist):
    result_map = defaultdict(int)
    for ant_status in alist:
        access_field = 'ant_status'
        if not ant_status.get(access_field):
            access_field = 'eval_status'
        conf_mtx = ant_status[access_field]['confusion_matrix']
        for pred_type, count in conf_mtx.items():
            result_map[pred_type] += count
        threshold = ant_status[access_field]['threshold']  # we just take the last one

    prec, recall, f1 = calc_precision_recall_f1(result_map['tn'],
                                                result_map['fp'],
                                                result_map['fn'],
                                                result_map['tp'])

    return {'ant_status': {'recall': recall,
                           'prec': prec,
                           'f1': f1,
                           'threshold': threshold,
                           'confusion_matrix': {'tp': result_map['tp'],
                                                'tn': result_map['tn'],
                                                'fp': result_map['fp'],
                                                'fn': result_map['fn']}}}

# pylint: disable=too-many-arguments, too-many-locals, too-many-branches, too-many-statements
def calc_doc_ant_confusion_matrix(prov_human_ant_list: List[ProvisionAnnotation],
                                  ant_list: List[Dict],
                                  file_id: str,
                                  txt: str,
                                  threshold: float,
                                  is_raw_mode: bool) \
                                  -> Tuple[int, int, int, int, int,
                                           Dict[str, List[Tuple[int, int, str, float, str]]]]:
    """Calculate the confusion matrix for one document only, based only on offsets.

    Args:
        prov_human_ant_list: human annotation list, only for 1 provision
        ant_list: annotation predicted by machine
        file_id: document id
        txt: text of the document
        threshold: the threshold to decide if a provision is positive

    Returns:
        tp, fn, fp, tn, fallout, Dictionary of tp, fn, fp, tn
    """
    tp, fp, tn, fn, fallout = 0, 0, 0, 0, 0
    # pylint: disable=line-too-long
    json_return = {'tp': [], 'fn': [], 'fp': []}  # type: Dict[str, List[Tuple[int, int, str, float, str]]]
    pred_ant_list = []  # type: List[AnnotationWithProb]
    for adict in ant_list:
        pred_ant_list.append(AnnotationWithProb(adict['label'],
                                                adict['start'],
                                                adict['end'],
                                                adict['prob']))
    linebreaks = re.compile("[\n\r]")
    # pylint: disable=line-too-long
    tp_inst_map = defaultdict(list)  # type: DefaultDict[Tuple[str, int, int, str], List[AnnotationWithProb]]
    fp_inst_list = []
    # pylint: disable=line-too-long
    fn_inst_map = defaultdict(list)  # type: DefaultDict[Tuple[str, int, int, str], List[AnnotationWithProb]]
    tp_fn_set = set([])  # type: Set[AnnotationWithProb]


    # checks predicted annotations against human annotations
    for hant in prov_human_ant_list:
        pred_overlap_list = find_annotation_overlap(hant.start, hant.end, pred_ant_list)
        # Any naive user would be using this, just all human annotation and predict annotation.
        # We don't assume there are FN annotation with prob in here.
        if is_raw_mode:
            prob = 0.0
            if pred_overlap_list:
                prob = max([x.prob for x in pred_overlap_list])

            if prob >= threshold:
                tp_inst_map[(file_id, hant.start, hant.end, hant.label)] = pred_overlap_list
                tp += 1
            else:
                fn_inst_map[(file_id, hant.start, hant.end, hant.label)] = pred_overlap_list
                fn += 1
        else:  # this is used by logformat to get prob for FN
            # postproc adds any annotations that have scores above the threshold or
            # have an overlap with the list of human annotations
            # so all tps and fns should be in pred_ant_list, if not, something is
            # wrong with pred_ant_list or prov_human_ant_list
            if pred_overlap_list:
                prob = max([x.prob for x in pred_overlap_list])
                if prob >= threshold:
                    tp_inst_map[(file_id, hant.start, hant.end, hant.label)] = pred_overlap_list
                    tp += 1
                else:
                    fn_inst_map[(file_id, hant.start, hant.end, hant.label)] = pred_overlap_list
                    fn += 1
                    if prob < 0:
                        fallout += 1
            else:
                logger.warning("Human annotation not present in the list of annotations, something is wrong!")
        tp_fn_set |= set(pred_overlap_list)

    # any remaining predicted annotations are false positives or true negatives
    for pant in pred_ant_list:
        if pant in tp_fn_set:
            continue
        if pant.prob > threshold:
            fp_inst_list.append(pant)
            fp += 1

    # we don't care about reporting true negatives
    if IS_DIAGNOSE_MODE:
        for xhant in sorted(tp_inst_map.keys()):
            _, hstart, hend, label = xhant
            tp_inst_list = tp_inst_map[xhant]
            tp_txt = " ".join([txt[x.start:x.end] for x in tp_inst_list])  # type: str
            min_start = min([x.start for x in tp_inst_list])  # type: int
            max_end = max([x.end for x in tp_inst_list])  # type: int
            max_prob = max([x.prob for x in tp_inst_list])  # type: float
            if IS_DEBUG:
                print("tp\t{}\t{}\t{}".format(file_id, linebreaks.sub(" ", tp_txt), str(max_prob)))
            json_return['tp'].append((min_start,
                                      max_end,
                                      label,
                                      max_prob,
                                      linebreaks.sub(" ", tp_txt)))

        for xhant in sorted(fn_inst_map.keys()):
            _, hstart, hend, label = xhant
            if is_raw_mode:
                fn_txt = txt[hstart:hend]
                max_prob = -1.0
                if IS_DEBUG:
                    print("fn\t{}\t{}\t{}".format(file_id, linebreaks.sub(" ", fn_txt), str(max_prob)))
                json_return['fn'].append((hstart,
                                          hend,
                                          label,
                                          max_prob,
                                          linebreaks.sub(" ", fn_txt)))
            else:
                fn_inst_list = fn_inst_map[xhant]
                fn_txt = " ".join([txt[x.start:x.end] for x in fn_inst_list])
                min_start = min([x.start for x in fn_inst_list])
                max_end = max([x.end for x in fn_inst_list])
                max_prob = max([x.prob for x in fn_inst_list])
                if IS_DEBUG:
                    print("fn\t{}\t{}\t{}".format(file_id, linebreaks.sub(" ", fn_txt), str(max_prob)))
                json_return['fn'].append((min_start,
                                          max_end,
                                          label,
                                          max_prob,
                                          linebreaks.sub(" ", fn_txt)))

        for pred_ant in fp_inst_list:
            if IS_DEBUG:
                print("fp\t{}\t{}\t{}".format(file_id, linebreaks.sub(" ", txt[pred_ant.start:pred_ant.end]), str(pred_ant.prob)))
            json_return['fp'].append((pred_ant.start,
                                      pred_ant.end,
                                      pred_ant.label,
                                      pred_ant.prob,
                                      linebreaks.sub(" ", txt[pred_ant.start:pred_ant.end])))
    return tp, fn, fp, tn, fallout, json_return


# for 'title', we want to match any title annotation
# if any matched, we passed.  Don't care about any other.
# pylint: disable=too-many-branches
def calc_doc_ant_confusion_matrix_anymatch(prov_human_ant_list: List[ProvisionAnnotation],
                                           ant_list: List[Dict],  # this is machine annotation
                                           file_id: str,
                                           txt: str) \
    -> Tuple[int, int, int, int,
             Dict[str, List[Tuple[int, int, str, float, str]]]]:
    """Calculate the confusion matrix for one document, if there is any match by offset or string.

    This differs from calc_doc_ant_confusion_matrix() because if there is any annotation match,
    either by offsets or surface form, TP will be max 1.  FP and FN can also maximally be 1.

    There is no threshold in the Args because there is no prob.

    Args:
        prov_human_ant_list: human annotation list, only for 1 provision
        ant_list: annotation predicted by machine
        file_id: document id
        txt: text of the document

    Returns:
        tp, fn, fp, tn, Dictionary of List of tp, fn, fp, tn
    """
    linebreaks = re.compile("[\n\r]")
    # pylint: disable=line-too-long
    json_return = {'tp': [], 'fn': [], 'fp': []}  # type: Dict[str, List[Tuple[int, int, str, float, str]]]
    tp, fp, tn, fn = 0, 0, 0, 0
    # print("calc_doc_ant_confusion_matrix:")

    # this is machine annotations
    pred_ant_list = []  # type: List[AnnotationWithProb]
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
    if tp > 0:
        tp = 1
        fn = 0
        fp = 0
    else:
        # Since this is "anymatch", only max 1
        if fp > 0:
            fp = 1

            # Since this is "anymatch", only max 1
        if fn:
            fn = 1

    # print("tp= {}, fn= {}, fp = {}, tn = {}".format(tp, fn, fp, tn))

    # there is no tn, because we deal with only annotations
    if IS_DIAGNOSE_MODE:
        for xhant in sorted(tp_inst_map.keys()):
            hstart, hend, label = xhant
            tp_inst_list = tp_inst_map[xhant]
            tp_txt = " ".join([txt[x.start:x.end] for x in tp_inst_list])
            min_start = min([x.start for x in tp_inst_list])
            max_end = max([x.end for x in tp_inst_list])
            prob = max([x.prob for x in tp_inst_list])
            if IS_DEBUG:
                print("tp\t{}\t{}\t{}".format(file_id, linebreaks.sub(" ", tp_txt), str(prob)))
            json_return['tp'].append((min_start,
                                      max_end,
                                      label,
                                      prob,
                                      linebreaks.sub(" ", tp_txt)))

        for prov_ant in fn_inst_list:
            hstart, hend, label = prov_ant.start, prov_ant.end, prov_ant.label
            fn_txt = txt[hstart:hend]
            prob = -1.0
            if IS_DEBUG:
                print("fn\t{}\t{}\t{}".format(file_id, linebreaks.sub(" ", fn_txt), str(prob)))
            json_return['fn'].append((hstart,
                                      hend,
                                      label,
                                      prob,
                                      linebreaks.sub(" ", fn_txt)))


        for pred_ant in fp_inst_list:
            if IS_DEBUG:
                print("fp\t{}\t{}\t{}".format(file_id, linebreaks.sub(" ", txt[pred_ant.start:pred_ant.end]), str(pred_ant.prob)))
            json_return['fp'].append((pred_ant.start,
                                      pred_ant.end,
                                      pred_ant.label,
                                      pred_ant.prob,
                                      linebreaks.sub(" ", txt[pred_ant.start:pred_ant.end])))

    return tp, fn, fp, tn, json_return

# pylint: disable=pointless-string-statement


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
