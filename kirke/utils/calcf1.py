import argparse
import logging
import pprint


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


def calc_pred_status(pred_id_val_list, gold_id_val_map, debug_mode=False):
    tn, fp, fn, tp = 0, 0, 0, 0

    for pred_id, pred_val, pred_score in pred_id_val_list:
        gval = gold_id_val_map[pred_id]
        if gval == 1:
            if pred_val == 1:
                tp += 1
                if debug_mode:
                    print('{}\t{}\t{}\t{}'.format('TP', pred_id, pred_val, pred_score))
            else:
                fn += 1
                if debug_mode:
                    print('{}\t{}\t{}\t{}'.format('xxFN', pred_id, pred_val, pred_score))
        else:
            if pred_val == 1:
                fp += 1
                if debug_mode:
                    print('{}\t{}\t{}\t{}'.format('xxFP', pred_id, pred_val, pred_score))
            else:
                tn += 1
                if debug_mode:
                    print('{}\t{}\t{}\t{}'.format('TN', pred_id, pred_val, pred_score))
    title = 'pred status'
    prec, recall, f1 = calc_precision_recall_f1(tn, fp, fn, tp, title)
    status = {'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
              'pred_threshold': 0.5,
              'prec': prec, 'recall': recall, 'f1': f1}
    return status


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('gold_file_name', help='gold tsv file, i.e. "id\\t{0,1}')
    parser.add_argument('pred_file_name', help='predict tsv file, i.e. "id\\t{0,1}')

    args = parser.parse_args()
    gold_file_name = args.gold_file_name
    pred_file_name = args.pred_file_name

    gold_id_val_map = {}
    with open(gold_file_name, 'rt') as fin:
        for line in fin:
            line = line.strip()
            cols = line.split('\t')

            doc_id = cols[0]
            binary_val = int(cols[1])
            gold_id_val_map[doc_id] = binary_val
            
    pred_id_val_list = []
    with open(pred_file_name, 'rt') as fin:
        for line in fin:
            line = line.strip()
            cols = line.split('\t')

            doc_id = cols[0]
            binary_val = int(cols[1])
            score = float(cols[2])            
            pred_id_val_list.append((doc_id, binary_val, score))

    result = calc_pred_status(pred_id_val_list, gold_id_val_map, debug_mode=True)
    pprint.pprint(result)
    
