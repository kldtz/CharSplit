#!/usr/bin/env python3

import argparse
import logging
import os

from typing import Any, Dict, List, Optional, Set, Tuple

import pprint


from kirke.utils import antutils, strutils, evalutils


# NOTE: Remove the following line to get rid of all logging messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


def test_antdoc_list(provision: str,
                     fname_list: List[str],
                     gold_dir: str,
                     pred_dir: str) -> Tuple[Dict[str, Any],
                                             Dict[str, Dict]]:
    logger.debug('test_antdoc_list(), len = %d', len(fname_list))

    fallout, tp, fn, fp, tn = 0, 0, 0, 0, 0
    log_json = {}

    gold_provision = 'effectivedate'

    for fcount, fname in enumerate(fname_list, 1):
        base_fname = os.path.basename(fname)
        gold_fname = os.path.join(gold_dir, base_fname)
        # List[EbProvisionAnnotation]
        gold_ant_fname = gold_fname.replace('.txt', '.ebdata')
        if os.path.exists(gold_ant_fname):
            prov_human_ant_list, unused_is_test_file = \
                antutils.load_prov_ebdata(gold_ant_fname, gold_provision)
            print('loading #{} from {}'.format(fcount, gold_ant_fname))
            for gidx, gold_ant in enumerate(prov_human_ant_list):
                print('  gold ant #{}\t{}'.format(gidx, gold_ant))
        else:
            gold_ant_fname = gold_ant_fname.replace('.ebdata', '.ant'),
            prov_human_ant_list = antutils.load_prov_ant(gold_ant_fname,
                                                         gold_provision)
            print('loading #{} from {}'.format(gold_ant_fname))
            for gidx, gold_ant in enumerate(prov_human_ant_list):
                print('  gold ant #{}\t{}'.format(gidx, gold_ant))            

        # this uses ebsentutils.ProvisionAnnotation
        # prov_human_ant_list = [hant for hant in ebantdoc.prov_annotation_list
        # if hant.label == self.provision]

        pred_ant_fname = '{}/{}'.format(pred_dir,
                                        base_fname.replace('.txt', '.ebprov.ants.json'))
        print('loading from {}'.format(pred_ant_fname))
        pred_ant_list = antutils.load_pred_prov_ant(pred_ant_fname, provision)
        # for pidx, pred_ant in enumerate(pred_ant_list):
        #     print('  pred ant #{}\t{}'.format(pidx, pred_ant))            

        xtp, xfn, xfp, xtn, json_return = \
            evalutils.calc_doc_ant_confusion_matrix_offline(prov_human_ant_list,
                                                            pred_ant_list,
                                                            fname)
        tp += xtp
        fn += xfn
        fp += xfp
        tn += xtn
        log_json[fname] = json_return

    title = 'evaluation for provision: {}'.format(provision)
    prec, recall, f1 = evalutils.calc_precision_recall_f1(tn, fp, fn, tp,
                                                          title=title)
    # max_recall = (tp + fn - fallout) / (tp + fn)
    # logger.info("MAX RECALL = %d, FALLOUT = %d", max_recall, fallout)
    ant_status = {}
    ant_status['eval_status'] = {'confusion_matrix': {'tn': tn, 'fp': fp,
                                                      'fn': fn, 'tp': tp},
                                 'prec': prec, 'recall': recall, 'f1': f1}
    
    return ant_status, log_json
    


def eval_ants(provision: str,
              fname: str,
              gold_dir: str,
              pred_dir: str) -> Tuple[Dict, Dict]:
    """Read the file name to be evaluated.

    Returns:
        Dict: precision, recall, etc
        Dict: error analysis
    """

    fname_list = strutils.load_str_list(fname)

    ant_status, logjson = test_antdoc_list(provision=provision,
                                           fname_list = fname_list,
                                           gold_dir=gold_dir,
                                           pred_dir=pred_dir)
    print('ant_status: {}'.format(ant_status))

    print('logjson:')
    pprint.pprint(logjson)
                         

if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='Evaluate the annotations against gold data')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('--prov', default='effectivedate_cand', help='provision')
    parser.add_argument('--pred_dir')
    parser.add_argument('file',  nargs='?',
                        help='input file')

    args = parser.parse_args()

    # if not PROVISION or not args.file:
    #     print('usage: printProvAnt.py <prov> <fn_list.txt>')

    # txt_list_fname = 'effectivedate_pred_doc3_list.txt'
    txt_list_fname = 'effectivedate_test_doclist.txt'
    if args.file:
        txt_list_fname = args.file

    # pred_dir = 'dir-out'
    pred_dir = '/eb_files/kirke_tmp/dir-work'
    if args.pred_dir:
        pred_dir = args.pred_dir
    
    eval_ants(provision=args.prov,
              # fname='effectivedate_train_doc3_list.txt',
              fname=txt_list_fname,
              gold_dir='export-train',
              pred_dir=pred_dir)
