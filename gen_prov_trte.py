#!/usr/bin/env python

import argparse
import logging

from sklearn.model_selection import train_test_split

from eblearn import ebtext2antdoc
from utils import splittrte, osutils


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    parser.add_argument('--docs', required=True, help='a file containing list of .txt files')
    parser.add_argument('--provisions', required=True, help='a comma separate list of provisions')
    parser.add_argument('--work_dir', required=True, help='output directory for cached documents')
    parser.add_argument('--model_dirs', required=True, help='output directory for trained models')

    args = parser.parse_args()
    print("args.provisions = [{}]".format(args.provisions))
    print("args.docs = [{}]".format(args.docs))
    print("args.model_dirs = [{}]".format(args.model_dirs))

    work_dir = None
    if args.work_dir is not None:
        work_dir = args.work_dir
        
    osutils.mkpath(work_dir)

    model_dir_list = args.model_dirs.split(",")
    for moddir in model_dir_list:
        osutils.mkpath(moddir)

    provision_list = args.provisions.split(',')
    txt_fn_list = args.docs
    
    eb_antdoc_list = ebtext2antdoc.doclist_to_ebantdoc_list(txt_fn_list, work_dir=work_dir)

    for provision in provision_list:
        X = eb_antdoc_list
        y = [provision in ebantdoc.get_provision_set()
             for ebantdoc in eb_antdoc_list]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        for moddir in model_dir_list:
            antdoc_fn_list = "{}/{}.doclist.txt".format(moddir, provision)
            splittrte.save_antdoc_fn_list(eb_antdoc_list, antdoc_fn_list)

            train_doclist_fn = "{}/{}_train_doclist.txt".format(moddir, provision)    
            splittrte.save_antdoc_fn_list(X_train, train_doclist_fn)

            test_doclist_fn = "{}/{}_test_doclist.txt".format(moddir, provision)
            splittrte.save_antdoc_fn_list(X_test, test_doclist_fn)
        
    logging.info('Done.')
