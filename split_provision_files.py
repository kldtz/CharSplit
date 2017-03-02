#!/usr/bin/env python


import argparse
from utils import splittrte, osutils

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    parser.add_argument('--docs', help='a file containing list of .txt files')
    parser.add_argument('--provisions', help='a comma separate list of provisions')
    parser.add_argument('--work_dir', help='output directory for cached documents')
    parser.add_argument('--model_dir', help='output directory for trained models')

    args = parser.parse_args()
    print("args.provisions = [{}]".format(args.provisions))
    print("args.docs = [{}]".format(args.docs))
    print("args.model_dir = [{}]".format(args.model_dir))

    work_dir = None
    if args.work_dir is not None:
        work_dir = args.work_dir
    osutils.mkpath(work_dir)
    osutils.mkpath(args.model_dir)

    provision_list = args.provisions.split(',')

    provision_filelist_map = splittrte.provisions_split(provision_list, args.docs, work_dir=work_dir)
    for provision in provision_list:
        eb_antdoc_list = provision_filelist_map[provision]
        antdoc_fn_list = "{}/{}.doclist.txt".format(args.model_dir, provision)
        splittrte.save_antdoc_fn_list(eb_antdoc_list, antdoc_fn_list)
        
    logging.info('Done.')
