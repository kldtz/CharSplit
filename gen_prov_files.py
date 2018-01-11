#!/usr/bin/env python

import argparse
import json
import logging
import os

from collections import defaultdict

from sklearn.model_selection import train_test_split

from kirke.utils import splittrte, osutils

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_provisions_istest_in_ebdata(filename):
    result = []
    is_test_set = False
    with open(filename, 'rt') as handle:
        parsed = json.load(handle)
        for provision, ajson_list in parsed['ants'].items():
            result.append(provision)
        is_test_set = parsed.get('isTestSet', False)

    return result, is_test_set


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate prov filesin dir-provfiles.')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    # parser.add_argument('--docs', required=True, help='a file containing list of .txt files')
    # parser.add_argument('--work_dir', required=True, help='output directory for cached documents')
    # parser.add_argument('--model_dirs', required=True, help='output directory for trained models')
    parser.add_argument('--data-dir', help='directory containing the training files')
    parser.add_argument('--provfiles-dir', help='directory that will have all the prov.doclist.txt')
    parser.add_argument('--work_dir', help='output directory for cached documents')
    parser.add_argument('--model_dirs', help='output directory for trained models')
    

    args = parser.parse_args()
    # print("args.docs = [{}]".format(args.docs))
    # print("args.model_dirs = [{}]".format(args.model_dirs))

    work_dir = 'dir-work'
    if args.work_dir is not None:
        work_dir = args.work_dir
    osutils.mkpath(work_dir)

    provfile_dir = 'dir-provfiles'
    if args.provfiles_dir is not None:
        provfile_dir = args.provfiles
    osutils.mkpath(provfile_dir)

    model_dir_list = 'dir-model,dir-scut-model'.split(',')
    if args.model_dirs is not None:
        model_dir_list = args.model_dirs.split(',')
    for moddir in model_dir_list:
        osutils.mkpath(moddir)

    data_dir = 'export-train'
    if args.data_dir is not None:
        dir_data = args.data_dir

    fileid_fnlist_map = defaultdict(list)
    num_ebdata = 0
    for file_name in os.listdir(data_dir):
        file_name = '{}/{}'.format(data_dir, file_name)
        print("file_name = [{}]".format(file_name))
        if file_name.endswith('.txt'):
            fileid = file_name[:-4]
            filetype = 'txt'
            fileid_fnlist_map[fileid].append((filetype, file_name))
        elif file_name.endswith('.htm'):
            fileid = file_name[:-4]
            filetype = 'htm'
            fileid_fnlist_map[fileid].append((filetype, file_name))
        elif file_name.endswith('.html'):
            fileid = file_name[:-5]
            filetype = 'htm'
            fileid_fnlist_map[fileid].append((filetype, file_name))            
        elif file_name.endswith('.pdf'):
            fileid = file_name[:-4]
            filetype = 'pdf'
            fileid_fnlist_map[fileid].append((filetype, file_name))
        elif file_name.endswith('.ebdata'):
            fileid = file_name[:-7]
            filetype = 'ebdata'
            fileid_fnlist_map[fileid].append((filetype, file_name))
            num_ebdata += 1
        else:
            print("unknown extension: {}".format(file_name))

    print('number of txt-ebfiles: {}'.format(len(fileid_fnlist_map)))
    print('number of ebfiles: {}'.format(num_ebdata))

    prov_fnlist_map = defaultdict(list)
    prov_train_fnlist_map = defaultdict(list)
    prov_test_fnlist_map = defaultdict(list)        
    
    for fileid, fnlist in fileid_fnlist_map.items():
        ebdata_fn = ''
        fname = ''
        for ftype, fn in fnlist:
            if ftype == 'ebdata':
                ebdata_fn = fn
            #elif ftype == 'html':
            #    fname = fn
            #elif ftype == 'htm':
            #    fname = fn
            #elif ftype == 'pdf':
            #    fname = fn
            # lowest priority
            # elif ftype == 'txt' and not fname:
            elif ftype == 'txt':
                fname = fn         
            
        prov_list, is_test = load_provisions_istest_in_ebdata(ebdata_fn)

        # print('is_test = {}'.format(is_test))
        
        for label in prov_list:

            prov_fnlist_map[label].append(fname)
            # below are not used
            if is_test:
                prov_test_fnlist_map[label].append(fname)
            else:
                prov_train_fnlist_map[label].append(fname)

    for prov, fnlist in prov_fnlist_map.items():
        out_fn = '{}/{}.doclist.txt'.format(provfile_dir, prov)
        with open(out_fn, 'wt') as fout:
            for fn in fnlist:
                print(fn, file=fout)
        print('wrote {}'.format(out_fn))

    for prov, fnlist in prov_train_fnlist_map.items():
        out_fn = '{}/{}_train_doclist.txt'.format(provfile_dir, prov)
        with open(out_fn, 'wt') as fout:
            for fn in fnlist:
                print(fn, file=fout)
        print('wrote {}'.format(out_fn))

    for prov, fnlist in prov_test_fnlist_map.items():
        out_fn = '{}/{}_test_doclist.txt'.format(provfile_dir, prov)
        with open(out_fn, 'wt') as fout:
            for fn in fnlist:
                print(fn, file=fout)
        print('wrote {}'.format(out_fn))                

