#!/usr/bin/env python

import argparse
import logging
import sys
import requests
import os


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def upload_train_dir(url, upload_dir):
    txt_fnames = []
    ant_fnames = []    
    for file in os.listdir(upload_dir):
        fn = '{}/{}'.format(args.upload_dir, file)        
        if file.endswith(".txt"):
            txt_fnames.append(fn)
        elif file.endswith(".ant"):
            ant_fnames.append(fn)

    if not txt_fnames:
        print("cannot find any .txt files", file=sys.stderr)
        return -1

    file_tuple_list = []            
    ant_fname_set = set(ant_fnames)
    for txt_fname in txt_fnames:
        ant_fname = txt_fname.replace('.txt', '.ant')
        if ant_fname in ant_fname_set:
            file_tuple_list.append(('file', open(txt_fname, 'rt', encoding='utf-8')))
            file_tuple_list.append(('file', open(ant_fname, 'rt', encoding='utf-8')))            
        else:
            print("cannot find matching ant file for {}".format(txt_fname), file=sys.stderr)

    txt_fname_set = set(txt_fnames)
    for ant_fname in ant_fnames:
        txt_fname = ant_fname.replace('.ant', '.txt')
        if not txt_fname in txt_fname_set:
            print("cannot find matching ant file for {}".format(txt_fname), file=sys.stderr)            
            
    print("Number of file uploaded: {}".format(len(file_tuple_list)))
    # print("file_tuple_list = {}".format(file_tuple_list))
    # payload = {'custom_id': 'custom_id2'}
    r = requests.post(url, files=file_tuple_list, timeout=600)
    print(r.text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='identify the language')
    parser.add_argument('-v','--verbosity', help='increase output verbosity')
    parser.add_argument('-u','--url', help='url to post the files')
    parser.add_argument('-c','--custid', default='12345', help='custom-id')        
    parser.add_argument('upload_dir', help='directory to upload')

    args = parser.parse_args()
    if args.verbosity:
        print('verbosity turned on')
    
    url = 'http://127.0.0.1:8000/custom-train/{}'.format(args.custid)
    if args.url:
        url = args.url

    # provision = 'cust_{}'.format(args.custid)
    upload_train_dir(url, args.upload_dir)
