#!/usr/bin/env python3

import sys
import argparse
import os
import requests

# pylint: disable=too-many-locals
def upload_train_dir(url_st: str, upload_dir: str, candidate_types: str, nbest: int = -1) -> None:
    txt_fnames, ant_fnames = [], []
    offsets_fnames = []
    for file in os.listdir(upload_dir):
        fname = '{}/{}'.format(upload_dir, file)
        if file.endswith(".txt"):
            txt_fnames.append(fname)
        elif file.endswith(".ant"):
            ant_fnames.append(fname)
        elif file.endswith(".offsets.json"):
            offsets_fnames.append(fname)

    if not txt_fnames:
        print("cannot find any .txt files", file=sys.stderr)
        return

    file_tuple_list = []
    ant_fname_set = set(ant_fnames)
    offsets_fname_set = set(offsets_fnames)
    for txt_fname in txt_fnames:
        ant_fname = txt_fname.replace('.txt', '.ant')
        offsets_fname = txt_fname.replace('.txt', '.offsets.json')
        if ant_fname in ant_fname_set:
            file_tuple_list.append(('file', open(txt_fname, 'rt', encoding='utf-8', newline='')))
            print("uploading [{}]".format(txt_fname))
            file_tuple_list.append(('file', open(ant_fname, 'rt', encoding='utf-8')))
            print("uploading [{}]".format(ant_fname))
            if offsets_fname in offsets_fname_set:
                print("uploading [{}]".format(offsets_fname))
                file_tuple_list.append(('file', open(offsets_fname, 'rt', encoding='utf-8')))
        else:
            print("cannot find matching ant file for {}".format(txt_fname), file=sys.stderr)

    print('candidate_types: %s' % (candidate_types, ))
    payload = {'candidate_types': candidate_types,
               'nbest': nbest}

    txt_fname_set = set(txt_fnames)
    for ant_fname in ant_fnames:
        txt_fname = ant_fname.replace('.ant', '.txt')
        if not txt_fname in txt_fname_set:
            print("cannot find matching ant file for {}".format(txt_fname), file=sys.stderr)

    print("Number of file uploaded: {}".format(len(file_tuple_list)))
    # print("file_tuple_list = {}".format(file_tuple_list))
    # payload = {'custom_id': 'custom_id2'}
    req = requests.post(url_st, files=file_tuple_list, data=payload, timeout=6000)  # type: ignore
    print(req.text)


# pylint: disable=too-many-locals
def main():
    parser = argparse.ArgumentParser(description='identify the language')
    parser.add_argument('-v', '--verbosity', help='increase output verbosity')
    parser.add_argument('--url', help='url to post the files')
    parser.add_argument('--custid', default='12345', help='custom-id')
    parser.add_argument('--candidate_types', default='SENTENCE',
                        help='SENTENCE, CURRENCY, DATE, ADDRESS, NUMBER, PERCENT')
    parser.add_argument('--nbest', default=-1, help='url to post the files')
    parser.add_argument('upload_dir', help='directory to upload')

    args = parser.parse_args()
    if args.verbosity:
        print('verbosity turned on')

    url = 'http://127.0.0.1:8000/custom-train/{}'.format(args.custid)
    if args.url:
        url = args.url

    if args.nbest:
        nbest = int(args.nbest)

    # provision = 'cust_{}'.format(args.custid)
    upload_train_dir(url, args.upload_dir, args.candidate_types, nbest)


if __name__ == '__main__':
    main()