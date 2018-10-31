#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import os
import sys
# pylint: disable=unused-import
from typing import Any, Dict, List, Optional

import requests

from kirke.utils import modelfileutils


MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'eb_files_test/pymodel'

UNIT_TEST_PROVS = ['change_control',
                   'choiceoflaw',
                   'date',
                   'effectivedate',
                   'force_majeure',
                   'limliability',
                   'noncompete',
                   'party',
                   'remedy',
                   'renewal',
                   'termination',
                   'term',
                   'title',
                   'warranty',
                   'cust_9.1005']


def upload_annotate_doc(file_name: str, prov_list: Optional[List[str]] = None) \
    -> Dict[str, Any]:
    if not prov_list:
        prov_list = UNIT_TEST_PROVS
    text = post_unittest_annotate_document(file_name, prov_list)
    ajson = json.loads(text)
    return ajson


def post_annotate_document(file_name: str,
                           prov_list: List[str],
                           is_detect_lang: bool = False,
                           is_classify_doc: bool = False,
                           is_show_header: bool = False,
                           is_dev_mode: bool = False) -> str:

    url = 'http://localhost:8000/annotate-doc'
    payload = {'types': ','.join(prov_list)}  # type: Dict[str, Any]

    if is_detect_lang:
        payload['detect-lang'] = True
    if is_classify_doc:
        payload['classify-doc'] = True
    if is_dev_mode:
        payload['dev-mode'] = True

    txt_file = Path(file_name)
    if txt_file.is_file() and file_name.endswith('.txt'):
        offset_filename = file_name.replace('.txt', '.offsets.json')
        if os.path.exists(offset_filename):
            files = [('file', open(file_name, 'rt', encoding='utf-8')),
                     ('file', open(offset_filename, 'rt', encoding='utf-8'))]
        else:
            files = {'file': open(file_name, 'rt', encoding='utf-8')}  # type: ignore

        resp = requests.post(url, files=files, data=payload)

        if is_show_header:
            print('status: [{}]'.format(resp.status_code), file=sys.stderr)
            print(resp.headers, file=sys.stderr)
        return resp.text

    print("file '{}' is not a valid file".format(file_name), file=sys.stderr)
    raise ValueError


def post_unittest_annotate_document(file_name: str,
                                    prov_list: Optional[List[str]] = None) -> str:
    if not prov_list:
        prov_list = UNIT_TEST_PROVS
    result = post_annotate_document(file_name,
                                    prov_list,
                                    is_detect_lang=True,
                                    is_classify_doc=True,
                                    is_dev_mode=True)
    return result


# pylint: disable=too-many-locals
def upload_train_dir(custid: str,
                     upload_dir: str,
                     candidate_types: str,
                     nbest: int = -1) -> str:
    resp = upload_train_dir_resp(custid,
                                 upload_dir,
                                 candidate_types,
                                 nbest)
    return resp.text


# pylint: disable=too-many-locals
def upload_train_dir_resp(custid: str,
                          upload_dir: str,
                          candidate_types: str,
                          nbest: int = -1):
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
        raise ValueError

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

    # print('candidate_types: %s' % (candidate_types, ))
    payload = {'candidate_types': candidate_types,
               'nbest': nbest}  # type: Dict[str, Any]

    txt_fname_set = set(txt_fnames)
    for ant_fname in ant_fnames:
        txt_fname = ant_fname.replace('.ant', '.txt')
        if not txt_fname in txt_fname_set:
            print("cannot find matching ant file for {}".format(txt_fname), file=sys.stderr)
            raise ValueError

    print("Number of file uploaded: {}".format(len(file_tuple_list)))

    url = 'http://127.0.0.1:8000/custom-train/{}'.format(custid)
    resp = requests.post(url,
                         files=file_tuple_list,
                         data=payload,
                         timeout=6000)
    return resp


def main() -> None:
    parser = argparse.ArgumentParser(description='identify the language')
    parser.add_argument('-v', '--verbosity', help='increase output verbosity')
    parser.add_argument('-u', '--url', help='url to post the file')
    parser.add_argument('-l', '--lang', action='store_true', help='to detect lang')
    parser.add_argument('--header', action='store_true', help='to print header')
    parser.add_argument('--doccat', action='store_true', help='to classify document')

    parser.add_argument('filename')

    args = parser.parse_args()
    if args.verbosity:
        print('verbosity turned on')

    result = post_unittest_annotate_document(args.filename)
    print(result)


# pylint: disable=C0103
if __name__ == '__main__':
    main()
