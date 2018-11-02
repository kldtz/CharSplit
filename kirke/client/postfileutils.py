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


def upload_annotate_doc(file_name: str,
                        prov_list: Optional[List[str]] = None,
                        is_classify_doc: bool = False,
                        is_detect_lang: bool = False,
                        is_dev_mode: bool = True) \
                        -> Dict[str, Any]:
    if prov_list is None:
        prov_list = []
    text = post_annotate_document(file_name,
                                  prov_list,
                                  is_detect_lang=is_detect_lang,
                                  is_classify_doc=is_classify_doc,
                                  is_dev_mode=is_dev_mode)
    ajson = json.loads(text)
    return ajson


def upload_unittest_annotate_doc(file_name: str,
                                 prov_list: Optional[List[str]] = None,
                                 is_classify_doc: bool = True,
                                 is_detect_lang: bool = True) \
                                 -> str:
    if prov_list is None:
        prov_list = UNIT_TEST_PROVS
    result = post_annotate_document(file_name,
                                    prov_list,
                                    is_detect_lang=is_detect_lang,
                                    is_classify_doc=is_classify_doc,
                                    is_dev_mode=True)
    return result


# pylint: disable=too-many-arguments
def post_annotate_document(file_name: str,
                           prov_list: List[str],
                           *,
                           is_classify_doc: bool = False,
                           is_detect_lang: bool = False,
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


# pylint: disable=too-many-locals
def upload_train_dir(custid: str,
                     upload_dir: str,
                     candidate_types: str,
                     nbest: int = -1,
                     url_prefix: str = 'http://127.0.0.1:8000/custom-train') \
                     -> str:
    resp = upload_train_dir_resp(custid,
                                 upload_dir,
                                 candidate_types=candidate_types,
                                 nbest=nbest,
                                 url_prefix=url_prefix)
    return resp.text


# pylint: disable=too-many-locals
def upload_train_dir_resp(custid: str,
                          upload_dir: str,
                          candidate_types: str,
                          nbest: int = -1,
                          url_prefix: str = 'http://127.0.0.1:8000/custom-train'):
    txt_fnames, ant_fnames = [], []
    offsets_fnames = []
    pdfxml_fnames = []
    for file in os.listdir(upload_dir):
        fname = '{}/{}'.format(upload_dir, file)
        if file.endswith(".txt"):
            txt_fnames.append(fname)
        elif file.endswith(".ant"):
            ant_fnames.append(fname)
        elif file.endswith(".offsets.json"):
            offsets_fnames.append(fname)
        elif file.endswith(".pdf.xml"):
            pdfxml_fnames.append(fname)

    if not txt_fnames:
        print("cannot find any .txt files", file=sys.stderr)
        raise ValueError

    file_tuple_list = []
    ant_fname_set = set(ant_fnames)
    offsets_fname_set = set(offsets_fnames)
    pdfxml_fname_set = set(pdfxml_fnames)
    for txt_fname in txt_fnames:
        ant_fname = txt_fname.replace('.txt', '.ant')
        offsets_fname = txt_fname.replace('.txt', '.offsets.json')
        pdfxml_fname = txt_fname.replace('.txt', '.pdf.xml')
        if ant_fname in ant_fname_set:
            file_tuple_list.append(('file', open(txt_fname, 'rt', encoding='utf-8', newline='')))
            print("uploading [{}]".format(txt_fname))
            file_tuple_list.append(('file', open(ant_fname, 'rt', encoding='utf-8')))
            print("uploading [{}]".format(ant_fname))
            if offsets_fname in offsets_fname_set:
                print("uploading [{}]".format(offsets_fname))
                file_tuple_list.append(('file', open(offsets_fname, 'rt', encoding='utf-8')))
            if pdfxml_fname in pdfxml_fname_set:
                print("uploading [{}]".format(pdfxml_fname))
                file_tuple_list.append(('file', open(pdfxml_fname, 'rt', encoding='utf-8')))
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

    url = '{}/{}'.format(url_prefix, custid)
    resp = requests.post(url,
                         files=file_tuple_list,
                         data=payload,
                         timeout=6000)
    return resp


def main():
    parser = argparse.ArgumentParser(description='identify the language')
    parser.add_argument('-v', '--verbosity', help='increase output verbosity')
    parser.add_argument('--custid', default='12345', help='custom-id')
    parser.add_argument('--cmd', help='Examples: upload_dir, annotate_doc, annotate_unittest_doc')
    parser.add_argument('--provision', help='provision instead of custid')
    parser.add_argument('--provisions', default='', help='provision instead of custid')
    parser.add_argument('--doccat', action='store_true', help='to classify document')
    parser.add_argument('--header', action='store_true', help='to print header')
    parser.add_argument('--lang', action='store_true', help='to detect lang')
    parser.add_argument('--url', help='url to post the file')
    parser.add_argument('--candidate_types', default='SENTENCE',
                        help='SENTENCE, CURRENCY, DATE, ADDRESS, NUMBER, PERCENT')
    parser.add_argument('--nbest', default=-1, help='url to post the files')

    parser.add_argument('filename')

    args = parser.parse_args()
    if args.verbosity:
        print('verbosity turned on')

    if args.nbest:
        nbest = int(args.nbest)

    if args.provision:
        provision = args.provision
    elif args.custid is not None:
        provision = args.custid

    is_classify_doc = False
    if args.doccat:
        is_classify_doc = True

    is_detect_lang = False
    if args.lang:
        is_detect_lang = True


    if not args.cmd:
        print('usage: postfileutils.py --cmd uploaddir|annotate_doc|annotate_unittest_doc xxx.txt')
        sys.exit(1)

    if args.cmd == 'annotate_unittest_doc':
        result = upload_unittest_annotate_doc(args.filename,
                                              args.provisions.split(','),
                                              is_classify_doc=is_classify_doc,
                                              is_detect_lang=is_detect_lang)
        print(result)
    elif args.cmd == 'annotate_doc':
        result = upload_annotate_doc(args.filename,
                                     args.provisions.split(','),
                                     is_classify_doc=is_classify_doc,
                                     is_detect_lang=is_detect_lang)
        print(result)
    elif args.cmd == 'uploaddir':
        if args.url is not None:
            url_prefix = args.url
        else:
            url_prefix = 'http://127.0.0.1:8000/custom-train'
        upload_train_dir(provision,
                         args.filename,
                         args.candidate_types,
                         nbest=nbest,
                         url_prefix=url_prefix)

# pylint: disable=C0103
if __name__ == '__main__':
    main()
