#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import pprint
import sys
# pylint: disable=unused-import
from typing import Any, Dict, List, Optional

import requests

from kirke.utils import antutils, modelfileutils


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
            files = [('file', open(file_name, 'rb')),
                     ('file', open(offset_filename, 'rt', encoding='utf-8'))]
        else:
            files = {'file': open(file_name, 'rb')}  # type: ignore

        resp = requests.post(url, files=files, data=payload)

        if is_show_header:
            print('status: [{}]'.format(resp.status_code), file=sys.stderr)
            print(resp.headers, file=sys.stderr)
        return resp.text

    print("file '{}' is not a valid file".format(file_name), file=sys.stderr)
    raise ValueError


# pylint: disable=too-many-locals
def upload_train_dir(custid: str,
                     *,
                     upload_dir: str,
                     candidate_types: str,
                     nbest: int = -1,
                     url_prefix: str = 'http://127.0.0.1:8000/custom-train') \
                     -> requests.Response:
    txt_fnames = []  # type: List[str]
    for fname in os.listdir(upload_dir):
        fname = '{}/{}'.format(upload_dir, fname)
        if fname.endswith(".txt"):
            txt_fnames.append(fname)

    if not txt_fnames:
        print("cannot find any .txt files", file=sys.stderr)
        raise ValueError

    return upload_train_fname_list(custid,
                                   text_fname_list=txt_fnames,
                                   candidate_types=candidate_types,
                                   nbest=nbest,
                                   url_prefix=url_prefix)


# pylint: disable=too-many-locals
def upload_train_files(custid: str,
                       *,
                       fname_list_fname: str,
                       candidate_types: str,
                       nbest: int = -1,
                       url_prefix: str = 'http://127.0.0.1:8000/custom-train') \
                       -> requests.Response:
    txt_fnames = []  # type: List[str]
    with open(fname_list_fname, 'rt') as fin:
        for line in fin:
            fname = line.strip()
            txt_fnames.append(fname)

    if not txt_fnames:
        print("cannot find any .txt files", file=sys.stderr)
        raise ValueError

    return upload_train_fname_list(custid,
                                   text_fname_list=txt_fnames,
                                   candidate_types=candidate_types,
                                   nbest=nbest,
                                   url_prefix=url_prefix)


def upload_train_fname_list(custid: str,
                            *,
                            text_fname_list: List[str],
                            candidate_types: str,
                            nbest: int = -1,
                            url_prefix: str = 'http://127.0.0.1:8000/custom-train') \
                            -> requests.Response:
    file_tuple_list = []  # type: List
    for txt_fname in text_fname_list:
        ant_fname = txt_fname.replace('.txt', '.ant')
        ebdata_fname = txt_fname.replace('.txt', '.ebdata')
        offsets_fname = txt_fname.replace('.txt', '.offsets.json')
        pdfxml_fname = txt_fname.replace('.txt', '.pdf.xml')

        found_txt, found_ant = False, False
        if os.path.exists(txt_fname):
            file_tuple_list.append(('file', open(txt_fname, 'rt', encoding='utf-8', newline='')))
            print("uploading [{}]".format(txt_fname))
            found_txt = True
        if os.path.exists(ant_fname):
            file_tuple_list.append(('file', open(ant_fname, 'rt', encoding='utf-8')))
            print("uploading [{}]".format(ant_fname))
            found_ant = True
        elif os.path.exists(ebdata_fname):
            file_tuple_list.append(('file', open(ebdata_fname, 'rt', encoding='utf-8')))
            print("uploading [{}]".format(ebdata_fname))
            found_ant = True
        # these later two are optional
        if os.path.exists(offsets_fname):
            print("uploading [{}]".format(offsets_fname))
            file_tuple_list.append(('file', open(offsets_fname, 'rt', encoding='utf-8')))
        if os.path.exists(pdfxml_fname):
            print("uploading [{}]".format(pdfxml_fname))
            file_tuple_list.append(('file', open(pdfxml_fname, 'rt', encoding='utf-8')))

        if not (found_txt and found_ant):
            print("cannot find matching ant file for {}".format(txt_fname), file=sys.stderr)
            raise ValueError

    # print('candidate_types: %s' % (candidate_types, ))
    payload = {'candidate_types': candidate_types,
               'nbest': nbest}  # type: Dict[str, Any]

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
        print(json.dumps(result))
    elif args.cmd == 'uploaddir':
        if args.url is not None:
            url_prefix = args.url
        else:
            url_prefix = 'http://127.0.0.1:8000/custom-train'
        result = upload_train_dir(provision,
                                  upload_dir=args.filename,
                                  candidate_types=args.candidate_types,
                                  nbest=nbest,
                                  url_prefix=url_prefix)
        pprint.pprint(json.loads(result.text))
    elif args.cmd == 'upload_train_files':
        if args.url is not None:
            url_prefix = args.url
        else:
            url_prefix = 'http://127.0.0.1:8000/custom-train'
        result = upload_train_files(provision,
                                    fname_list_fname=args.filename,
                                    candidate_types=args.candidate_types,
                                    nbest=nbest,
                                    url_prefix=url_prefix)
        pprint.pprint(json.loads(result.text))



# pylint: disable=C0103
if __name__ == '__main__':
    main()
