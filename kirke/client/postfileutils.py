#!/usr/bin/env python3

import argparse
from pathlib import Path
import os
import sys
# pylint: disable=unused-import
from typing import Any, Dict, List

import requests


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
                   'cust_9']

def post_annotate_document(file_name: str,
                           prov_list: List[str],
                           is_detect_lang: bool = False,
                           is_classify_doc: bool = False,
                           is_show_header: bool = False) -> str:

    url = 'http://localhost:8000/annotate-doc'
    payload = {'types': ','.join(prov_list)}  # type: Dict[str, Any]

    if is_detect_lang:
        payload['detect-lang'] = True
    if is_classify_doc:
        payload['classify-doc'] = True

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


def post_unittest_annotate_document(file_name: str) -> str:
    result = post_annotate_document(file_name,
                                    UNIT_TEST_PROVS,
                                    is_detect_lang=True,
                                    is_classify_doc=True)
    return result


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
