#!/usr/bin/env python3

import json
import unittest
from typing import Any, Dict

from kirke.client import postfileutils
from kirke.utils import antdocutils


MODEL_DIR = 'dir-scut-model'
WORK_DIR = 'dir-work'
CUSTOM_MODEL_DIR = 'dir-custom-model'

def upload_annotate_doc(file_name: str) -> Dict[str, Any]:

    text = postfileutils.post_unittest_annotate_document(file_name)

    ajson = json.loads(text)

    return ajson


def get_antdoc_lang(file_name) -> str:
    ajson = upload_annotate_doc(file_name)

    lang = antdocutils.get_ant_out_json_lang(ajson)
    return lang


class TestAntLang(unittest.TestCase):

    def test_antdoc_lang(self):
        lang = get_antdoc_lang('demo-txt/8285.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8286.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8287.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8288.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8289.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8290.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8291.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8292.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8293.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8294.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8295.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8296.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8297.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8298.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8299.txt')
        self.assertEqual(lang, 'en')

        lang = get_antdoc_lang('demo-txt/8300.txt')
        self.assertEqual(lang, 'en')


if __name__ == "__main__":
    unittest.main()
