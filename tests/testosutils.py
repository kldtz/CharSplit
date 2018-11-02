#!/usr/bin/env python3

import unittest

from kirke.utils import osutils


class TestOsUtils(unittest.TestCase):

    def test_split_docid_md5(self):

        md5_docid_fname = 'e6404884e22d247d9b5423fe047b193d-8421.txt'
        docid_md5_fname = '8421-e6404884e22d247d9b5423fe047b193d.txt'

        self.assertEqual(osutils.split_docid_md5(md5_docid_fname),
                         ('8421', 'e6404884e22d247d9b5423fe047b193d', '.txt'))
        self.assertEqual(osutils.split_docid_md5(docid_md5_fname),
                         ('8421', 'e6404884e22d247d9b5423fe047b193d', '.txt'))


    def test_get_docid(self):
        md5_docid_fname = 'e6404884e22d247d9b5423fe047b193d-8421.txt'
        docid_md5_fname = '8421-e6404884e22d247d9b5423fe047b193d.txt'

        md5_docid_full_name = '/home/jshaw/tmp/e6404884e22d247d9b5423fe047b193d-8421.txt'
        docid_md5_full_name = '/home/jshaw/tmp/8421-e6404884e22d247d9b5423fe047b193d.txt'

        other_fname = '7d9b5423fe047b193d.txt'
        other_full_name = '/home/jshaw/tmp/7d9b5423fe047b193d.txt'

        self.assertEqual(osutils.get_docid(md5_docid_fname), '8421')
        self.assertEqual(osutils.get_docid(docid_md5_fname), '8421')

        self.assertEqual(osutils.get_docid(md5_docid_full_name), '8421')
        self.assertEqual(osutils.get_docid(docid_md5_full_name), '8421')

        self.assertEqual(osutils.get_docid(other_fname), None)
        self.assertEqual(osutils.get_docid(other_full_name), None)


    def test_get_docid_or_basename_prefix(self):
        md5_docid_fname = 'e6404884e22d247d9b5423fe047b193d-8421.txt'
        docid_md5_fname = '8421-e6404884e22d247d9b5423fe047b193d.txt'

        md5_docid_full_name = '/home/jshaw/tmp/e6404884e22d247d9b5423fe047b193d-8421.txt'
        docid_md5_full_name = '/home/jshaw/tmp/8421-e6404884e22d247d9b5423fe047b193d.txt'

        self.assertEqual(osutils.get_docid_or_basename_prefix(md5_docid_fname), '8421')
        self.assertEqual(osutils.get_docid_or_basename_prefix(docid_md5_fname), '8421')

        self.assertEqual(osutils.get_docid_or_basename_prefix(md5_docid_full_name), '8421')
        self.assertEqual(osutils.get_docid_or_basename_prefix(docid_md5_full_name), '8421')

        other_fname = '7d9b5423fe047b193d.txt'
        other_full_name = '/home/jshaw/tmp/7d9b5423fe047b193d.txt'

        self.assertEqual(osutils.get_docid_or_basename_prefix(other_fname), '7d9b5423fe047b193d')
        self.assertEqual(osutils.get_docid_or_basename_prefix(other_full_name), '7d9b5423fe047b193d')


    def test_knorm_base_file_name(self):

        md5_docid_fname = 'e6404884e22d247d9b5423fe047b193d-8421.txt'
        docid_md5_fname = '8421-e6404884e22d247d9b5423fe047b193d.txt'

        self.assertEqual(osutils.get_knorm_base_file_name(md5_docid_fname),
                         docid_md5_fname)
        self.assertEqual(osutils.get_knorm_base_file_name(docid_md5_fname),
                         docid_md5_fname)

    def test_knorm_base_file_name(self):

        md5_docid_fname = 'e6404884e22d247d9b5423fe047b193d-8421.txt'
        docid_md5_fname = '8421-e6404884e22d247d9b5423fe047b193d.txt'

        md5_docid_full_name = '/home/jshaw/tmp/e6404884e22d247d9b5423fe047b193d-8421.txt'
        docid_md5_full_name = '/home/jshaw/tmp/8421-e6404884e22d247d9b5423fe047b193d.txt'

        self.assertEqual(osutils.get_knorm_file_name(md5_docid_fname),
                         docid_md5_fname)
        self.assertEqual(osutils.get_knorm_file_name(docid_md5_fname),
                         docid_md5_fname)

        self.assertEqual(osutils.get_knorm_file_name(md5_docid_full_name),
                         docid_md5_full_name)
        self.assertEqual(osutils.get_knorm_file_name(docid_md5_full_name),
                         docid_md5_full_name)


if __name__ == "__main__":
    unittest.main()
