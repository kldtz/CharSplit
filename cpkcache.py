#!/usr/bin/env python3

import argparse
import os
import shutil

# This program is used to copy .txt and associated files from extractor's kirke
# cache directory to another directory, but with better names (no md5 hash).

def log_line(line: str) -> None:
    with open('cpkcache.log', 'at') as fout:
        fout.write(line + '\n')


# pylint: disable=too-many-locals
def main():
    parser = argparse.ArgumentParser(description='Copy hashed Kirke input txt file to a directory.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    parser.add_argument('file', help='input file')
    parser.add_argument('dir', help='output dirrectory')


    args = parser.parse_args()
    txt_fname = args.file
    txt_bname = os.path.basename(txt_fname)
    adir = os.path.dirname(txt_fname)

    if not txt_bname.endswith('.txt'):
        print('error: input file must ends with .txt: "{}"'.format(txt_fname))
        exit(1)

    offset_bname = txt_bname.replace('.txt', '.offsets.json')
    xml_bname = txt_bname.replace('.txt', '.pdf.xml')

    offset_fname = '{}/{}'.format(adir, offset_bname)
    xml_fname = '{}/{}'.format(adir, xml_bname)

    txt_id_bname = txt_bname.split('-')[1]
    offset_id_bname = txt_id_bname.replace('.txt', '.offsets.json')
    xml_id_bname = txt_id_bname.replace('.txt', '.pdf.xml')


    out_dir = args.dir
    out_txt_fname = '{}/{}'.format(out_dir, txt_id_bname)
    out_offset_fname = '{}/{}'.format(out_dir, offset_id_bname)
    out_xml_fname = '{}/{}'.format(out_dir, xml_id_bname)


    shutil.copy2(txt_fname, out_txt_fname)
    print('copied {} -> {}'.format(txt_fname, out_txt_fname))
    shutil.copy2(offset_fname, out_offset_fname)
    print('copied {} -> {}'.format(offset_fname, out_offset_fname))
    shutil.copy2(xml_fname, out_xml_fname)
    print('copied {} -> {}'.format(xml_fname, out_xml_fname))

    log_line('copied {} -> {}'.format(txt_fname, out_txt_fname))


if __name__ == '__main__':
    main()
