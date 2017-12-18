#!/usr/bin/env python3

import argparse
from kirke.utils import txtreader, ebsentutils
from kirke.utils.textoffset import TextCpointCunitMapper


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('prov', help='provision')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    provision = args.prov
    fname = args.file

    if not provision or not fname:
        print('usage: printProvAnt.py <prov> <fn_list.txt>')

    fname_list = []
    with open(fname, 'rt') as fin:
        for line in fin:
            fname_list.append(line.strip())

    for fname in fname_list:
        ebdata_fname = fname.replace('.txt', '.ebdata')

        doc_text = txtreader.loads(fname)
        cpoint_cunit_mapper = TextCpointCunitMapper(doc_text)
        tmp_prov_ant_list, is_test = ebsentutils.load_prov_annotation_list(fname,
                                                                           cpoint_cunit_mapper)

        count = 1
        for prov_ant in tmp_prov_ant_list:
            label = prov_ant.label
            start = prov_ant.start
            end = prov_ant.end
            if label == provision:
                print("====={}\t{}\t{}=====".format(count, label, fname))
                print(doc_text[start:end])
                count += 1
        
    print("done.")
