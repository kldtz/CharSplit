#!/usr/bin/env python3

import argparse
from typing import List

from kirke.utils import ebsentutils, strutils, txtreader
from kirke.utils.textoffset import TextCpointCunitMapper


def print_doc_prov_ant(fname: str, provision: str) -> List[str]:
    doc_text = txtreader.loads(fname)
    cpoint_cunit_mapper = TextCpointCunitMapper(doc_text)
    tmp_prov_ant_list, unused_is_test = \
        ebsentutils.load_prov_annotation_list(fname,
                                              cpoint_cunit_mapper)
    out_list = []  # type: List[str]
    for prov_ant in tmp_prov_ant_list:
        label = prov_ant.label
        start = prov_ant.start
        end = prov_ant.end
        if label == provision:
            # print("====={}\t{}\t{}=====".format(count, label, fname))
            st_list = []
            st_list.append(label)
            st_list.append(fname)
            st_list.append(strutils.sub_nltab_with_space(doc_text[start:end]))
            # print('\t'.join(st_list))
            out_list.append('\t'.join(st_list))

    return out_list


def main() -> None:
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('prov', help='provision')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    PROVISION = args.prov

    if not PROVISION or not args.file:
        print('usage: printProvAnt.py <prov> <fn_list.txt>')

    fname_list = []
    with open(args.file, 'rt') as fin:
        for line in fin:
            fname_list.append(line.strip())


    count = 0
    for fname in fname_list:
        # ebdata_fname = fname.replace('.txt', '.ebdata')

        st_list = print_doc_prov_ant(fname, PROVISION)

        for st in st_list:
            out_list = [str(count)]
            count += 1
            out_list.append(st)
            print('\t'.join(out_list))


if __name__ == '__main__':
    main()
