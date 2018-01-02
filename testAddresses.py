#!/usr/bin/env python

import argparse
import logging
import sys

from kirke.ebrules import addresses

def main(args):

    if not args.file:
        print("usage: testAddresses.py l_tenant_notice_train.lines2.txt")
        return 0

    # ignore args.file for now
    # fname = args.file
    fname = 'l_tenant_notice_train.lines2.txt'

    with open(fname, 'rt') as fin:
        for line in fin:
            line = line.strip()
            prob = addresses.classify(line)
            if prob >= 0.5:
                print("addr\t{}\t{}".format(prob, line))
            else:
                print("other\t{}\t{}".format(prob, line))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    main(args)
    
