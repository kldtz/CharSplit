#!/usr/bin/env python3

import argparse
import logging
import operator

from kirke.utils import ebantdoc4
from kirke.utils import memutils

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    parser.add_argument('file')

    args = parser.parse_args()

    txt_file_name = args.file
    work_dir = 'dir-work'
    eb_antdoc = ebantdoc4.text_to_ebantdoc4(txt_file_name, work_dir)

    # ebantdoc4.print_attrvec_list(eb_antdoc)
    # ebantdoc4.print_para_list(eb_antdoc)

    print("memory_size(eb_antdoc) = %d bytes" % (memutils.get_size(eb_antdoc),))
    print("memory_size(eb_antdoc) = %.2f kilobytes" % (memutils.get_size_kbytes(eb_antdoc),))
    print("memory_size(eb_antdoc) = %.2f megabytes" % (memutils.get_size_mbytes(eb_antdoc),))

    print(vars(eb_antdoc))

    ebantdoc_size = memutils.get_size(eb_antdoc)

    val_sz_acc = 0
    attr_size_list = []
    for attr, value in eb_antdoc.__dict__.items():
        val_sz = memutils.get_size(value)
        attr_size_list.append((attr, val_sz))

    for attr, val_sz in sorted(attr_size_list, key=operator.itemgetter(1), reverse=True):
        val_sz_acc += val_sz
        print("{}\tsize\t{}\t{:.5f}%\tacc\t{}\t{:.3f}%".format(attr,
                                                               val_sz,
                                                               val_sz * 100.0 / ebantdoc_size,
                                                               val_sz_acc,
                                                               val_sz_acc * 100.0 / ebantdoc_size))


if __name__ == '__main__':
    main()
