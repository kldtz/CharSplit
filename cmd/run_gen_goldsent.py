#!/usr/bin/env python3

import argparse
from collections import defaultdict
import logging
import os

from kirke.utils import ebantdoc4, osutils, strutils

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt',
    # help='input directory for .txt files')
    parser.add_argument('file', help='input file')

    args = parser.parse_args()

    fname = args.file

    work_dir = 'dir-work'
    osutils.mkpath(work_dir)
    ebantdoc4.clear_cache(fname, work_dir)
    ebdoc = ebantdoc4.text_to_ebantdoc4(fname, work_dir)

    print('loaded %s' % ebdoc.file_id)

    """
    para_out_fname = ebdoc.file_id.replace('.txt', '.paragraph.tsv')
    ebantdoc4.save_para_text(ebdoc,
                             para_out_fname)    
    logger.info('wrote %s', para_out_fname)
    """

    base_fname = os.path.basename(fname)
    work_fn = os.path.join(work_dir, base_fname)

    lines_out_fname = work_fn.replace('.txt', '.lines.txt')
    nl_text = ebdoc.get_nl_text()
    strutils.dumps(nl_text, lines_out_fname)
    print('wrote %s' % lines_out_fname)

    paraline_out_fname = work_fn.replace('.txt', '.paraline.txt')
    paraline_text = ebdoc.get_paraline_text()
    strutils.dumps(paraline_text, paraline_out_fname)
    print('wrote %s' % paraline_out_fname)    

    nlp_out_fname = work_fn.replace('.txt', '.nlp.txt')
    nlp_text = ebdoc.get_nlp_text()
    strutils.dumps(nlp_text, nlp_out_fname)
    print('wrote %s' % nlp_out_fname)
    
    sent_out_fname = work_fn.replace('.txt', '.sent.txt')
    ebantdoc4.save_sent_se_text(ebdoc,
                                sent_out_fname)
    print('wrote %s' % sent_out_fname)
    

    """
    prov_count_map = defaultdict(int)
    for prov_ant in ebdoc.prov_annotation_list:
        print("prov_ant: {}".format(prov_ant))
        prov_count_map[prov_ant.label] += 1

    for prov, count in prov_count_map.items():
        print("prov_count[{}] = {}".format(prov, count))


    for attrvec in ebdoc.attrvec_list:
        print("attrvec: {}".format(attrvec))
    """
    
    
if __name__ == '__main__':
    main()

