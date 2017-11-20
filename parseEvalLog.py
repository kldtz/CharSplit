#!/usr/bin/env python3

import sys
import argparse
import json
import textwrap

from collections import defaultdict, namedtuple

EvalInstance = namedtuple('EvalInstance', ['score', 'text', 'type', 'file_id', 'provision'])


def load_prov_eval_log(file_name: str):
    etype_inst_list_map = defaultdict(list)
    provision = file_name.split("-")[0]

    with open(file_name, "r") as handle:
        parsed = json.load(handle)

    for file_id, adict in parsed.items():
        for etype, alist in adict.items():
            if alist:
                # print("alist: {}".format(alist))
                for inst_json in alist:
                    text = inst_json[0]
                    score = inst_json[1]
                    # print("score = {}, text = [{}]".format(score, text))
                    etype_inst_list_map[etype].append(EvalInstance(score, text, etype, provision, file_id))

    return etype_inst_list_map
    
    
def print_etype_inst_list(etype, alist):
    if etype in ['tp', 'fp']:
        sorted_list = sorted(alist, reverse=True)
    else:
        sorted_list = sorted(alist)
    for count, inst in enumerate(sorted_list):
        print("\n    {} #{}, score={}, prov= {}, file_id={}".format(etype.upper(), count, inst.score, inst.provision, inst.file_id))
        lines = textwrap.wrap(inst.text)
        for line in lines:
            print("        {}".format(line))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='normalize address v1')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    parser.add_argument("filename")

    args = parser.parse_args()
    if args.verbosity:
        print("verbosity turned on")
    if args.debug:
        isDebug= True

    etype_inst_list_map = load_prov_eval_log(args.filename)

    print("=====tp instances: (len = {})".format(len(etype_inst_list_map['tp'])))
    print_etype_inst_list('tp', etype_inst_list_map['tp'])

    print("\n===== fp instances: (len = {})".format(len(etype_inst_list_map['fp'])))
    print_etype_inst_list('fp', etype_inst_list_map['fp'])

    print("\n===== fn instances: (len = {})".format(len(etype_inst_list_map['fn'])))
    print_etype_inst_list('fn', etype_inst_list_map['fn'])    
