#!/usr/bin/env python

import argparse
import time
from collections import defaultdict 

# usage: writeUTF16Offset.py 16 22 short2.txt

# to reduce the size, can make it
# if the same, do nothing, don't insert
class TextCptCunitMapper:

    def __init__(self, text:str):
        cpt_bytesize_map = {}
        for ch in text:
            utf16_bytes = ch.encode('utf-16-le')
        
            if len(utf16_bytes) == 2:
                if not cpt_bytesize_map.get(ch):
                    cpt_bytesize_map[ch] = 1

            elif len(utf16_bytes) == 4:
                if not cpt_bytesize_map.get(ch):
                    cpt_bytesize_map[ch] = 2

        cpt_to_cunit_map = {}
        cunit_to_cpt_map = {}
        cpt_offset = 0
        cunit_offset = 0
        for cpt_offset, ch in enumerate(text, 1):
            cunit = cpt_bytesize_map[ch]
            cunit_offset += cunit

            if cpt_offset != cunit_offset:
                cpt_to_cunit_map[cpt_offset] = cunit_offset
                cunit_to_cpt_map[cunit_offset] = cpt_offset

        # for cpt_offset, cunit_offset in cpt_to_cunit_map.items():
        #    print("cpt = {}\tcunit = {}".format(cpt_offset, cunit_offset))

        self.cpt_to_cunit_map = cpt_to_cunit_map
        self.cunit_to_cpt_map = cunit_to_cpt_map
        self.max_cunit = cunit_offset
        self.max_cpt = cpt_offset
        
    def to_codepoint_offsets(self, start, end):
        if start > self.max_cunit:
            out_start = self.max_cunit
        else:
            out_start = self.cunit_to_cpt_map.get(start, start)
            
        if end > self.max_cunit:
            out_end = self.max_cunit
        else:
            out_end = self.cunit_to_cpt_map.get(end, end)
        return out_start, out_end

    def to_codepoint_offset(self, start):
        if start > self.max_cunit:
            out_start = self.max_cunit
        else:
            out_start = self.cunit_to_cpt_map.get(start, start)
        return out_start

    def to_cunit_offsets(self, start, end):
        if start > self.max_cpt:
            out_start = self.max_cpt
        else:
            out_start = self.cpt_to_cunit_map.get(start, start)
            
        if end > self.max_cpt:
            out_end = self.max_cpt
        else:
            out_end = self.cpt_to_cunit_map.get(end, end)
        return out_start, out_end

    def to_cunit_offset(self, start):
        if start > self.max_cpt:
            out_start = self.max_cpt
        else:
            out_start = self.cpt_to_cunit_map.get(start, start)
        return out_start
    
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='normalize address v1')
    parser.add_argument("-v","--verbosity", help="increase output verbosity")
    parser.add_argument("-d","--debug", action="store_true", help="print debug information")
    parser.add_argument("-q","--quote", action="store_true", help="put brackets around the text span")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("filename")

    args = parser.parse_args()
    if args.verbosity:
        print("verbosity turned on")
    if args.debug:
        isDebug= True

    with open(args.filename, "r", newline= '') as fin:
        text = fin.read()

    txt_cpt_cunit_mapper = TextCptCunitMapper(text)

    cpt_start, cpt_end = txt_cpt_cunit_mapper.to_codepoint_offsets(args.start, args.end)
    print("cpt_start, cpt_end = {}, {}".format(cpt_start, cpt_end))
    print("[{}]".format(text[cpt_start:cpt_end]))


"""
𝑃𝑃𝑃𝑃𝑃 𝑡𝑡 𝐶𝑆𝐶
"""
