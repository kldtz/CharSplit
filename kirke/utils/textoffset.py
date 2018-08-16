#!/usr/bin/env python3

import argparse
# pylint: disable=unused-import
from typing import Dict, Tuple

# If the offset doesn't change between code point and
# code unit, we don't add them to the mapping table.
# This saves memory.
class TextCpointCunitMapper:

    def __init__(self, text: str) -> None:
        cpoint_bytesize_map = {}  # type: Dict[str, int]
        # pylint: disable=invalid-name
        for ch in text:
            utf16_bytes = ch.encode('utf-16-le')

            if len(utf16_bytes) == 2:
                if not cpoint_bytesize_map.get(ch):
                    cpoint_bytesize_map[ch] = 1

            elif len(utf16_bytes) == 4:
                if not cpoint_bytesize_map.get(ch):
                    cpoint_bytesize_map[ch] = 2

        cpoint_to_cunit_map = {}  # type: Dict[int, int]
        cunit_to_cpoint_map = {}  # type: Dict[int, int]
        cpoint_offset = 0
        cunit_offset = 0
        for cpoint_offset, ch in enumerate(text, 1):
            cunit = cpoint_bytesize_map[ch]
            cunit_offset += cunit

            if cpoint_offset != cunit_offset:
                cpoint_to_cunit_map[cpoint_offset] = cunit_offset
                cunit_to_cpoint_map[cunit_offset] = cpoint_offset

        # for cpoint_offset, cunit_offset in cpoint_to_cunit_map.items():
        #    print("cpoint = {}\tcunit = {}".format(cpoint_offset, cunit_offset))

        self.cpoint_to_cunit_map = cpoint_to_cunit_map
        self.cunit_to_cpoint_map = cunit_to_cpoint_map
        self.max_cunit = cunit_offset
        self.max_cpoint = cpoint_offset

    def to_codepoint_offsets(self, start: int, end: int) -> Tuple[int, int]:
        if start > self.max_cunit:
            out_start = self.max_cunit
        else:
            out_start = self.cunit_to_cpoint_map.get(start, start)

        if end > self.max_cunit:
            out_end = self.max_cunit
        else:
            out_end = self.cunit_to_cpoint_map.get(end, end)
        return out_start, out_end

    def to_codepoint_offset(self, start: int) -> int:
        if start > self.max_cunit:
            out_start = self.max_cunit
        else:
            out_start = self.cunit_to_cpoint_map.get(start, start)
        return out_start

    def to_cunit_offsets(self, start: int, end: int) -> Tuple[int, int]:
        if start > self.max_cpoint:
            out_start = self.max_cpoint
        else:
            out_start = self.cpoint_to_cunit_map.get(start, start)

        if end > self.max_cpoint:
            out_end = self.max_cpoint
        else:
            out_end = self.cpoint_to_cunit_map.get(end, end)
        return out_start, out_end

    def to_cunit_offset(self, start: int) -> int:
        if start > self.max_cpoint:
            out_start = self.max_cpoint
        else:
            out_start = self.cpoint_to_cunit_map.get(start, start)
        return out_start


def main():
    parser = argparse.ArgumentParser(description='normalize address v1')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug",
                        action="store_true",
                        help="print debug information")
    parser.add_argument("-q", "--quote",
                        action="store_true",
                        help="put brackets around the text span")
    parser.add_argument("start", type=int)
    parser.add_argument("end", type=int)
    parser.add_argument("filename")

    args = parser.parse_args()
    if args.verbosity:
        print("verbosity turned on")

    with open(args.filename, "r", newline='') as fin:
        text = fin.read()

    txt_cpoint_cunit_mapper = TextCpointCunitMapper(text)

    cpoint_start, cpoint_end = txt_cpoint_cunit_mapper.to_codepoint_offsets(args.start, args.end)
    print("cpoint_start, cpoint_end = {}, {}".format(cpoint_start, cpoint_end))
    print("[{}]".format(text[cpoint_start:cpoint_end]))


if __name__ == "__main__":
    main()
