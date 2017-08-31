from collections import namedtuple, Counter
import copy
from functools import total_ordering
import re
import json
from typing import List

from kirke.docstruct import tokenizer
from kirke.utils import strutils, engutils, stopwordutils

MAX_Y_DIFF = 10000
MIN_X_END = -1

@total_ordering
class PBlockInfo:

    def __init__(self,
                 start,
                 end,
                 bid,
                 pagenum,
                 text,
                 lineinfo_list,
                 is_multi_lines):
        self.start = start
        self.end = end
        self.bid = bid
        self.pagenum = pagenum
        self.text = text
        self.length = len(text)
        self.lineinfo_list = lineinfo_list
        self.is_multi_lines = is_multi_lines

        pb_min_xStart, pb_min_yStart = MAX_Y_DIFF, MAX_Y_DIFF
        pb_max_xEnd, pb_max_yEnd = MIN_X_END, MIN_X_END

        for lineinfo in self.lineinfo_list:
            strinfo_list = lineinfo.strinfo_list
            lx_min_xStart, lx_min_yStart = MAX_Y_DIFF, MAX_Y_DIFF
            lx_max_xEnd = MIN_X_END

            for strinfo in strinfo_list:
                sxStart = strinfo.xStart
                sxEnd = strinfo.xEnd
                syStart = strinfo.yStart

                # whichever is lowest in the y-axis of page, use that
                # Not sure what to do when y-axis equal, or very close
                if syStart < lx_min_yStart:
                    lx_min_yStart = syStart
                    lx_min_xStart = sxStart
                #print("block_id = {}, is_multi_lines = {}, len(lxinfo_list)= {}, sxEnd = {}, lx_max_xEnd = {}".format(
                #      bid, is_multi_lines, len(lineinfo_list), sxEnd, lx_max_xEnd))
                if sxEnd > lx_max_xEnd:
                    lx_max_xEnd = sxEnd

                # whichever is lowest in the y-axis of page, use that
                # Not sure what to do when y-axis equal, or very close
                if syStart < pb_min_yStart:
                    pb_min_yStart = syStart
                    pb_min_xStart = sxStart
                #print("block_id = {}, is_multi_lines = {}, len(lxinfo_list)= {}, sxEnd = {}, pb_max_xEnd = {}".format(
                #      bid, is_multi_lines, len(lineinfo_list), sxEnd, pb_max_xEnd))
                # if (len(lineinfo_list) == 1 or is_multi_lines) and sxEnd > pb_max_xEnd:
                if sxEnd > pb_max_xEnd:
                    pb_max_xEnd = sxEnd
                if syStart > pb_max_yEnd:
                    pb_max_yEnd = syStart

            lineinfo.xStart = lx_min_xStart
            lineinfo.xEnd = lx_max_xEnd
            lineinfo.yStart = lx_min_yStart

        self.xStart = pb_min_xStart
        self.xEnd = pb_max_xEnd
        self.yStart = pb_min_yStart
        self.yEnd = pb_max_yEnd
        # print("self.xStart = {}, xEnd = {}, yStart= {}".format(self.xStart, self.xEnd, self.yStart))
        self.is_english = engutils.classify_english_sentence(text)
        self.ydiff = MAX_Y_DIFF  # will compute this across PBlockInfo

    def __eq__(self, other):
        #if hasattr(other, 'start') and hasattr(other, 'end'):
        return (self.start, self.end).__eq__((other.start, other.end))

    def __lt__(self, other):
        #if hasattr(other, 'start') and hasattr(other, 'end'):
        return (self.start, self.end).__lt__((other.start, other.end))

    def __hash__(self):
        return hash((self.start, self.end))

    def __repr__(self):
        # self.words is not printed, intentionally
        return str(('start={}'.format(self.start),
                    'end={}'.format(self.end),
                    'bid={}'.format(self.bid),
                    'xStart={:.2f}'.format(self.xStart),
                    'yStart={:.2f}'.format(self.yStart),
                    'pagenum={}'.format(self.pagenum),
                    'english={}'.format(self.is_english)))


    def tostr2(self, text):
        not_en_mark = ''
        if self.is_english:
            tags = ['EN']
        else:
            tags = ['NOT_EN']
            not_en_mark = '>>>'
        if self.category:
            tags.append('CAT-{}'.format(self.category))
        if self.is_close_prev_line:
            tags.append("CLSE")
        else:
            tags.append("FAR")
        if self.align_label:
            tags.append(self.align_label)
        tags.append('LN-{}'.format(self.length))
        # tags.append('PG-{}'.format(self.page))

        tags_st = ' '.join(tags)

        # self.words is not printed, intentionally
        return '%-25s\t%s\t%s' % (tags_st, not_en_mark, text[self.start:self.end])

    
    def is_center(self):
        return self.align_label == 'CN'

    def is_LF(self):
        return self.align_label and self.algign_label.startswith('LF')

    def is_LF1(self):
        return self.align_label == 'LF1'

    def is_LF2(self):
        return self.align_label == 'LF2'

    def is_LF3(self):
        return self.align_label == 'LF3'

