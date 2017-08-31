
from collections import namedtuple

from kirke.utils import strutils, mathutils, engutils
from kirke.docstruct import jenksutils
from time import time

StrInfo = namedtuple('StrInfo', ['start', 'end',
                                 'xStart', 'xEnd', 'yStart'])

MAX_Y_DIFF = 10000
MIN_X_END = -1



class LineInfo3:

    def __init__(self, start, end, line_num, block_num, strinfo_list):
        self.start = start
        self.end = end
        self.line_num = line_num
        self.block_num = block_num
        self.strinfo_list = strinfo_list

        min_xStart, min_yStart = MAX_Y_DIFF, MAX_Y_DIFF
        max_xEnd = MIN_X_END
        for strinfo in self.strinfo_list:
            syStart = strinfo.yStart
            sxStart = strinfo.xStart
            sxEnd = strinfo.xEnd

            # whichever is lowest in the y-axis of page, use that
            # Not sure what to do when y-axis equal, or very close
            if syStart < min_yStart:
                min_yStart = syStart
                min_xStart = sxStart
            if sxEnd > sxEnd:
                max_xEnd = sxEnd
        self.xStart = min_xStart
        self.xEnd = max_xEnd
        self.yStart = min_yStart

    def tostr2(self):
        return 'se=({}, {}), bnum= {}, xs={:.2f}, ys={:.2f}'.format(self.start,
                                                                    self.end,
                                                                    self.block_num,
                                                                    self.xStart,
                                                                    self.yStart)

class LineWithAttrs:

    def __init__(self, lineinfo, ydiff, linebreak, align, is_english):
        self.lineinfo = lineinfo
        self.ydiff = ydiff
        self.linebreak = linebreak
        self.align = align  # inferred
        self.is_english = is_english

    def tostr2(self):
        return '{}, lbk={:.1f}, algn={}, en={}'.format(self.lineinfo.tostr2(), self.linebreak, self.align, self.is_english)


class PageInfo3:

    def __init__(self, doc_text, start, end, page_num, pblockinfo_list):
        self.start = start
        self.end = end
        self.page_num = page_num
        self.pblockinfo_list = pblockinfo_list
        self.avg_single_line_break_ydiff = self.compute_avg_single_line_break_ydiff()

        # self.line_list = init_line_with_attr_list()

        lineinfo_list = [lineinfo
                         for pblockinfo in self.pblockinfo_list
                         for lineinfo in pblockinfo.lineinfo_list]
        xstart_list = [lineinfo.xStart for lineinfo in lineinfo_list]
        if len(xstart_list) == 1:
            xstart_list.append(xstart_list[0])  # duplicate itself to avoid jenks error with only element
        jenks = jenksutils.Jenks(xstart_list)

        line_attrs = []
        prev_yStart = 0
        for lineinfo in lineinfo_list:
          ydiff = lineinfo.yStart - prev_yStart
          num_linebreak = round(ydiff / self.avg_single_line_break_ydiff, 1)
          align = jenks.classify(lineinfo.xStart)
          line_text = doc_text[lineinfo.start:lineinfo.end]
          is_english = engutils.classify_english_sentence(line_text)
          line_attrs.append(LineWithAttrs(lineinfo, ydiff, num_linebreak, align, is_english))
          prev_yStart = lineinfo.yStart
        self.line_list = line_attrs


    def compute_avg_single_line_break_ydiff(self):
        total_merged_ydiff, total_merged_lines = 0, 0
        for pblockinfo in self.pblockinfo_list:
            is_multi_lines = pblockinfo.is_multi_lines
            if not (is_multi_lines or len(pblockinfo.lineinfo_list) == 1):
                total_merged_ydiff += pblockinfo.yEnd - pblockinfo.yStart
                total_merged_lines += len(pblockinfo.lineinfo_list) - 1

        result = 14.0  # default value, just in case
        if total_merged_lines != 0:
            result = total_merged_ydiff / total_merged_lines
        # print("\npage #{}, avg_single_line_ydiff = {}".format(self.page_num, result))
        return result

        
class Line4Nlp:

    def __init__(self, orig_start, orig_end, nlp_start, nlp_end, line_num, xStart, xEnd, yStart, yEnd):
        self.orig_start = orig_start
        self.orig_end = orig_end
        self.nlp_start = nlp_start
        self.nlp_end = nlp_end        
        self.line_num = line_num
        self.xStart = xStart
        self.xEnd = xEnd
        self.yStart = yStart
        self.yEnd = yEnd
        self.ydiff = MAX_Y_DIFF
        self.linebreak = 1000
        # there is no more eoln
        # self.is_multi_lines = is_multi_lines

    def tostr2(self):
        return 'ostart={}, oend={}, nlp_start={}, nlp_end={}'.format(self.orig_start,
                                                                     self.orig_end,
                                                                     self.nlp_start,
                                                                     self.nlp_end)

class PageInfo:

    def __init__(self, start, end, page_num, pblockinfo_list):
        self.start = start
        self.end = end
        self.page_num = page_num
        self.pblockinfo_list = pblockinfo_list
        self.line4nlp_list = []
        self.avg_single_line_break_ydiff = 12.0

    def init_line4nlp_list(self, nlp_offset):
        line4nlp_list = self.line4nlp_list
        line_num = 0

        # for computing page-based single-line-break-ydiff
        total_merged_lines = 0
        total_merged_ydiff = 0        

        for pblockinfo in self.pblockinfo_list:
            is_multi_lines = pblockinfo.is_multi_lines
            # print("len_lineinfo_list = {}".format(len(pblockinfo.lineinfo_list)))

            if is_multi_lines or len(pblockinfo.lineinfo_list) == 1:
                for lineinfo in pblockinfo.lineinfo_list:
                    line_len = (lineinfo.end - lineinfo.start)
                    lx4nlp = Line4Nlp(lineinfo.start,
                                      lineinfo.end,
                                      nlp_offset,
                                      nlp_offset + line_len,
                                      line_num,
                                      lineinfo.xStart,
                                      lineinfo.xEnd,
                                      lineinfo.yStart,
                                      lineinfo.yStart)
                    line4nlp_list.append(lx4nlp) 
                    nlp_offset += line_len + 2  # for 2 eoln
                    # print('nlp_offset = {}'.format(nlp_offset))
                    line_num += 1
            else:
                line_len = len(pblockinfo.text)
                lx4nlp = Line4Nlp(pblockinfo.start,
                                  pblockinfo.end,
                                  nlp_offset,
                                  nlp_offset + line_len,
                                  line_num,
                                  pblockinfo.xStart,
                                  pblockinfo.xEnd,
                                  pblockinfo.yStart,
                                  pblockinfo.yEnd)

                total_merged_ydiff += pblockinfo.yEnd - pblockinfo.yStart
                total_merged_lines += len(pblockinfo.lineinfo_list) - 1
                # print("pblock linediff-avg = {}, ydiff= {}, nline = {}".format((pblockinfo.yEnd - pblockinfo.yStart) / (len(pblockinfo.lineinfo_list) - 1),
                #                                                               (pblockinfo.yEnd - pblockinfo.yStart),
                # len(pblockinfo.lineinfo_list)))
                
                line4nlp_list.append(lx4nlp) 
                nlp_offset += line_len + 2  # for 2 eoln
                # print('nlp_offset = {}'.format(nlp_offset))                
                line_num += 1
        weird_ydiff_factor = 1.1   # 25 / 11.5 / 2 = 1.08
        if total_merged_lines != 0:
            self.avg_single_line_break_ydiff = total_merged_ydiff / total_merged_lines * weird_ydiff_factor
        # print("page #{}, avg_single_line_ydiff = {}".format(self.page_num, self.avg_single_line_break_ydiff))

        # now compute the ydiff
        if not line4nlp_list:
            return nlp_offset
        for line4nlp in line4nlp_list:
            prev_yEnd = line4nlp_list[0].yEnd
            for line4nlp in line4nlp_list[1:]:
                cur_yStart = line4nlp.yStart
                line4nlp.ydiff = cur_yStart - prev_yEnd
                line4nlp.linebreak = int(round(line4nlp.ydiff / self.avg_single_line_break_ydiff))
                # prev_line4nlp = line4nlp
                prev_yEnd = line4nlp.yEnd

        # to continue to next page
        return nlp_offset
