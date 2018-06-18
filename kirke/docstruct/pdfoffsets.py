from collections import namedtuple, Counter, defaultdict
from functools import total_ordering
import os
import sys
from typing import Any, DefaultDict, Dict, List, Tuple

from kirke.docstruct import jenksutils, docstructutils
from kirke.utils import engutils, strutils


StrInfo = namedtuple('StrInfo', ['start', 'end',
                                 'xStart', 'xEnd', 'yStart'])

MAX_Y_DIFF = 10000
MIN_X_END = -1


class PageInfo3:

    def __init__(self, doc_text, start, end, page_num, pblockinfo_list) -> None:
        self.start = start
        self.end = end
        self.page_num = page_num

        # need to order the blocks by their yStart first.
        # this impact the line list also.
        # Fixes header and footer issues due to out of order lines.
        # Also out of order blocks due to tables and header.  p76 carousel.txt
        self.pblockinfo_list = sorted(pblockinfo_list, key=lambda x: x.start)
        self.avg_single_line_break_ydiff = self.compute_avg_single_line_break_ydiff()

        # self.line_list = init_line_with_attr_list()

        lineinfo_list = [lineinfo
                         for pblockinfo in self.pblockinfo_list
                         for lineinfo in pblockinfo.lineinfo_list]
        xstart_list = [lineinfo.xStart for lineinfo in lineinfo_list]

        # print("leninfo_list = {}, xstart_list = {}, page_num = {}".format(len(lineinfo_list),
        # xstart_list, page_num))
        if lineinfo_list:  # not an empty page
            if len(xstart_list) == 1:
                xstart_list.append(xstart_list[0])  # duplicate itself to avoid jenks error with only element
            jenks = jenksutils.Jenks(xstart_list)

        line_attrs = []
        prev_yStart = 0
        for page_line_num, lineinfo in enumerate(lineinfo_list, 1):
          ydiff = lineinfo.yStart - prev_yStart
          # it possible for self.avg_single_line_break_ydiff to be 0 when
          # the document has only vertical lines.
          if self.avg_single_line_break_ydiff == 0:
              num_linebreak = 1  # hopeless, default to 1 for now
          else:
              num_linebreak = round(ydiff / self.avg_single_line_break_ydiff, 1)
          align = jenks.classify(lineinfo.xStart)
          line_text = doc_text[lineinfo.start:lineinfo.end]
          is_english = engutils.classify_english_sentence(line_text)
          is_centered = docstructutils.is_line_centered(line_text, lineinfo.xStart, lineinfo.xEnd)
          line_attrs.append(LineWithAttrs(page_line_num,
                                          lineinfo, line_text, page_num, ydiff, num_linebreak,
                                          align, is_centered, is_english))
          prev_yStart = lineinfo.yStart
        self.line_list = line_attrs
        # attrs of page, such as 'page_num_index'
        self.attrs = {}  # type: Dict[str, Any]

        # conent_line_list is for lines that are not
        #   - toc
        #   - page_num
        #   - header, footer
        self.content_line_list = []  # type: List[LineWithAttrs]


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

    def get_blocked_lines(self):
        if not self.line_list:
            return []
        prev_block_num = -1
        cur_block = []
        block_list = [cur_block]
        for linex in self.line_list:
            if linex.lineinfo.obid != prev_block_num:  # separate blocks
                if cur_block:
                    cur_block = []
                    block_list.append(cur_block)
            cur_block.append(linex)
            prev_block_num = linex.lineinfo.obid
        return block_list



class PDFTextDoc:

    def __init__(self,
                 file_name: str,
                 doc_text: str,
                 page_list: List[PageInfo3]) -> None:
        self.file_name = file_name
        self.doc_text = doc_text
        self.page_list = page_list
        self.num_pages = len(page_list)
        # each page is a list of grouped_block
        self.paged_grouped_block_list = []  # type: List[List[GroupedBlockInfo]]
        self.special_blocks_map = defaultdict(list)  # type: DefaultDict[str, List[Tuple[int, int, Dict[str, Any]]]]

    def get_page_offsets(self) -> List[Tuple[int, int]]:
        return [(page.start, page.end) for page in self.page_list]

    def print_debug_blocks(self):
        for page_num, grouped_block_list in enumerate(self.paged_grouped_block_list, 1):
            print('\n===== page #%d, len(block_list)= %d' % (page_num, len(grouped_block_list)))

            apage = self.page_list[page_num - 1]
            if apage.attrs:
                print('  attrs: {}'.format(', '.join(strutils.dict_to_sorted_list(apage.attrs))))
            """
            attr_list = []
            for attr, value in sorted(apage.attrs.items()):
                attr_list.append('{}={}'.format(attr, value))
            if attr_list:
                print('  attrs: {}'.format(', '.join(attr_list)))
            """

            for grouped_block in grouped_block_list:
                print()
                for linex in grouped_block.line_list:
                    print('{}\t{}'.format(linex.tostr3(), self.doc_text[linex.lineinfo.start:linex.lineinfo.end]))
        for block_type, span_list in sorted(self.special_blocks_map.items()):
            print("\nblock_type: {}".format(block_type))
            for start, end, adict in span_list:
                alist = [(attr, value) for attr, value in sorted(adict.items())]
                print("\t{}\t{}\t{}\t[{}...]".format(start, end, alist, self.doc_text[start:end][:15].replace('\n', ' ')))

    def save_debug_pages(self, extension: str, work_dir='dir-work'):
        base_fname = os.path.basename(self.file_name)
        paged_debug_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))
        with open(paged_debug_fname, 'wt') as fout:
            for page in self.page_list:
                print('\n===== page #%d, start=%d, end=%d, len(lines)= %d' %
                      (page.page_num, page.start, page.end, len(page.line_list)), file=fout)

                if page.attrs:
                    print('  attrs: {}'.format(', '.join(strutils.dict_to_sorted_list(page.attrs))), file=fout)

                grouped_block_list = line_list_to_grouped_block_list(page.line_list, page.page_num)
                for grouped_block in grouped_block_list:
                    print(file=fout)
                    for linex in grouped_block.line_list:
                        print('{}\t{}'.format(linex.tostr5(), self.doc_text[linex.lineinfo.start:linex.lineinfo.end]), file=fout)

        print('wrote {}'.format(paged_debug_fname), file=sys.stderr)


    def save_raw_pages(self,
                       extension: str,
                       work_dir: str = 'dir-work'):
        base_fname = os.path.basename(self.file_name)
        paged_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))
        with open(paged_fname, 'wt') as fout:
            for page in self.page_list:
                print('\n===== page #%d, start=%d, end=%d, len(lines)= %d' %
                      (page.page_num, page.start, page.end, len(page.line_list)), file=fout)
                if page.attrs:
                    print('  attrs: {}'.format(', '.join(strutils.dict_to_sorted_list(page.attrs))), file=fout)

                prev_block_num = -1
                for linex in page.line_list:
                    if linex.lineinfo.obid != prev_block_num:  # separate blocks
                        print(file=fout)
                    print('{}\t{}'.format(linex.tostr2(),
                                          self.doc_text[linex.lineinfo.start:linex.lineinfo.end]), file=fout)
                    prev_block_num = linex.lineinfo.obid

        print('wrote {}'.format(paged_fname), file=sys.stderr)


    def save_debug_lines(self, extension, work_dir='dir-work'):
        base_fname = os.path.basename(self.file_name)
        paged_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))
        with open(paged_fname, 'wt') as fout:
            for page in self.page_list:
                print('\n===== page #%d, start=%d, end=%d, len(lines)= %d' %
                      (page.page_num, page.start, page.end, len(page.line_list)), file=fout)

                attr_list = []
                for attr, value in page.attrs.items():
                    attr_list.append('{}={}'.format(attr, value))
                if attr_list:
                    print('  attrs: {}'.format(', '.join(attr_list)), file=fout)

                prev_block_num = -1
                for linex in page.line_list:

                    if linex.block_num != prev_block_num:  # this is not obid
                        print(file=fout)
                    print('{}\t{}'.format(linex.tostr2(),
                                          self.doc_text[linex.lineinfo.start:linex.lineinfo.end]), file=fout)
                    prev_block_num = linex.block_num

        print('wrote {}'.format(paged_fname), file=sys.stderr)


    # we do not do our own block merging in pwc version
    def save_debug_lines_pwc(self, extension, work_dir='dir-work'):
        base_fname = os.path.basename(self.file_name)
        paged_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))
        with open(paged_fname, 'wt') as fout:
            for page in self.page_list:
                print('\n===== page #%d, start=%d, end=%d, len(lines)= %d' %
                      (page.page_num, page.start, page.end, len(page.line_list)), file=fout)

                attr_list = []
                for attr, value in page.attrs.items():
                    attr_list.append('{}={}'.format(attr, value))
                if attr_list:
                    print('  attrs: {}'.format(', '.join(attr_list)), file=fout)

                prev_block_num = -1
                for linex in page.line_list:

                    if linex.block_num != prev_block_num:  # this is not obid
                        print(file=fout)
                    print('{}\t{}'.format(linex.tostr2(),
                                          self.doc_text[linex.lineinfo.start:linex.lineinfo.end]), file=fout)
                    prev_block_num = linex.block_num

        print('wrote {}'.format(paged_fname), file=sys.stderr)

class LineInfo3:

    def __init__(self, start, end, line_num, block_num, strinfo_list) -> None:
        self.start = start
        self.end = end
        self.line_num = line_num
        self.obid = block_num   # original block id from pdfbox
        self.bid = block_num    # ordered by block's yStart
        self.strinfo_list = strinfo_list

        min_xStart, min_yStart = MAX_Y_DIFF, MAX_Y_DIFF
        max_xEnd = MIN_X_END
        for strinfo in self.strinfo_list:
            syStart = strinfo.yStart
            sxStart = strinfo.xStart
            sxEnd = strinfo.xEnd

            ## Incorrect??
            ## whichever is lowest in the y-axis of page, use that
            ## Not sure what to do when y-axis equal, or very close
            #if syStart < min_yStart:
            #    min_yStart = syStart
            #    min_xStart = sxStart
            #if sxEnd > max_xEnd:
            #    max_xEnd = sxEnd

            # for a line, str_list should be sorted by xStart, not yStart
            # do we care about min_yStart??
            if sxStart < min_xStart:
                min_yStart = syStart
                min_xStart = sxStart
            if sxEnd > max_xEnd:
                max_xEnd = sxEnd

        self.xStart = min_xStart
        self.xEnd = max_xEnd
        self.yStart = min_yStart
        # jshaw, maybe this is simpler?
        # self.xStart = self.strinfo_list[0].xStart
        # self.yStart = self.strinfo_list[0].yStart
        #self.xEnd = self.strinfo_list[-1].xEnd

    def tostr2(self):
        return 'se=(%d, %d), bid= %d, obid = %d, xs=%.1f, xe= %.1f, ys=%.1f' % (self.start, self.end,
                                                                                self.bid, self.obid,
                                                                                self.xStart,
                                                                                self.xEnd,
                                                                                self.yStart)

    def tostr3(self):
        return 'se=(%d, %d), bid= %d, obid = %d, pn= %d' % (self.start, self.end,
                                                            self.bid, self.obid, self.page_num)


    def tostr4(self):
        return 'bid= %d, obid= %d' % (self.bid, self.obid)


class LineWithAttrs:

    def __init__(self,
                 page_line_num: int,
                 lineinfo: LineInfo3,
                 line_text: str,
                 page_num: int,
                 ydiff: float,
                 linebreak: float,
                 align: str,
                 is_centered: bool,
                 is_english: bool) -> None:
        self.page_line_num = page_line_num  # start from 1, not 0
        self.lineinfo = lineinfo
        self.block_num = lineinfo.bid
        self.line_text = line_text
        self.num_word = len(line_text.split())
        self.page_num = page_num
        self.ydiff = ydiff
        self.linebreak = linebreak
        self.align = align  # inferred
        self.is_centered = is_centered
        self.is_english = is_english
        self.attrs = {}  # type: Dict[str, Any]

    def __lt__(self, other):

        return (self.block_num, self.lineinfo.start).__lt__((other.block_num, other.lineinfo.start))


    def tostr2(self) -> str:
        alist = []
        alist.append('plno=%d' % self.page_line_num)
        alist.append('bnum=%d' % self.block_num)
        alist.append(self.lineinfo.tostr2())
        alist.append('lbk=%.1f' % self.linebreak)
        alist.append('align=%s' % self.align)
        if self.is_centered:
            alist.append('center')
        if not self.is_english:
            alist.append('not_en')
        for attr, value in self.attrs.items():
            alist.append('{}={}'.format(attr, value))
        alist.append('  ||')
        return ', '.join(alist)

    def tostr3(self) -> str:
        alist = []
        alist.append('pn=%d' % self.page_num)
        alist.append('bnum=%d' % self.block_num)
        # alist.append('plno=%d' % self.page_line_num)
        alist.append(self.lineinfo.tostr2())
        alist.append('align=%s' % self.align)
        if self.linebreak != 1.0:
            alist.append('lbk=%.1f' % self.linebreak)

        if len(self.lineinfo.strinfo_list) != 1:
            alist.append('len(strlst)=%d' % len(self.lineinfo.strinfo_list))

        if self.is_centered:
            alist.append('center')
        if not self.is_english:
            alist.append('not_en')
        for attr, value in sorted(self.attrs.items()):
            alist.append('{}={}'.format(attr, value))
        alist.append('  ||')
        return ', '.join(alist)

    def tostr4(self) -> str:
        alist = []
        alist.append('pn=%d' % self.page_num)
        alist.append('bnum=%d' % self.block_num)
        alist.append('align=%s' % self.align)
        if self.linebreak != 1.0:
            alist.append('lbk=%.1f' % self.linebreak)

        if self.is_centered:
            alist.append('center')
        if not self.is_english:
            alist.append('not_en')
        for attr, value in self.attrs.items():
            alist.append('{}={}'.format(attr, value))
        return ', '.join(alist)

    # mainly for detailed debugging purpose
    def tostr5(self) -> str:
        alist = []
        alist.append('pn=%d' % self.page_num)
        alist.append('bnum=%d' % self.block_num)
        alist.append(self.lineinfo.tostr2())
        alist.append('align=%s' % self.align)
        if self.linebreak != 1.0:
            alist.append('lbk=%.1f' % self.linebreak)

        if len(self.lineinfo.strinfo_list) != 1:
            alist.append('len(strlst)=%d' % len(self.lineinfo.strinfo_list))

        if self.is_centered:
            alist.append('center')
        if not self.is_english:
            alist.append('not_en')
        for attr, value in self.attrs.items():
            alist.append('{}={}'.format(attr, value))
        return ', '.join(alist)

    def to_attrvals(self) -> Dict[str, Any]:
        """returns a dict()"""
        adict = {}
        adict['pnum'] = self.page_num
        adict['bnum'] = self.block_num
        # adict.append(('bn', self.lineinfo.block_num))
        # adict.append(('align', self.align))
        if self.is_centered:
            adict['center'] = 1
        if not self.is_english:
            adict['not_en'] = 1
        adict.update(self.attrs)
        return adict

    def __str__(self):
        return str(self.to_attrvals())

    def to_para_attrvals(self) -> List[Any]:
        """returns a dict()"""
        adict = {}
        adict['pnum'] = self.page_num
        adict['bnum'] = self.block_num
        # adict.append(('bn', self.lineinfo.block_num))
        # adict.append(('align', self.align))
        if self.is_centered:
            adict['center'] = True
        if not self.is_english:
            adict['not_en'] = True
        adict.update(self.attrs)

        result = []  # type: List[Any]
        for attr, value in adict.items():
            if attr == 'sechead':
                if value:  # value is false sometimes?? TODO, jshaw, fix
                    result.append(value)
                # else: pass
            else:
                result.append((attr, value))
        return sorted(result)


@total_ordering
class PBlockInfo:

    def __init__(self,
                 start: int,
                 end: int,
                 bid: int,
                 pagenum: int,
                 text: str,
                 lineinfo_list: List[LineInfo3],
                 is_multi_lines: bool) -> None:
        self.start = start
        self.end = end
        self.obid = bid     # original block id
        self.bid = bid      # sorted by yStart
        self.pagenum = pagenum
        self.text = text
        self.length = len(text)
        self.lineinfo_list = lineinfo_list
        self.is_multi_lines = is_multi_lines

        pb_min_xStart, pb_min_yStart = MAX_Y_DIFF, MAX_Y_DIFF
        pb_max_xEnd, pb_max_yEnd = MIN_X_END, MIN_X_END

        for lineinfo in self.lineinfo_list:
            lx_min_xStart, lx_min_yStart = lineinfo.xStart, lineinfo.yStart
            lx_max_xEnd = lineinfo.xEnd

            if lx_min_yStart < pb_min_yStart:
                pb_min_yStart = lx_min_yStart
                pb_min_xStart = lx_min_xStart
                #print("block_id = {}, is_multi_lines = {}, len(lxinfo_list)= {}, sxEnd = {}, pb_max_xEnd = {}".format(
                #      bid, is_multi_lines, len(lineinfo_list), sxEnd, pb_max_xEnd))
                # if (len(lineinfo_list) == 1 or is_multi_lines) and sxEnd > pb_max_xEnd:
            if lx_max_xEnd > pb_max_xEnd:
                pb_max_xEnd = lx_max_xEnd
            if lx_min_yStart > pb_max_yEnd:
                pb_max_yEnd = lx_min_yStart

        self.xStart = pb_min_xStart
        self.xEnd = pb_max_xEnd
        self.yStart = pb_min_yStart
        self.yEnd = pb_max_yEnd
        # print("self.xStart = {}, xEnd = {}, yStart= {}".format(self.xStart, self.xEnd, self.yStart))
        self.is_english = engutils.classify_english_sentence(text)
        self.ydiff = MAX_Y_DIFF  # will compute this across PBlockInfo

    def __eq__(self, other) -> bool:
        #if hasattr(other, 'start') and hasattr(other, 'end'):
        return (self.start, self.end).__eq__((other.start, other.end))

    def __lt__(self, other):
        #if hasattr(other, 'start') and hasattr(other, 'end'):
        return (self.start, self.end).__lt__((other.start, other.end))

    def __hash__(self):
        return hash((self.start, self.end))

    def __repr__(self) -> str:
        # self.words is not printed, intentionally
        return str(('start={}'.format(self.start),
                    'end={}'.format(self.end),
                    'bid={}'.format(self.bid),
                    'xStart={:.2f}'.format(self.xStart),
                    'yStart={:.2f}'.format(self.yStart),
                    'pagenum={}'.format(self.pagenum),
                    'english={}'.format(self.is_english)))

    """
    def tostr2(self, text) -> str:
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
"""
    # there is no self.align_label??

    """
    def is_center(self) -> bool:
        return self.align_label == 'CN'

    def is_LF(self) -> bool:
        return self.align_label and self.align_label.startswith('LF')

    def is_LF1(self) -> bool:
        return self.align_label == 'LF1'

    def is_LF2(self) -> bool:
        return self.align_label == 'LF2'

    def is_LF3(self) -> bool:
        return self.align_label == 'LF3'
"""


class GroupedBlockInfo:

    def __init__(self,
                 pagenum: int,
                 bid: int,
                 line_list: List[LineWithAttrs]) \
                 -> None:
        self.bid = bid
        self.pagenum = pagenum
        self.line_list = line_list
        self.attrs = {}  # type: Dict[str, Any]


def lines_to_block_offsets(linex_list: List[LineWithAttrs],
                           block_type: str,
                           pagenum: int) \
                           -> Tuple[int, int, Dict[str, Any]]:
    if linex_list:
        min_start, max_end = linex_list[0].lineinfo.start, linex_list[-1].lineinfo.end
        # in case the original line order are not correct from pdfbox, we
        # ensure the min and max are correct
        for linex in linex_list:
            if linex.lineinfo.start < min_start:
                min_start = linex.lineinfo.start
            if linex.lineinfo.end > max_end:
                max_end = linex.lineinfo.end
        return min_start, max_end, {'block-type': block_type, 'pagenum': pagenum}
    # why would this happen?
    return 0, 0, {'block-type': block_type}


def line_to_block_offsets(linex: LineWithAttrs,
                          block_type: str,
                          pagenum: int) \
                          -> Tuple[int, int, Dict]:
    start = linex.lineinfo.start
    end = linex.lineinfo.end
    return start, end, {'block-type': block_type, 'pagenum': pagenum}


def lines_to_blocknum_map(linex_list: List[LineWithAttrs]) \
    -> DefaultDict[str, List[LineWithAttrs]]:
    result = defaultdict(list)  # type: DefaultDict[str, List[LineWithAttrs]]
    for linex in linex_list:
        block_num = linex.block_num
        result[block_num].append(linex)
    return result

# this doesn't work anymore because we added the ability to increase "block_num" by 10000
# when breaking up a block due to header/english separation
"""
def line_list_to_grouped_block_list(linex_list, page_num):
    block_list_map = defaultdict(list)
    for linex in linex_list:
        block_num = linex.block_num
        block_list_map[block_num].append(linex)

    grouped_block_list = []
    for block_num, line_list in sorted(block_list_map.items()):
        grouped_block_list.append(GroupedBlockInfo(page_num, block_num, line_list))

    return grouped_block_list
"""

def line_list_to_grouped_block_list(linex_list: List[LineWithAttrs],
                                    page_num: int) \
                                    -> List[GroupedBlockInfo]:
    tmp_block_list = docstructutils.line_list_to_block_list(linex_list)
    grouped_block_list = []  # type: List[GroupedBlockInfo]
    for linex_list in tmp_block_list:
        block_num = linex_list[0].block_num
        grouped_block_list.append(GroupedBlockInfo(page_num, block_num, linex_list))

    return grouped_block_list
