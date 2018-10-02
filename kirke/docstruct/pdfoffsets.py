from array import ArrayType
from collections import namedtuple, defaultdict
from functools import total_ordering
import os
import sys
from typing import Any, DefaultDict, Dict, List, Tuple

from kirke.docstruct import docstructutils, jenksutils
# pylint: disable=unused-import
from kirke.docstruct import linepos
from kirke.docstruct.docutils import PLineAttrs
from kirke.utils import engutils
from kirke.utils.textoffset import TextCpointCunitMapper


StrInfo = namedtuple('StrInfo', ['start', 'end',
                                 'xStart', 'xEnd', 'yStart', 'yEnd',
                                 'height', 'font_size'])

MAX_Y_DIFF = 10000
MIN_X_END = -1


# pylint: disable=too-few-public-methods
class PageAttrs:

    def __init__(self) -> None:
        self.has_toc = False
        # these are set using setattr() in reset_all_is_english()
        self.has_signature = False
        self.has_address = False
        self.has_table = False
        # page_num_index points to the line that has the page number
        # so we can detect footers after page_num, 1-based
        self.page_num_index = -1

    def __str__(self) -> str:
        alist = []  # List[str]
        if self.has_toc:
            alist.append('{}={}'.format('has_toc', self.has_toc))
        if self.has_signature:
            alist.append('{}={}'.format('has_signature', self.has_signature))
        if self.has_address:
            alist.append('{}={}'.format('has_address', self.has_address))
        if self.has_table:
            alist.append('{}={}'.format('has_table', self.has_table))
        if self.page_num_index != -1:
            alist.append('{}={}'.format('page_num_index', self.page_num_index))
        return '|'.join(alist)

    def has_any_attrs(self) -> bool:
        return (self.has_toc or
                self.has_signature or
                self.has_address or
                self.has_table or
                self.page_num_index != -1)


# pylint: disable=too-many-instance-attributes
class LineInfo3:

    # pylint: disable=too-many-arguments
    def __init__(self, start, end, line_num, block_num, strinfo_list) -> None:
        self.start = start
        self.end = end
        self.line_num = line_num
        self.obid = block_num   # original block id from pdfbox
        self.bid = block_num    # ordered by block's yStart
        self.strinfo_list = strinfo_list

        # pylint: disable=invalid-name
        min_xStart, min_yStart = MAX_Y_DIFF, MAX_Y_DIFF
        # pylint: disable=invalid-name
        max_xEnd, max_yEnd = MIN_X_END, MIN_X_END
        # the smaller than the smallest we have found so far
        max_height, max_font_size = 4.0, 6
        for strinfo in self.strinfo_list:
            syStart = strinfo.yStart
            syEnd = strinfo.yEnd
            sxStart = strinfo.xStart
            sxEnd = strinfo.xEnd
            sHeight = strinfo.height
            sFontSize = strinfo.font_size

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
                min_xStart = sxStart
            if sxEnd > max_xEnd:
                max_xEnd = sxEnd
            if syStart < min_yStart:
                min_yStart = syStart
            if syEnd > max_yEnd:
                max_yEnd = syEnd
            if sHeight > max_height:
                max_height = sHeight
            if sFontSize > max_font_size:
                max_font_size = sFontSize

        # pylint: disable=invalid-name
        self.xStart = min_xStart
        # pylint: disable=invalid-name
        self.xEnd = max_xEnd
        # pylint: disable=invalid-name
        self.yStart = min_yStart
        # pylint: disable=invalid-name
        self.yEnd = max_yEnd
        self.height = max_height
        self.font_size = max_font_size

        # jshaw, maybe this is simpler?
        # self.xStart = self.strinfo_list[0].xStart
        # self.yStart = self.strinfo_list[0].yStart
        #self.xEnd = self.strinfo_list[-1].xEnd

    def tostr2(self):
        return 'se=(%d, %d), ybid= %d, obid = %d, xs=%.1f, xe= %.1f, ys=%.1f ye=%.1f h=%.1f, font=%.1f' % \
            (self.start, self.end,
             self.bid, self.obid,
             self.xStart, self.xEnd,
             self.yStart, self.yEnd,
             self.height, self.font_size)

    def tostr3(self):
        return 'se=(%d, %d), bid= %d, obid = %d' % (self.start, self.end,
                                                    self.bid, self.obid)


    def tostr4(self):
        return 'bid= %d, obid= %d' % (self.bid, self.obid)


# pylint: disable=too-many-instance-attributes
class LineWithAttrs:

    # pylint: disable=too-many-arguments
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
        self.attrs = PLineAttrs()  # type: PLineAttrs

    def __lt__(self, other) -> bool:

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
        alist.append(str(self.attrs))
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
        alist.append(str(self.attrs))
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
        alist.append(str(self.attrs))
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
        alist.append(str(self.attrs))
        # for attr, value in self.attrs.items():
        #    alist.append('{}={}'.format(attr, value))
        return ', '.join(alist)

    def to_attrvals(self) -> PLineAttrs:
        """returns a copy of PLineAttrs"""
        attrs = PLineAttrs()

        attrs.pnum = self.page_num
        attrs.bnum = self.block_num
        if self.is_centered:
            attrs.center = True
        if not self.is_english:
            attrs.not_en = True
        attrs.toc = self.attrs.toc
        attrs.signature = self.attrs.signature
        attrs.sechead = self.attrs.sechead
        return attrs

    def __str__(self):
        return str(self.to_attrvals())


@total_ordering
# pylint: disable=too-few-public-methods, too-many-instance-attributes
class PBlockInfo:

    # pylint: disable=too-many-arguments, too-many-locals
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

        # pylint: disable=invalid-name
        pb_min_xStart, pb_min_yStart = MAX_Y_DIFF, MAX_Y_DIFF
        # pylint: disable=invalid-name
        pb_max_xEnd, pb_max_yEnd = MIN_X_END, MIN_X_END

        for lineinfo in self.lineinfo_list:
            # pylint: disable=invalid-name
            lx_min_xStart, lx_min_yStart = lineinfo.xStart, lineinfo.yStart
            # pylint: disable=invalid-name
            lx_max_xEnd = lineinfo.xEnd

            if lx_min_yStart < pb_min_yStart:
                pb_min_yStart = lx_min_yStart
                pb_min_xStart = lx_min_xStart
                #print("block_id = {}, is_multi_lines = {}, len(lxinfo_list)= {}, sxEnd = {}, "
                #      "pb_max_xEnd = {}".format(
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
        # print("self.xStart = {}, xEnd = {}, yStart= {}".format(self.xStart, self.xEnd,
        #                                                        self.yStart))
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

# pylint: disable=too-many-locals
def init_lines_with_attr(pblockinfo_list: List[PBlockInfo],
                         avg_single_line_break_ydiff: float,
                         doc_text: str,
                         page_num: int) \
                         -> List[LineWithAttrs]:
    """Add identation information by using jenks library (clustering).
       Add the line break information to indicate how far apart two lines are.
       Other information include 'centered', is_engish.

    Return a list of LineWithAttrs.
    """
    lineinfo_list = [lineinfo
                     for pblockinfo in pblockinfo_list
                     for lineinfo in pblockinfo.lineinfo_list]
    xstart_list = [lineinfo.xStart for lineinfo in lineinfo_list]

    # print("leninfo_list = {}, xstart_list = {}, page_num = {}".format(len(lineinfo_list),
    # xstart_list, page_num))
    if lineinfo_list:  # not an empty page
        if len(xstart_list) == 1:
            # duplicate itself to avoid jenks error with only element
            xstart_list.append(xstart_list[0])
        jenks = jenksutils.Jenks(xstart_list)

    line_attrs = []  # type: List[LineWithAttrs]
    # pylint: disable=invalid-name
    prev_yStart = 0
    for page_line_num, lineinfo in enumerate(lineinfo_list, 1):
        ydiff = lineinfo.yStart - prev_yStart
        # it possible for self.avg_single_line_break_ydiff to be 0 when
        # the document has only vertical lines.
        if avg_single_line_break_ydiff == 0:
            num_linebreak = 1.0  # hopeless, default to 1 for now
        else:
            num_linebreak = round(ydiff / avg_single_line_break_ydiff, 1)
        align = jenks.classify(lineinfo.xStart)
        line_text = doc_text[lineinfo.start:lineinfo.end]
        is_english = engutils.classify_english_sentence(line_text)
        is_centered = docstructutils.is_line_centered(line_text,
                                                      lineinfo.xStart,
                                                      lineinfo.xEnd)
        line_attrs.append(LineWithAttrs(page_line_num,
                                        lineinfo,
                                        line_text,
                                        page_num,
                                        ydiff,
                                        num_linebreak,
                                        align,
                                        is_centered,
                                        is_english))
        prev_yStart = lineinfo.yStart

    return line_attrs


# pylint: disable=invalid-name
def compute_avg_single_line_break_ydiff(pblockinfo_list: List[PBlockInfo]) \
    -> float:
    """This avg_single_line_break_ydff is based on the distance between lines inside
    paragraphs in this page.
    This value is used for setting num_linebreak for LineWithAttrs in above init_lines_with_attr().
    """
    total_merged_ydiff, total_merged_lines = 0, 0
    for pblockinfo in pblockinfo_list:
        is_multi_lines = pblockinfo.is_multi_lines
        if not (is_multi_lines or len(pblockinfo.lineinfo_list) == 1):
            total_merged_ydiff += pblockinfo.yEnd - pblockinfo.yStart
            total_merged_lines += len(pblockinfo.lineinfo_list) - 1

    result = 14.0  # default value, just in case
    if total_merged_lines != 0:
        result = total_merged_ydiff / total_merged_lines
    # print("\npage #{}, avg_single_line_ydiff = {}".format(self.page_num, result))
    return result


# pylint: disable=too-many-instance-attributes
class PageInfo3:

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(self,
                 doc_text: str,
                 start: int,
                 end: int,
                 page_num: int,
                 pblockinfo_list) \
                 -> None:
        self.start = start
        self.end = end
        self.page_num = page_num

        # Fixes header and footer issues due to out of order lines.
        # Also out of order blocks due to tables and header.  p76 carousel.txt
        self.pblockinfo_list = sorted(pblockinfo_list, key=lambda x: x.start)

        avg_single_line_break_ydiff = \
            compute_avg_single_line_break_ydiff(self.pblockinfo_list)
        self.line_list = init_lines_with_attr(self.pblockinfo_list,
                                              avg_single_line_break_ydiff,
                                              doc_text=doc_text,
                                              page_num=page_num)

        self.attrs = PageAttrs()  # type: PageAttrs
        self.is_continued_para_to_next_page = False
        self.is_continued_para_from_prev_page = False

        # conent_line_list is for lines that are not
        #   - toc
        #   - page_num
        #   - header, footer
        self.content_line_list = []  # type: List[LineWithAttrs]

    def get_blocked_lines(self) -> List[List[LineWithAttrs]]:
        if not self.line_list:
            return []
        prev_block_num = -1
        cur_block = []  # type: List[LineWithAttrs]
        block_list = [cur_block]  # type: List[List[LineWithAttrs]]
        for linex in self.line_list:
            if linex.lineinfo.obid != prev_block_num:  # separate blocks
                if cur_block:
                    cur_block = []
                    block_list.append(cur_block)
            cur_block.append(linex)
            prev_block_num = linex.lineinfo.obid
        return block_list


class PDFTextDoc:

    # pylint: disable=too-many-arguments
    def __init__(self,
                 file_name: str,
                 doc_text: str,
                 *,
                 cpoint_cunit_mapper: TextCpointCunitMapper,
                 pageinfo_list: List[PageInfo3],
                 linebreak_arr: ArrayType) \
                 -> None:
        self.file_name = file_name
        self.doc_text = doc_text
        self.cpoint_cunit_mapper = cpoint_cunit_mapper
        self.page_list = pageinfo_list
        self.num_pages = len(pageinfo_list)
        # pylint: disable=line-too-long
        self.special_blocks_map = defaultdict(list)  # type: DefaultDict[str, List[Tuple[int, int, Dict[str, Any]]]]
        self.linebreak_arr = linebreak_arr
        self.removed_lines = []  # type: List[LineWithAttrs]
        self.exclude_offsets = []  # type: List[Tuple[int, int]]

        # pylint: disable=line-too-long
        self.nlp_paras_with_attrs = []  # type: List[Tuple[List[Tuple[linepos.LnPos, linepos.LnPos]], PLineAttrs]]

    def get_nlp_text(self) -> str:
        return docstructutils.text_from_para_with_attrs(self.doc_text, self.nlp_paras_with_attrs)

    def get_page_offsets(self) -> List[Tuple[int, int]]:
        return [(page.start, page.end) for page in self.page_list]

    def save_raw_pages(self,
                       extension: str,
                       work_dir: str = 'dir-work'):
        base_fname = os.path.basename(self.file_name)
        out_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))
        with open(out_fname, 'wt') as fout:
            for page in self.page_list:
                print('\n===== page #%d, start=%d, end=%d, len(lines)= %d' %
                      (page.page_num, page.start, page.end, len(page.line_list)), file=fout)
                if page.attrs:
                    print('  attrs: {}'.format(str(page.attrs)),
                          file=fout)

                prev_block_num = -1
                for linex in page.line_list:
                    if linex.lineinfo.obid != prev_block_num:  # separate blocks
                        print(file=fout)
                    print('{}\t{}'.format(linex.tostr2(),
                                          self.doc_text[linex.lineinfo.start:linex.lineinfo.end]),
                          file=fout)
                    prev_block_num = linex.lineinfo.obid

        print('wrote {}'.format(out_fname), file=sys.stderr)


    def save_debug_lines(self, extension, work_dir='dir-work'):
        base_fname = os.path.basename(self.file_name)
        out_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))
        with open(out_fname, 'wt') as fout:
            for page in self.page_list:
                print('\n===== page #%d, start=%d, end=%d, len(lines)= %d' %
                      (page.page_num, page.start, page.end, len(page.line_list)), file=fout)

                if page.attrs.has_any_attrs():
                    print('  attrs: {}'.format(str(page.attrs)), file=fout)

                prev_block_num = -1
                for linex in page.line_list:

                    if linex.block_num != prev_block_num:  # this is not obid
                        print(file=fout)
                    print('{}\t{}'.format(linex.tostr2(),
                                          self.doc_text[linex.lineinfo.start:linex.lineinfo.end]),
                          file=fout)
                    prev_block_num = linex.block_num

        print('wrote {}'.format(out_fname), file=sys.stderr)


    # we do not do our own block merging in pwc version
    def save_debug_lines_pwc(self, extension, work_dir='dir-work'):
        base_fname = os.path.basename(self.file_name)
        out_fname = '{}/{}'.format(work_dir, base_fname.replace('.txt', extension))
        with open(out_fname, 'wt') as fout:
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
                                          self.doc_text[linex.lineinfo.start:linex.lineinfo.end]),
                          file=fout)
                    prev_block_num = linex.block_num

        print('wrote {}'.format(out_fname), file=sys.stderr)


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
