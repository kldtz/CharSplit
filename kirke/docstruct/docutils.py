from operator import itemgetter

from typing import Dict, List, Tuple

from kirke.utils import strutils
from kirke.docstruct import linepos


# pylint: disable=too-few-public-methods, too-many-instance-attributes
class PLineAttrs:

    def __init__(self):
        self.pnum = -1
        self.bnum = -1
        self.not_en = False
        self.center = False
        self.toc = False
        self.sechead = None  # type: Optional[Tuple[str, str, str, int]]
        self.footer = False
        self.header = False

        self.signature = False
        self.address = False
        self.table = ''
        self.chart = ''

        # whether a page has a page number line or not
        self.has_page_num = False

        self.table_row = -1

    def __str__(self):
        alist = []  # List[str]
        if self.pnum != -1:
            alist.append('{}={}'.format('pnum', self.pnum))
        if self.bnum != -1:
            alist.append('{}={}'.format('bnum', self.bnum))
        if self.not_en:
            alist.append('{}={}'.format('not_en', self.not_en))
        if self.center:
            alist.append('{}={}'.format('center', self.center))
        if self.toc:
            alist.append('{}={}'.format('toc', self.toc))
        if self.sechead:
            alist.append('{}={}'.format('sechead', self.sechead))
        if self.footer:
            alist.append('{}={}'.format('footer', self.footer))
        if self.header:
            alist.append('{}={}'.format('header', self.header))

        if self.signature:
            alist.append('{}={}'.format('signature', self.signature))
        if self.address:
            alist.append('{}={}'.format('address', self.address))
        if self.table:
            alist.append('{}={}'.format('table', self.table))
        if self.chart:
            alist.append('{}={}'.format('chart', self.chart))

        if self.table_row != -1:
            alist.append('{}={}'.format('table_row', self.table_row))

        if self.has_page_num:
            alist.append('{}={}'.format('has_page_num', self.has_page_num))

        return ', '.join(alist)


def lnpos2dict(start: int, end: int, lnpos: linepos.LnPos) -> Dict:
    adict = {'start': start,
             'end': end,
             'line_num': lnpos.line_num}
    if lnpos.start == lnpos.end:
        adict['gap'] = True
    return adict


# startx and endx are related to to_list
# we want the offset in from_list
# pylint: disable=too-many-locals
def find_se_offset_list(startx: int,
                        endx: int,
                        from_list: List[Tuple[int, linepos.LnPos]],
                        to_list: List[Tuple[int, linepos.LnPos]]) -> List[Dict]:

    # result is an offset list
    result = []  # type: List[Dict]

    # start_lnpos = None  # this cause Optional[linepos.LnPos], problems
    _, start_lnpos = to_list[0]  # set start_lnpos to a value, will be overriden
    start_diff = 0
    start_idx = 0
    for i, ((fstart, unused_from_lnpos), (unused_tstart, to_lnpos)) in \
        enumerate(zip(from_list, to_list)):
        if fstart == startx:
            start_lnpos = to_lnpos
            start_diff = 0
            start_idx = i
            break
        elif fstart > startx:
            _, start_lnpos = to_list[i-1]
            prev_start, unused_lnpos = from_list[i-1]
            start_diff = startx - prev_start
            start_idx = i - 1
            break

    # end_lnpos = None  # this cause Optional[linepos.LnPos], problems
    _, end_lnpos = to_list[0]  # set start_lnpos to a value, will be overriden
    end_diff = 0
    end_idx = 0  # this is inclusive
    for i, ((fstart, unused_from_lnpos), (unused_tstart, to_lnpos)) in \
        enumerate(zip(from_list, to_list)):
        if fstart == endx:
            end_lnpos = to_lnpos
            end_diff = 0
            end_idx = i
            break
        elif fstart > endx:
            _, end_lnpos = to_list[i-1]
            prev_start, unused_lnpos = from_list[i-1]
            end_diff = endx - prev_start
            end_idx = i - 1
            break

    # they are in the same line
    if start_idx == end_idx:
        result = [{'start': start_lnpos.start + start_diff,
                   'end': start_lnpos.start + end_diff}]
    else:
        # skipping all middle parts for now
        result = [lnpos2dict(start_lnpos.start + start_diff,
                             start_lnpos.end,
                             start_lnpos),
                  lnpos2dict(end_lnpos.start,
                             end_lnpos.start + end_diff,
                             end_lnpos)]
        result = sorted(result, key=itemgetter('start'))
    return result


def read_fromto_json(file_name: str) -> Tuple[List[int],
                                              List[int]]:
    fromto_list = strutils.load_json_list(file_name)

    alist = []  # type: List[Tuple[int, int]]
    for fromto in fromto_list:
        from_start = fromto['from_start']
        # from_end = fromto['from_end']
        to_start = fromto['to_start']
        # to_end = fromto['to_end']
        alist.append((from_start, to_start))
        # alist.append((from_end, to_end))

    sorted_alist = sorted(alist)

    from_list = [a for a, b in sorted_alist]
    to_list = [b for a, b in sorted_alist]
    return from_list, to_list


def span_frto_list_to_fromto(span_frto_list: List[Tuple[linepos.LnPos,
                                                        linepos.LnPos]]) -> Tuple[Tuple[int, int],
                                                                                  Tuple[int, int]]:
    from_lnpos, to_lnpos = span_frto_list[0]
    from_start, from_end = from_lnpos.start, from_lnpos.end
    to_start, to_end = to_lnpos.start, to_lnpos.end

    for span_fromto in span_frto_list[1:]:
        from_lnpos2, to_lnpos2 = span_fromto
        fe2 = from_lnpos2.end
        te2 = to_lnpos2.end
        if fe2 > from_end:
            from_end = fe2
        if te2 > to_end:
            to_end = te2
    return (from_start, from_end), (to_start, to_end)
