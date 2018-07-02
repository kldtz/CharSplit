#!/usr/bin/env python3

from collections import OrderedDict
import logging
import re
import sys
# pylint: disable=unused-import
from typing import Dict, List, Match, Optional, TextIO, Tuple

from kirke.abbyxml import abbyxmlparser
from kirke.abbyxml.pdfoffsets import AbbyLine, AbbyPage, UnsyncedPBoxLine, UnsyncedStrWithY
from kirke.abbyxml.pdfoffsets import AbbyTableBlock, AbbyTextBlock, AbbyXmlDoc
from kirke.abbyxml.pdfoffsets import print_abby_page_unsynced, print_abby_page_unsynced_aux
from kirke.docstruct.pdfoffsets import PDFTextDoc, PageInfo3
from kirke.utils.alignedstr import AlignedStrMapper


# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

IS_DEBUG_SYNC = True
# level 2, sync debug, more detailed info
IS_DEBUG_SYNC_L2 = False

# This is just to print out the unsync information
IS_DEBUG_UNSYNC = True

IS_DEBUG_XY_DIFF = False

X_BOX_CHECK_MIN = 2
X_BOX_CHECK_MAX = 3

Y_BOX_CHECK_MIN_1 = 12
Y_BOX_CHECK_MIN_2 = 21
Y_BOX_CHECK_MAX = 3


# pylint: disable=invalid-name
def find_um_pbox_line_by_xy(x: int,
                            y: int,
                            pbox_xy_map: Dict[Tuple[int, int],
                                              UnsyncedPBoxLine]) \
                                              -> Optional[UnsyncedPBoxLine]:
    """
    Returns
        The first tuple, is original X, Y
        The 2nd tuple, the the start, end
        The 3rd string is the text
    """
    # print("x, y = {}, {}".format(x, y))
    # for attr, val in pbox_xy_map.items():
    #     print("pbox_xy_map[{}] = {}".format(attr, val))

    # in experiment, x - tmp_x ranges from 0 to 1, inclusive
    for tmp_x in range(x-X_BOX_CHECK_MIN, x+X_BOX_CHECK_MAX):
        # in experiment, y - tmp_y ranges from 0 to 12, inclusive,
        # heavy toward 12.
        # the y diff between lines is in the 60 range, so ok.
        for tmp_y in range(y-Y_BOX_CHECK_MIN_1, y+Y_BOX_CHECK_MAX):
            # print("try tmp_x, tmp_y = {}, {}".format(tmp_x, tmp_y))
            um_pbox_line = pbox_xy_map.get((tmp_x, tmp_y))
            if um_pbox_line:
                if IS_DEBUG_XY_DIFF:
                    print("jjjdiff1 abby-pbox:\t{}\t{}".format(x - tmp_x, y - tmp_y))
                    # to see the typical diffs between x and y
                    # grep jjjdiff kkk | cut -f 2 | sort | uniq -c
                    # grep jjjdiff kkk | cut -f 3 | sort | uniq -c
                return um_pbox_line
        # in another doc, I have seen 17, so do y-13 to (y+3] first
        # then do y-14 to (y-21]
        # Y_BOX_CHECK_MIN_1+1 = 13
        for tmp_y in range(y-(Y_BOX_CHECK_MIN_1+1), y-Y_BOX_CHECK_MIN_2, -1):
            # print("try tmp_x, tmp_y = {}, {}".format(tmp_x, tmp_y))
            um_pbox_line = pbox_xy_map.get((tmp_x, tmp_y))
            if um_pbox_line:
                if IS_DEBUG_XY_DIFF:
                    print("jjjdiff2 abby-pbox:\t{}\t{}".format(x - tmp_x, y - tmp_y))
                return um_pbox_line
    return None


class StrMappedTracker:

    def __init__(self) -> None:
        self.strxy_used_map = {}  # type: Dict[Tuple[int, int], bool]

    def add(self, xy_pair: Tuple[int, int]) -> None:
        self.strxy_used_map[xy_pair] = False

    def get_unused_xy_list(self) -> List[Tuple[int, int]]:
        return [xy_pair for xy_pair, is_used
                in self.strxy_used_map.items() if not is_used]

    def set_used(self, xy_pair: Tuple[int, int]) -> None:
        self.strxy_used_map[xy_pair] = True


# pylint: disable=invalid-name
def find_unique_str_in_unmatched_ablines(stext: str,
                                         unmatched_ablines: List[AbbyLine]) \
                                         -> Optional[Tuple[Match[str],
                                                           AbbyLine]]:
    mats_um_abline_list = []  # type: List[Tuple[List[Match[str]], AbbyLine]]
    for um_abline in unmatched_ablines:
        mat_st = re.escape(stext)
        tmp_mat_list = list(re.finditer(mat_st, um_abline.text))
        if tmp_mat_list:
            mats_um_abline_list.append((tmp_mat_list, um_abline))
    # Must only have only 1 ab_line has the str,
    # otherwise, too ambiguous and return None
    if len(mats_um_abline_list) == 1:
        # if ab_line.text matches, it can match once
        mats, um_abline = mats_um_abline_list[0]
        if len(mats) != 1:
            return None
        return mats[0], um_abline
    return None


def pbox_xy_list_to_pbox_lines(um_pbox_xy_list: List[Tuple[int, int]],
                               pbox_xy_map: Dict[Tuple[int, int],
                                                 UnsyncedPBoxLine]) \
                                                 -> List[UnsyncedPBoxLine]:
    return [pbox_xy_map[um_pbox_xy]
            for um_pbox_xy in um_pbox_xy_list]


# pylint: disable=too-many-locals
def find_um_abline_in_pbox_strs_by_y_strmatch(um_abline: AbbyLine,
                                              um_pbox_xy_list: List[Tuple[int, int]],
                                              pbox_xy_map: Dict[Tuple[int, int],
                                                                UnsyncedPBoxLine],
                                              pbox_extra_se_list: List[UnsyncedStrWithY],
                                              is_skip_y_check: bool = False) \
                                            -> Tuple[bool,
                                                     List[Tuple[int, int]],
                                                     List[UnsyncedStrWithY]]:

    # print("find_aligned_abline_in_pbox_strs()")
    # print("       um_abline: {}".format(um_abline))
    # for i, xy_se_str in enumerate(xy_se_str_list):
    #     print("   unused_pbox_str #{}: {}".format(i, xy_se_str))
    # for i, extra_se in enumerate(pbox_extra_se_list):
    #     pbox_y, (start, end), text, unused_ignore = extra_se
    #    print("    extra_pbox_str #{}: {} ({}, {}) [{}]".format(i, pbox_y, start, end, text))

    um_pbox_lines = pbox_xy_list_to_pbox_lines(um_pbox_xy_list,
                                               pbox_xy_map)

    # pylint: disable=line-too-long
    asm_list = []  # type: List[AlignedStrMapper]

    abline_st = um_abline.text
    um_abline_y = um_abline.infer_attr_dict['y']
    to_remove_pbox_xy_list = []  # type: List[Tuple[int, int]]
    for um_pbox_line in um_pbox_lines:
        xypair, (to_start, unused_to_end), text = um_pbox_line.to_tuple()
        unused_pbox_x, pbox_y = xypair

        if is_skip_y_check or \
           (pbox_y - Y_BOX_CHECK_MIN_2 <= um_abline_y and
            um_abline_y < pbox_y + Y_BOX_CHECK_MAX):
            abby_pbox_offset_mapper = AlignedStrMapper(abline_st,
                                                       text,
                                                       to_start)
            if abby_pbox_offset_mapper.is_fully_synced:
                asm_list.append(abby_pbox_offset_mapper)
                to_remove_pbox_xy_list.append(xypair)
                if IS_DEBUG_SYNC:
                    print("aligned matched 1 !! abline_st [{}]".format(abline_st))
                    print("                     pdfbox_st [{}]".format(text))

    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(asm_list) == 1:
        um_abline.abby_pbox_offset_mapper = asm_list[0]
        out_um_pbox_xy_list = [xypair for xypair in um_pbox_xy_list
                               if xypair not in to_remove_pbox_xy_list]
        # pbox_extra_se_list is NOT modified
        return True, out_um_pbox_xy_list, pbox_extra_se_list

    # pylint: disable=line-too-long
    to_remove_pbox_extra_se_list = []  # type: List[UnsyncedStrWithY]
    for unused_i, pbox_extra_se in enumerate(pbox_extra_se_list):
        pbox_y, (start, unused_end), text, unused_as_mapper = pbox_extra_se.to_tuple()
        # print("   extra_pbox_str #{}: {} ({}, {}) [{}]".format(i, pbox_y, start, end, text))

        if is_skip_y_check or \
           (pbox_y - Y_BOX_CHECK_MIN_2 <= um_abline_y and
            um_abline_y < pbox_y + Y_BOX_CHECK_MAX):

            abby_pbox_offset_mapper = AlignedStrMapper(abline_st,
                                                       text,
                                                       start)

            if abby_pbox_offset_mapper.is_fully_synced:
                asm_list.append(abby_pbox_offset_mapper)
                to_remove_pbox_extra_se_list.append(pbox_extra_se)
                if IS_DEBUG_SYNC:
                    print("aligned matched 2 !! abline_st [{}]".format(abline_st))
                    print("                     pdfbox_st [{}]".format(text))

    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(asm_list) == 1:
        um_abline.abby_pbox_offset_mapper = asm_list[0]
        out_pbox_extra_se_list = [pbox_extra_se for pbox_extra_se in pbox_extra_se_list
                                  if pbox_extra_se not in to_remove_pbox_extra_se_list]
        # pbox_extra_se_list is NOT modified
        return True, um_pbox_xy_list, out_pbox_extra_se_list

    return False, um_pbox_xy_list, pbox_extra_se_list


def setup_pdfbox_xy_maps(pbox_page, doc_text: str) \
    -> Tuple[Dict[Tuple[int, int], UnsyncedPBoxLine],
             StrMappedTracker]:
    """Set the data structure for performing matching based on x, y coordinates.

    pbox_xy_map: the coordinates all str in pdfbox's output
    str_mapped_tracker: all the xy pairs in pdfbox.  Will be used to figure
                        which str are not mapped.
    """
    # pylint: disable=line-too-long
    pbox_xy_map = OrderedDict()  # type: Dict[Tuple[int, int], UnsyncedPBoxLine]
    str_mapped_tracker = StrMappedTracker()
    for pblockinfo in pbox_page.pblockinfo_list:
        # print('\n    pbox block ---------------------------')
        for lineinfo in pblockinfo.lineinfo_list:
            for strinfo in lineinfo.strinfo_list:
                start = strinfo.start
                end = strinfo.end
                multiplier = 300.0 / 72
                x = int(strinfo.xStart * multiplier)
                y = int(strinfo.yStart * multiplier)
                str_text = doc_text[start:end]
                # pylint: disable=line-too-long
                # print("        strinfo se={}, x,y={}    [{}]".format((start, end), (x, y), doc_text[start:end]))

                # strinfo_list.append(((start, end), (x, y), doc_text[start:end]))
                xy_pair = (x, y)
                pbox_xy_map[xy_pair] = UnsyncedPBoxLine(xy_pair, (start, end), str_text)
                str_mapped_tracker.add(xy_pair)
    return pbox_xy_map, str_mapped_tracker


def find_abby_extra_in_pbox_xy_list_by_y_strmatch(abby_ypoint: int,
                                                  extra_text: str,
                                                  um_pbox_xy_list: List[Tuple[int, int]],
                                                  pbox_xy_map: Dict[Tuple[int, int],
                                                                    UnsyncedPBoxLine]) \
                                                  -> Optional[Tuple[Tuple[int, int],
                                                                    Tuple[int, int]]]:
    """Find 'fully synced' abby extra se in um_pbox_xy_list.

    The requirement is
        - the abby_y matches an unused_pbox_xy
        - the abby extra text matched an unused_pdbox text

    Returns:
        A tuple of two int pairs
          -  first tuple: pbox start, end
          - second tuple: pbox x, y
    """
    match_y_list = []  # type: List[Tuple[Tuple[int, int], Tuple[int, int]]]
    for xypair in um_pbox_xy_list:
        pbox_xypair, se_pair, pbox_text = pbox_xy_map[xypair].to_tuple()
        unused_pbox_x, pbox_y = pbox_xypair

        if pbox_y == 491 and se_pair == (44287, 44309):
            print("heterewtwewr")

        if pbox_text.startswith("CHARLES J."):
            print("heterewtwewr2")

        # use y coordinate and extract string to perform the matching
        # This probably is too strict:
        #   if pbox_y == abby_ypoint and extra_text == pbox_text:
        if pbox_y - Y_BOX_CHECK_MIN_2 <= abby_ypoint and \
           abby_ypoint < pbox_y + Y_BOX_CHECK_MAX:

            abby_pbox_offset_mapper = AlignedStrMapper(extra_text,
                                                       pbox_text)
            # please not this is a fully sync, not partial
            if abby_pbox_offset_mapper.is_fully_synced:
                match_y_list.append((se_pair, pbox_xypair))

    # Must only have exact matching only once in a page, otherwise ambiguous and return None.
    if len(match_y_list) == 1:
        return match_y_list[0]
    return None


# This doesn't handle pbox_fragments
# It find only pbox strs with x, y info known
def filter_um_pbox_xy_list_by_y_strmatch(um_pbox_xy_list: List[Tuple[int, int]],
                                         abby_extra_se_list: List[UnsyncedStrWithY],
                                         pbox_xy_map: Dict[Tuple[int, int],
                                                           Tuple[UnsyncedPBoxLine]]) \
                                         -> Tuple[List[Tuple[int, int]],
                                                  List[UnsyncedStrWithY]]:
    """Filter the um_pbox_xy_list by using abby_extra_se_list.

    Returns
        first: filtered um_pbox_xy_list: pbox xy list that are not found.
        second: unmatched_abby_extra_se_list
    """
    if not um_pbox_xy_list:
        return [], abby_extra_se_list

    # store the xy of aligned pbox xy
    found_extra_xypair = []  # type: List[Tuple[int, int]]
    # pylint: disable=line-too-long
    unmatched_abby_extra_se_list = []  # type: List[UnsyncedStrWithY]
    for unused_i, abby_extra_se in enumerate(abby_extra_se_list):
        abby_y, (unused_fstart, unused_fend), extra_text, asmapper = abby_extra_se.to_tuple()
        # logger.debug("  -- extra abby str #%d: xy=%r, %r [%s]",
        # i, (-1, abby_y), (fstart, fend), extra_text)

        found_pbox_se_xypair = \
            find_abby_extra_in_pbox_xy_list_by_y_strmatch(abby_y,
                                                          extra_text,
                                                          um_pbox_xy_list,
                                                          pbox_xy_map)
        if found_pbox_se_xypair:
            found_se, found_xypair = found_pbox_se_xypair
            # add the new found pdfbox offsets to abline.asmapper.from/to_se_list
            asmapper.add_aligned_se_pair(found_se)
            found_extra_xypair.append(found_xypair)
        else:
            unmatched_abby_extra_se_list.append(abby_extra_se)

    filtered_um_pbox_xy_list = [xypair for xypair in um_pbox_xy_list
                                if xypair not in found_extra_xypair]

    return filtered_um_pbox_xy_list, unmatched_abby_extra_se_list


# Note:
# The offset mapping between abby and pdfbox differs only when
# There are '_', '-', or space misalignment for 'aligned' str.

# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def sync_page_offsets(abby_page: AbbyPage,
                      pbox_page: PageInfo3,
                      doc_text: str) -> None:

    # pbox_xy_map: the coordinates all str in pdfbox's output
    # pbox_str_mapped_tracker: all the xy pairs in pdfbox.  Will be used to figure
    #                     which str are not mapped.
    # pylint: disable=line-too-long
    pbox_xy_map, pbox_str_mapped_tracker = \
        setup_pdfbox_xy_maps(pbox_page, doc_text)  # type: Dict[Tuple[int, int], UnsyncedPBoxLine], StrMappedTracker

    unmatched_ablines = []  # type: List[AbbyLine]
    ab_line_list = abbyxmlparser.get_page_abby_lines(abby_page)

    # These are leftover from matched 'str' between abby and pdfbox
    # Because these are left overs, so we don't know the exact x location of those extras.
    # As a result, the first 'int' is the y coordinate only.
    # The 2nd xy pair are the offsets in the pdfbox.
    # The 3rd str is the text in abby or pdfbox
    abby_extra_se_list = []  # type: List[UnsyncedStrWithY]
    pbox_extra_se_list = []  # type: List[UnsyncedStrWithY]

    # Here are all the data structure trying to keep track of what is used in abby and pdfbox
    # abby:
    #     unmatched_ablines
    #     abby_extra_se_list
    # pdfbox:
    #     pbox_str_mapped_tracker: all xy coordinates of used str in pdfbox
    #     um_pbox_xy_list: unmatched pbox xy list
    #     pbox_extra_se_list: leftover of partial matched pdfbox str
    #
    # All the matched or aligned strs will have synched
    #     - abline.abby_pbox_offset_mapper.from_se_list
    #     - abline.abby_pbox_offset_mapper.to_selist

    # Go through all abby lines and find the corresponding pdfbox offsets.
    #   page_unmatch_ab_lines will have all abline that's not even partially matched.
    #       For those partially matched ablines, their from/to_se_list are set.
    #   pbox_str_mapped_tracker will have all partially aligned pbox strs
    #   abby_extra_se_list will have the extra abby str's
    #   pbox_extra_se_list will have the extra pbox str's
    for ab_line in ab_line_list:
        um_pbox_line = find_um_pbox_line_by_xy(ab_line.infer_attr_dict['x'],
                                               ab_line.infer_attr_dict['y'],
                                               pbox_xy_map)
        if um_pbox_line:
            # found X, Y that mached in pdfbox
            xypair, (start, unused_end), pbox_text = um_pbox_line.to_tuple()

            asmapper = AlignedStrMapper(ab_line.text,
                                        pbox_text,
                                        start)

            # *.is_aligned means that some prefix match is found, but
            # not be complete match.
            if asmapper.is_aligned:
                # setting the abline.*.from/to_se_list
                ab_line.abby_pbox_offset_mapper = asmapper
                # as long as there is a partial match, mark
                # the pdfbox str x, y as used.  (NOTE, not abby's str)
                pbox_str_mapped_tracker.set_used(xypair)
                if not asmapper.is_fully_synced:
                    # logger.info("not fully synced: %s", str(asmapper))
                    if asmapper.extra_fse:  # abby has this, but not pdfbox
                        fstart, fend = asmapper.extra_fse
                        abby_extra_se_list.append(UnsyncedStrWithY(xypair[-1],
                                                                   asmapper.extra_fse,
                                                                   ab_line.text[fstart:fend],
                                                                   ab_line.abby_pbox_offset_mapper))
                    if asmapper.extra_tse:  # pdfbox has this, but not abby
                        tstart, tend = asmapper.extra_tse
                        pbox_extra_se_list.append(UnsyncedStrWithY(xypair[-1],
                                                                   asmapper.extra_tse,
                                                                   pbox_text[tstart-start:tend-start],
                                                                   ab_line.abby_pbox_offset_mapper))
            else:
                if IS_DEBUG_SYNC_L2:
                    # This should be very rare, and why
                    print('  - found pdfbox str by x, y, but still not matching:')
                    print("        abline [%s]", ab_line.text)
                    print("          pbox [%s]", pbox_text)
                unmatched_ablines.append(ab_line)
        else:
            # Cannot find the correponding pdfbox str by x, y coordinate.
            # There are 2 reason for this to occur:
            #    - the difference in Y is too large
            #    - the abby str is pointing to the middle pdfbox str
            #          pdfbox str: 'I16606'
            #          abby str: 'I' and '16606'
            #    - In the above case, the abby str, '16606', will not have matching
            #      pdfbox str.
            if IS_DEBUG_SYNC_L2:
                print("  - cannot find pdfbox str for ab_line by x, y: {}".format(ab_line))
            # print("adding unmatched_ablines... page {}, {}".format(pbox_page.page_num,
            #                                                   ab_line))
            # raise Exception("cannot find ab_line '%r' in pbox" % (ab_line, ))
            unmatched_ablines.append(ab_line)

    unmatched_pbox_xy_list = pbox_str_mapped_tracker.get_unused_xy_list()
    # TODO, I believe um_pbox_lines should be used to replace
    # unmatched_pbox_xy_list and pbox_xy_map here, but
    # will deal with this later.
    if IS_DEBUG_SYNC:
        um_pbox_lines = pbox_xy_list_to_pbox_lines(unmatched_pbox_xy_list,
                                                   pbox_xy_map)
        if not unmatched_ablines and \
           not abby_extra_se_list and \
           not um_pbox_lines and \
           not pbox_extra_se_list:
            return

        print("\n\n====== print_unsynced, page #{}====\n".format(pbox_page.page_num))
        print("\n----- {}\n".format("after sync on abby_line xy with pbox_xy_list, has abby_frags, pbox_frags"))
        print_abby_page_unsynced_aux(unmatched_ablines,
                                     abby_extra_se_list,
                                     um_pbox_lines,
                                     pbox_extra_se_list)
        print('^^^^^\n')

    if pbox_page.page_num == 16:
        print("hello")

    # filter unmatched_pbox_xy_list by applying filter using abby_extra_se_list
    # This only handle pbox_lines, not pbox_frags
    unmatched_pbox_xy_list, abby_extra_se_list = \
        filter_um_pbox_xy_list_by_y_strmatch(unmatched_pbox_xy_list,
                                             abby_extra_se_list,
                                             pbox_xy_map)
    if IS_DEBUG_SYNC:
        print("\n----- {}\n".format("after sync pbox_lines by abby_frags, y_strmatch"))
        um_pbox_lines = pbox_xy_list_to_pbox_lines(unmatched_pbox_xy_list,
                                                   pbox_xy_map)
        print_abby_page_unsynced_aux(unmatched_ablines,
                                     abby_extra_se_list,
                                     um_pbox_lines,
                                     pbox_extra_se_list)
        print('^^^^^\n')

    # At this point,
    #   unmatched_pbox_xy_list has all pbox str that's not found by
    #     abline xy, partial match
    #     abline extra str, based on y and complete match
    # We have done all we can looking at things from pdfbox's str
    # list view point

    # try match abby_lines with both pbox_line and pbox_frags by
    #   1. y_strmatch
    #   2. strmatch only
    #
    # Look for sync'ing from Abby's str list view point
    if unmatched_ablines:
        out_unmatched_ablines = []  # type: List[AbbyLine]
        for um_abline in unmatched_ablines:
            is_um_abline_found, unmatched_pbox_xy_list, pbox_extra_se_list = \
                find_um_abline_in_pbox_strs_by_y_strmatch(um_abline,
                                                          unmatched_pbox_xy_list,
                                                          pbox_xy_map,
                                                          pbox_extra_se_list)
            if not is_um_abline_found:
                is_um_abline_found, unmatched_pbox_xy_list, pbox_extra_se_list = \
                    find_um_abline_in_pbox_strs_by_y_strmatch(um_abline,
                                                              unmatched_pbox_xy_list,
                                                              pbox_xy_map,
                                                              pbox_extra_se_list,
                                                              is_skip_y_check=True)
                if not is_um_abline_found:
                    out_unmatched_ablines.append(um_abline)
        unmatched_ablines = out_unmatched_ablines

    if IS_DEBUG_SYNC:
        print("\n----- {}\n".format("after synch abby_lines with pbox_lines and pbox_frag by y_strmatch & strmatch"))
        um_pbox_lines = pbox_xy_list_to_pbox_lines(unmatched_pbox_xy_list,
                                                   pbox_xy_map)
        print_abby_page_unsynced_aux(unmatched_ablines,
                                     abby_extra_se_list,
                                     um_pbox_lines,
                                     pbox_extra_se_list)
        print('^^^^^\n')

    # maybe remove in future, but still not sure if this is
    # ever needed.
    # pylint: disable=pointless-string-statement
    """
    # abbydoc.unmatched_ablines are not found in pbox_doc
    # Now try to use more expensive string matching to find
    # those str's in pdfbox in unmatched_ablines
    # try to find missing
    if unmatched_pbox_xy_list:
        um_abline_fromto_selist_map = defaultdict(list)
        to_remove_unmatched_pbox_xy_list = []  # type: List[Tuple[int, int]]
        for xypair in unmatched_pbox_xy_list:
            unused_xypair2, se_pair, stext = pbox_xy_map[xypair]

            mat_unmatched_abline = find_unique_str_in_unmatched_ablines(stext,
                                                                        unmatched_ablines)
            if mat_unmatched_abline:
                mat, unmatched_abline = mat_unmatched_abline
                um_abline_fromto_selist_map[unmatched_abline].append(((mat.start(), mat.end()),
                                                                      se_pair))
                to_remove_unmatched_pbox_xy_list.append(xypair)

        # add fromto_selist to abline, so it is set up correctly
        # remove unmatched_abline
        for um_abline, fromto_selist in um_abline_fromto_selist_map.items():
            # now remove it
            unmatched_ablines.remove(um_abline)
            # note, the semantics for alignedstr.make_aligned_str_mapper() is changed.
            um_abline.abby_pbox_offset_mapper = \
                alignedstr.make_aligned_str_mapper(sorted(fromto_selist))
        for xypair in to_remove_unmatched_pbox_xy_list:
            unmatched_pbox_xy_list.remove(xypair)

    if unmatched_pbox_xy_list:
        logger.debug("unused pbox strs in page #%d", pbox_page.page_num)
        for xypair in unmatched_pbox_xy_list:
            unused_xypair2, se_pair, stext = pbox_xy_map[xypair]
            logger.debug("    unused str: xy=%r, %r [%s]", xypair, se_pair, stext)
    """

    # now record unsync stuff, per page
    abby_page.unsync_abby_lines = unmatched_ablines
    abby_page.unsync_abby_frags = abby_extra_se_list
    abby_page.unsync_pbox_lines = pbox_xy_list_to_pbox_lines(unmatched_pbox_xy_list,
                                                             pbox_xy_map)
    abby_page.unsync_pbox_frags = pbox_extra_se_list

    if IS_DEBUG_SYNC:
        print("\n----- {}\n".format("after page sync finished 33540"))
        print_abby_page_unsynced(abby_page,
                                 file=sys.stdout)
        print('^^^^^\n')


def sync_doc_offsets(abby_doc: AbbyXmlDoc,
                     pbox_doc: PDFTextDoc) -> None:
    """Update lines in abb_doc withe offsets in pbox_doc lines.

    The synching process does ONE page at a time.  Because of this 1
    page limit, there are opportunities to perform stray matching
    based on the fact that there are only limited unmatched candidates
    in a page.
    """
    # doc_text = pbox_doc.doc_text
    # for special hyphen char
    doc_text = re.sub('[­¬]', '-', pbox_doc.doc_text)
    for page_num, ab_page in enumerate(abby_doc.ab_pages):
        pbox_page = pbox_doc.page_list[page_num]
        sync_page_offsets(ab_page,
                          pbox_page,
                          doc_text)


def print_abby_pbox_unsync(abby_doc: AbbyXmlDoc,
                           file: TextIO = sys.stdout) -> int:
    count_diff = 0
    for abby_page in abby_doc.ab_pages:
        if abby_page.has_unsynced_strs():
            print("\n========= page  #{:3d} ========".format(abby_page.num), file=file)

        count_diff += print_abby_page_unsynced(abby_page,
                                               file=file)
    if count_diff:
        print('{} has {} unsynched strs'.format(abby_doc.file_id, count_diff))
    else:
        print('All syched.')
    return count_diff


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def print_abby_pbox_sync(abby_doc: AbbyXmlDoc,
                         doc_text: str,
                         file: TextIO) -> None:
    count_diff = 0
    out_st_buf = []  # type: List[str]
    # pylint: disable=too-many-nested-blocks
    for abby_page in abby_doc.ab_pages:

        print("========= page  #{:3d} ========".format(abby_page.num), file=file)

        for ab_block in abby_page.ab_blocks:

            if isinstance(ab_block, AbbyTextBlock):
                ab_text_block = ab_block
                for ab_par in ab_text_block.ab_pars:

                    for ab_line in ab_par.ab_lines:
                        st_list = []
                        amapper = ab_line.abby_pbox_offset_mapper
                        print("\nabby_line: [{}]".format(ab_line.text), file=file)

                        if amapper:
                            print("from_se_list: {}".format(amapper.from_se_list), file=file)
                            print("  to_se_list: {}".format(amapper.to_se_list), file=file)

                            for start, end in amapper.from_se_list:
                                to_start = amapper.get_to_offset(start)
                                to_end = amapper.get_to_offset(end)
                                st_list.append(doc_text[to_start:to_end])
                            to_st = ''.join(st_list)

                            if ab_line.text == to_st:
                                print("pbox_line: [{}]".format(to_st), file=file)
                                # pass
                            else:
                                # slight differ due to space or '_', not important enough
                                # print("\nabby_line: [{}]".format(ab_line.text), file=file)
                                print("pbox_line: [{}]".format(to_st), file=file)
                                # count_diff += 1
                                # pass
                        else:
                            # pylint: disable=line-too-long
                            out_st_buf.append("\n#{}, page {}, text ab_line: [{}]".format(count_diff,
                                                                                          abby_page.num,
                                                                                          ab_line.text))
                            count_diff += 1
            elif isinstance(ab_block, AbbyTableBlock):
                ab_table_block = ab_block

                for ab_row in ab_table_block.ab_rows:
                    for ab_cell in ab_row.ab_cells:
                        for ab_par in ab_cell.ab_pars:

                            for ab_line in ab_par.ab_lines:
                                amapper = ab_line.abby_pbox_offset_mapper
                                print("\ntable abby_line: [{}]".format(ab_line.text),
                                      file=file)

                                if amapper:
                                    print("table from_se_list: {}".format(amapper.from_se_list),
                                          file=file)
                                    print("table   to_se_list: {}".format(amapper.to_se_list),
                                          file=file)

                                    st_list = []
                                    for to_start, to_end in amapper.to_se_list:
                                        st_list.append(doc_text[to_start:to_end])
                                    pbox_st = ''.join(st_list)

                                    st_list = []
                                    for start, end in amapper.from_se_list:
                                        to_start = amapper.get_to_offset(start)
                                        to_end = amapper.get_to_offset(end)
                                        st_list.append(doc_text[to_start:to_end])
                                    abby_st = ''.join(st_list)

                                    # if ab_line.text == to_st:
                                    if abby_st == pbox_st:
                                        print("table pbox_line: [{}]".format(abby_st),
                                              file=file)
                                        # pass
                                    else:
                                        # slight differ due to space or '_', not important enough
                                        # out_st_buf.append("\ntable ab_line: [{}]".format(abby_st))
                                        # out_st_buf.append("table   line: [{}]".format(pbox_st))
                                        # count_diff += 1
                                        pass
                                else:
                                    # pylint: disable=line-too-long
                                    out_st_buf.append("\n#{}, page {}, table ab_line: [{}]".format(count_diff,
                                                                                                   abby_page.num,
                                                                                                   ab_line.text))
                                    print("\ntable ab_line: [{}]".format(ab_line.text),
                                          file=file)
                                    print("table ---Not found in PDFBox---", file=file)
                                    count_diff += 1

    if count_diff:
        print("\n\n===== Abby lines not found in pdfbox =====", file=file)
        print("\n".join(out_st_buf), file=file)
        print("\ncount_diff = {}".format(count_diff), file=file)
