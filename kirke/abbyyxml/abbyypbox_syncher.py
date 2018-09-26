#!/usr/bin/env python3
# pylint: disable=too-many-lines

from collections import OrderedDict
import logging
import re
import sys
# pylint: disable=unused-import
from typing import Dict, List, Match, Optional, TextIO, Tuple

from kirke.abbyyxml import abbyyxmlparser
from kirke.abbyyxml.pdfoffsets import AbbyyLine, AbbyyPage, UnsyncedPBoxLine, UnsyncedStrWithY
from kirke.abbyyxml.pdfoffsets import AbbyyTableBlock, AbbyyTextBlock, AbbyyXmlDoc
from kirke.abbyyxml.pdfoffsets import print_abbyy_page_unsynced, print_abbyy_page_unsynced_aux
from kirke.docstruct.pdfoffsets import PDFTextDoc, PageInfo3
from kirke.utils import alignedstr, strutils
from kirke.utils.alignedstr import AlignedStrMapper, MatchedStrMapper

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARN)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

IS_DEBUG_SYNC = False
# IS_DEBUG_SYNC = True

# level 2, sync debug, more detailed info
IS_DEBUG_SYNC_L2 = False

# This is just to print out the unsync information
IS_DEBUG_UNSYNC = False

IS_DEBUG_XY_DIFF = False

X_BOX_CHECK_MIN = 2
X_BOX_CHECK_MAX = 3

Y_BOX_CHECK_MIN_1 = 12
# Y_BOX_CHECK_MIN_2 = 21
Y_BOX_CHECK_MIN_2 = 38
Y_BOX_CHECK_MAX = 3
# found some instance of this, abbyy y = 1344, pbox y= 1336, a diff of 8
Y_BOX_CHECK_MAX_2 = 12

# remove all frags with only special chars
# -_.
IS_REMOVE_SPECIAL_CHAR_ONLY_FRAGS = True

# pylint: disable=invalid-name
def find_unsync_pbox_line_by_xy(x: int,
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
                    print("jjjdiff1 abbyy-pbox:\t{}\t{}".format(x - tmp_x, y - tmp_y))
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
                    print("jjjdiff2 abbyy-pbox:\t{}\t{}".format(x - tmp_x, y - tmp_y))
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
                                         unmatched_ablines: List[AbbyyLine]) \
                                         -> Optional[Tuple[Match[str],
                                                           AbbyyLine]]:
    mats_um_abline_list = []  # type: List[Tuple[List[Match[str]], AbbyyLine]]
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

SPECIAL_CHAR_ONLY_PAT = re.compile(r'^[\-_\. ]+$')

def remove_special_char_only_frags(frag_list: List[UnsyncedStrWithY]) \
    -> List[UnsyncedStrWithY]:
    out_list = []  # type: List[UnsyncedStrWithY]
    for frag in frag_list:
        unused_frag_y, (unused_start, unused_end), frag_text, unused_as_mapper = frag.to_tuple()

        if not SPECIAL_CHAR_ONLY_PAT.match(frag_text):
            out_list.append(frag)
    return out_list


def remove_special_char_only_pbox_lines(pbox_lines: List[UnsyncedPBoxLine]) \
    -> List[UnsyncedPBoxLine]:
    out_list = []  # type: List[UnsyncedPBoxLine]
    for pbox_line in pbox_lines:
        unused_xypair, (unused_start, unused_end), pbox_line_text = pbox_line.to_tuple()

        if not SPECIAL_CHAR_ONLY_PAT.match(pbox_line_text):
            out_list.append(pbox_line)
    return out_list


def remove_special_char_only_abbyy_lines(abbyy_lines: List[AbbyyLine]) \
    -> List[AbbyyLine]:
    out_list = []  # type: List[AbbyyLine]
    for abbyy_line in abbyy_lines:
        if not SPECIAL_CHAR_ONLY_PAT.match(abbyy_line.text):
            out_list.append(abbyy_line)
    return out_list


def pbox_xy_list_to_pbox_lines(um_pbox_xy_list: List[Tuple[int, int]],
                               pbox_xy_map: Dict[Tuple[int, int],
                                                 UnsyncedPBoxLine]) \
                                                 -> List[UnsyncedPBoxLine]:
    return [pbox_xy_map[um_pbox_xy]
            for um_pbox_xy in um_pbox_xy_list]


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def find_unsync_abline_in_pbox_strs_by_y_align_str(unsync_abbyy_line: AbbyyLine,
                                                   unsync_pbox_lines: List[UnsyncedPBoxLine],
                                                   pbox_extra_se_list: List[UnsyncedStrWithY],
                                                   # not used for sync'ing, only for store unsync'ed
                                                   abbyy_extra_se_list: List[UnsyncedStrWithY],
                                                   is_skip_y_check: bool = False) \
                                                   -> Tuple[bool,
                                                            List[UnsyncedPBoxLine],
                                                            List[UnsyncedStrWithY],
                                                            List[UnsyncedStrWithY]]:
    # pylint: disable=line-too-long
    as_mapper_list = []  # type: List[AlignedStrMapper]
    out_abbyy_frag, out_pbox_frag = None, None
    out_pbox_lines = []  # type: List[UnsyncedPBoxLine]

    abline_st = unsync_abbyy_line.text
    um_abline_y = unsync_abbyy_line.infer_attr_dict['y']
    for um_pbox_line in unsync_pbox_lines:
        xypair, (pstart, unused_to_end), pbox_text = um_pbox_line.to_tuple()
        unused_pbox_x, pbox_y = xypair

        if is_skip_y_check or \
           (pbox_y - Y_BOX_CHECK_MIN_2 <= um_abline_y and
            um_abline_y < pbox_y + Y_BOX_CHECK_MAX_2):

            asmapper = AlignedStrMapper(abline_st,
                                        pbox_text,
                                        pstart)

            if asmapper.is_aligned:
                as_mapper_list.append(asmapper)

                if IS_DEBUG_SYNC:
                    if is_skip_y_check:
                        print("aligned matched 3 !! abline_st [{}]".format(abline_st))
                    else:
                        print("aligned matched 1 !! abline_st [{}]".format(abline_st))
                    print("                     pdfbox_st [{}]".format(pbox_text))

                if not asmapper.is_fully_synced:
                    # logger.info("not fully synced: %s", str(asmapper))
                    if asmapper.extra_fse:  # abbyy has this, but not pdfbox
                        fstart, fend = asmapper.extra_fse
                        out_abbyy_frag = UnsyncedStrWithY(um_abline_y,
                                                          asmapper.extra_fse,
                                                          abline_st[fstart:fend],
                                                          asmapper)
                    if asmapper.extra_tse:  # pdfbox has this, but not abbyy
                        tstart, tend = asmapper.extra_tse
                        out_pbox_frag = UnsyncedStrWithY(pbox_y,
                                                         asmapper.extra_tse,
                                                         pbox_text[tstart-pstart:tend-pstart],
                                                         asmapper)
            else:
                out_pbox_lines.append(um_pbox_line)
        else:
            out_pbox_lines.append(um_pbox_line)

    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(as_mapper_list) == 1:
        unsync_abbyy_line.abbyy_pbox_offset_mapper = as_mapper_list[0]
        if out_abbyy_frag:
            abbyy_extra_se_list.append(out_abbyy_frag)
        if out_pbox_frag:
            pbox_extra_se_list.append(out_pbox_frag)
        # pbox_extra_se_list is NOT modified
        return True, out_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list

    as_mapper_list = []
    out_abbyy_frag, out_pbox_frag = None, None
    out_pbox_extra_se_list = []  # type: List[UnsyncedStrWithY]
    for unused_i, pbox_extra_se in enumerate(pbox_extra_se_list):
        pbox_y, (pstart, unused_end), pbox_text, unused_as_mapper = pbox_extra_se.to_tuple()
        # print("   extra_pbox_str #{}: {} ({}, {}) [{}]".format(i, pbox_y, start, end, pbox_text))

        if is_skip_y_check or \
           (pbox_y - Y_BOX_CHECK_MIN_2 <= um_abline_y and
            um_abline_y < pbox_y + Y_BOX_CHECK_MAX_2):

            asmapper = AlignedStrMapper(abline_st,
                                        pbox_text,
                                        pstart)

            if asmapper.is_aligned:
                as_mapper_list.append(asmapper)

                if IS_DEBUG_SYNC:
                    if is_skip_y_check:
                        print("aligned matched 4 !! abline_st [{}]".format(abline_st))
                    else:
                        print("aligned matched 2 !! abline_st [{}]".format(abline_st))
                    print("                     pdfbox_st [{}]".format(pbox_text))

                if not asmapper.is_fully_synced:
                    # logger.info("not fully synced: %s", str(asmapper))
                    if asmapper.extra_fse:  # abbyy has this, but not pdfbox
                        fstart, fend = asmapper.extra_fse
                        out_abbyy_frag = UnsyncedStrWithY(um_abline_y,
                                                          asmapper.extra_fse,
                                                          abline_st[fstart:fend],
                                                          asmapper)
                    if asmapper.extra_tse:  # pdfbox has this, but not abbyy
                        tstart, tend = asmapper.extra_tse
                        out_pbox_frag = UnsyncedStrWithY(pbox_y,
                                                         asmapper.extra_tse,
                                                         pbox_text[tstart-pstart:tend-pstart],
                                                         asmapper)
            else:
                out_pbox_extra_se_list.append(pbox_extra_se)
        else:
            out_pbox_extra_se_list.append(pbox_extra_se)

    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(as_mapper_list) == 1:
        unsync_abbyy_line.abbyy_pbox_offset_mapper = as_mapper_list[0]
        if out_abbyy_frag:
            abbyy_extra_se_list.append(out_abbyy_frag)
        if out_pbox_frag:
            out_pbox_extra_se_list.append(out_pbox_frag)
        # pbox_extra_se_list is NOT modified
        return True, unsync_pbox_lines, abbyy_extra_se_list, out_pbox_extra_se_list

    return False, unsync_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list



# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def find_unsync_abline_in_pbox_strs_by_match_str(unsync_abbyy_line: AbbyyLine,
                                                 unsync_pbox_lines: List[UnsyncedPBoxLine],
                                                 pbox_extra_se_list: List[UnsyncedStrWithY],
                                                 # not used for sync'ing, only for store unsync'ed
                                                 abbyy_extra_se_list: List[UnsyncedStrWithY],
                                                 allow_partial_match: bool = False) \
                                                 -> Tuple[bool,
                                                          List[UnsyncedPBoxLine],
                                                          List[UnsyncedStrWithY],
                                                          List[UnsyncedStrWithY]]:
    # pylint: disable=line-too-long
    as_mapper_list = []  # type: List[MatchedStrMapper]
    out_abbyy_frag, out_pbox_frag = None, None
    out_pbox_lines = []  # type: List[UnsyncedPBoxLine]

    abline_st = unsync_abbyy_line.text
    um_abline_y = unsync_abbyy_line.infer_attr_dict['y']

    for um_pbox_line in unsync_pbox_lines:
        xypair, (pstart, unused_to_end), pbox_text = um_pbox_line.to_tuple()
        unused_pbox_x, pbox_y = xypair

        asmapper = MatchedStrMapper(abline_st,
                                    pbox_text,
                                    pstart)

        if asmapper.is_fully_synced:
            as_mapper_list.append(asmapper)
        elif allow_partial_match and asmapper.is_aligned:
            as_mapper_list.append(asmapper)

            if IS_DEBUG_SYNC:
                print("matched str 1 !! abline_st [{}]".format(abline_st))
                print("                 pdfbox_st [{}]".format(pbox_text))

            if not asmapper.is_fully_synced:
                # logger.info("not fully synced: %s", str(asmapper))
                if asmapper.extra_fse:  # abbyy has this, but not pdfbox
                    fstart, fend = asmapper.extra_fse
                    out_abbyy_frag = UnsyncedStrWithY(um_abline_y,
                                                      asmapper.extra_fse,
                                                      abline_st[fstart:fend],
                                                      asmapper)
                if asmapper.extra_tse:  # pdfbox has this, but not abbyy
                    tstart, tend = asmapper.extra_tse
                    out_pbox_frag = UnsyncedStrWithY(pbox_y,
                                                     asmapper.extra_tse,
                                                     pbox_text[tstart-pstart:tend-pstart],
                                                     asmapper)
        else:
            out_pbox_lines.append(um_pbox_line)


    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(as_mapper_list) == 1:
        unsync_abbyy_line.abbyy_pbox_offset_mapper = as_mapper_list[0]
        if out_abbyy_frag:
            abbyy_extra_se_list.append(out_abbyy_frag)
        if out_pbox_frag:
            pbox_extra_se_list.append(out_pbox_frag)
        # pbox_extra_se_list is NOT modified
        return True, out_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list

    as_mapper_list = []
    out_abbyy_frag_list, out_pbox_frag_list = [], []
    out_pbox_extra_se_list = []  # type: List[UnsyncedStrWithY]
    for unused_i, pbox_extra_se in enumerate(pbox_extra_se_list):
        pbox_y, (pstart, unused_end), pbox_text, unused_as_mapper = pbox_extra_se.to_tuple()
        # print("   extra_pbox_str #{}: {} ({}, {}) [{}]".format(i, pbox_y, start, end, pbox_text))

        asmapper = MatchedStrMapper(abline_st,
                                    pbox_text,
                                    pstart)

        if asmapper.is_fully_synced:
            as_mapper_list.append(asmapper)
        elif allow_partial_match and asmapper.is_aligned:
            as_mapper_list.append(asmapper)

            if IS_DEBUG_SYNC:
                print("matched str 2 !! abline_st [{}]".format(abline_st))
                print("                 pdfbox_st [{}]".format(pbox_text))

            if not asmapper.is_fully_synced:
                # logger.info("not fully synced: %s", str(asmapper))
                if asmapper.extra_fse_list:  # abbyy has this, but not pdfbox
                    for extra_fse in asmapper.extra_fse_list:
                        fstart, fend = extra_fse
                        out_abbyy_frag = UnsyncedStrWithY(um_abline_y,
                                                          extra_fse,
                                                          abline_st[fstart:fend],
                                                          asmapper)
                        out_abbyy_frag_list.append(out_abbyy_frag)
                if asmapper.extra_tse_list:  # pdfbox has this, but not abbyy
                    for extra_tse in asmapper.extra_tse_list:
                        tstart, tend = extra_tse
                        out_pbox_frag = UnsyncedStrWithY(pbox_y,
                                                         extra_tse,
                                                         pbox_text[tstart-pstart:tend-pstart],
                                                         asmapper)
                        out_pbox_frag_list.append(out_pbox_frag)
        else:
            out_pbox_extra_se_list.append(pbox_extra_se)

    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(as_mapper_list) == 1:
        unsync_abbyy_line.abbyy_pbox_offset_mapper = as_mapper_list[0]
        if out_abbyy_frag_list:
            abbyy_extra_se_list.extend(out_abbyy_frag_list)
        if out_pbox_frag_list:
            out_pbox_extra_se_list.extend(out_pbox_frag_list)
        # pbox_extra_se_list is NOT modified
        return True, unsync_pbox_lines, abbyy_extra_se_list, out_pbox_extra_se_list

    return False, unsync_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list


# pylint: disable=too-many-locals, too-many-statements, too-many-branches
def find_unsync_abline_first_in_pbox_strs_by_match_str(unsync_abbyy_line: AbbyyLine,
                                                       unsync_pbox_lines: List[UnsyncedPBoxLine],
                                                       pbox_extra_se_list: List[UnsyncedStrWithY],
                                                       # not used for sync'ing, only for store unsync'ed
                                                       abbyy_extra_se_list: List[UnsyncedStrWithY],
                                                       allow_partial_match: bool = False) \
                                                       -> Tuple[bool,
                                                                List[UnsyncedPBoxLine],
                                                                List[UnsyncedStrWithY],
                                                                List[UnsyncedStrWithY]]:
    """This is an aggresive version of find_unsync_abline_in_pbox_strs_by_match_str().

    It will return the first match, even if ambiguous match across the unsync_pbox_lines.
    This occurs when PDFBox for some reasons is returning MULTIPLE lines in a table as
    a pdfbox "str" instead of multiple lines or "strs".
    Carousel, p89.  This doesn't always happen,
    but it is so in test environment for the whole Carousel document.  If sending just that
    page, the behavior is as expected.  Not sure why this is the case.  Looked into
    extractor NoIndentXtoText, but didn't help.
    """

    # pylint: disable=line-too-long
    as_mapper_list = []  # type: List[MatchedStrMapper]
    out_abbyy_frag, out_pbox_frag = None, None
    out_pbox_lines = []  # type: List[UnsyncedPBoxLine]

    abline_st = unsync_abbyy_line.text
    um_abline_y = unsync_abbyy_line.infer_attr_dict['y']

    is_found_one = False
    for um_pbox_line in unsync_pbox_lines:
        xypair, (pstart, unused_to_end), pbox_text = um_pbox_line.to_tuple()
        unused_pbox_x, pbox_y = xypair

        asmapper = MatchedStrMapper(abline_st,
                                    pbox_text,
                                    pstart)

        if not is_found_one:
            if asmapper.is_fully_synced:
                as_mapper_list.append(asmapper)
                is_found_found = True
            elif allow_partial_match and asmapper.is_aligned:
                as_mapper_list.append(asmapper)

                if IS_DEBUG_SYNC:
                    print("matched str 13 !! abline_st [{}]".format(abline_st))
                    print("                 pdfbox_st [{}]".format(pbox_text))

                if not asmapper.is_fully_synced:
                    # logger.info("not fully synced: %s", str(asmapper))
                    if asmapper.extra_fse:  # abbyy has this, but not pdfbox
                        fstart, fend = asmapper.extra_fse
                        out_abbyy_frag = UnsyncedStrWithY(um_abline_y,
                                                          asmapper.extra_fse,
                                                          abline_st[fstart:fend],
                                                          asmapper)
                    if asmapper.extra_tse:  # pdfbox has this, but not abbyy
                        tstart, tend = asmapper.extra_tse
                        out_pbox_frag = UnsyncedStrWithY(pbox_y,
                                                         asmapper.extra_tse,
                                                         pbox_text[tstart-pstart:tend-pstart],
                                                         asmapper)
                is_found_found = True
            else:
                out_pbox_lines.append(um_pbox_line)
        else:
            out_pbox_lines.append(um_pbox_line)

    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(as_mapper_list) == 1:
        unsync_abbyy_line.abbyy_pbox_offset_mapper = as_mapper_list[0]
        if out_abbyy_frag:
            abbyy_extra_se_list.append(out_abbyy_frag)
        if out_pbox_frag:
            pbox_extra_se_list.append(out_pbox_frag)
        # pbox_extra_se_list is NOT modified
        return True, out_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list

    as_mapper_list = []
    out_abbyy_frag_list, out_pbox_frag_list = [], []
    out_pbox_extra_se_list = []  # type: List[UnsyncedStrWithY]
    is_found_one = False
    for unused_i, pbox_extra_se in enumerate(pbox_extra_se_list):
        pbox_y, (pstart, unused_end), pbox_text, unused_as_mapper = pbox_extra_se.to_tuple()
        # print("   extra_pbox_str #{}: {} ({}, {}) [{}]".format(i, pbox_y, start, end, pbox_text))

        asmapper = MatchedStrMapper(abline_st,
                                    pbox_text,
                                    pstart)

        if not is_found_one:
            if asmapper.is_fully_synced:
                as_mapper_list.append(asmapper)
                if IS_DEBUG_SYNC:
                    print("matched str 23 !! abline_st [{}]".format(abline_st))
                    print("                 pdfbox_st [{}]".format(pbox_text))
                is_found_one = True
            elif allow_partial_match and asmapper.is_aligned:
                as_mapper_list.append(asmapper)

                if IS_DEBUG_SYNC:
                    print("matched str 24 !! abline_st [{}]".format(abline_st))
                    print("                 pdfbox_st [{}]".format(pbox_text))

                if not asmapper.is_fully_synced:
                    # logger.info("not fully synced: %s", str(asmapper))
                    if asmapper.extra_fse_list:  # abbyy has this, but not pdfbox
                        for extra_fse in asmapper.extra_fse_list:
                            fstart, fend = extra_fse
                            out_abbyy_frag = UnsyncedStrWithY(um_abline_y,
                                                              extra_fse,
                                                              abline_st[fstart:fend],
                                                              asmapper)
                            out_abbyy_frag_list.append(out_abbyy_frag)
                    if asmapper.extra_tse_list:  # pdfbox has this, but not abbyy
                        for extra_tse in asmapper.extra_tse_list:
                            tstart, tend = extra_tse
                            out_pbox_frag = UnsyncedStrWithY(pbox_y,
                                                             extra_tse,
                                                             pbox_text[tstart-pstart:tend-pstart],
                                                             asmapper)
                            out_pbox_frag_list.append(out_pbox_frag)
                is_found_one = True
            else:
                out_pbox_extra_se_list.append(pbox_extra_se)
        else:
            out_pbox_extra_se_list.append(pbox_extra_se)

    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(as_mapper_list) == 1:
        unsync_abbyy_line.abbyy_pbox_offset_mapper = as_mapper_list[0]
        if out_abbyy_frag_list:
            abbyy_extra_se_list.extend(out_abbyy_frag_list)
        if out_pbox_frag_list:
            out_pbox_extra_se_list.extend(out_pbox_frag_list)
        # pbox_extra_se_list is NOT modified
        return True, unsync_pbox_lines, abbyy_extra_se_list, out_pbox_extra_se_list

    return False, unsync_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list


# pylint: disable=too-many-locals
def find_unsync_abbyy_frag_in_pbox_strs_by_match_str(unsync_abbyy_frag: UnsyncedStrWithY,
                                                     unsync_pbox_lines: List[UnsyncedPBoxLine],
                                                     pbox_extra_se_list: List[UnsyncedStrWithY]) \
                                                     -> Tuple[bool,
                                                              List[UnsyncedPBoxLine],
                                                              List[UnsyncedStrWithY],
                                                              List[UnsyncedStrWithY]]:

    abbyy_more_extra_frags = []  # type: List[UnsyncedStrWithY]

    # pylint: disable=line-too-long
    as_mapper_list = []  # type: List[MatchedStrMapper]
    out_abbyy_frag_list, out_pbox_frag_list = [], []
    out_pbox_lines = []  # type: List[UnsyncedPBoxLine]

    abline_st = unsync_abbyy_frag.text
    um_abline_y = unsync_abbyy_frag.y_val
    frag_start = unsync_abbyy_frag.se_pair[0]
    abbyy_line_abbyy_pbox_as_mapper = unsync_abbyy_frag.as_mapper
    for um_pbox_line in unsync_pbox_lines:
        xypair, (pstart, unused_to_end), pbox_text = um_pbox_line.to_tuple()
        unused_pbox_x, pbox_y = xypair

        asmapper = MatchedStrMapper(abline_st,
                                    pbox_text,
                                    pstart)

        if asmapper.is_aligned:
            as_mapper_list.append(asmapper)

            if IS_DEBUG_SYNC:
                print("matched str 11 !! abline_st [{}]".format(abline_st))
                print("                  pdfbox_st [{}]".format(pbox_text))

            if not asmapper.is_fully_synced:
                # logger.info("not fully synced: %s", str(asmapper))
                if asmapper.extra_fse_list:  # abbyy has this, but not pdfbox
                    for extra_fse in asmapper.extra_fse_list:
                        fstart, fend = extra_fse
                        out_abbyy_frag = UnsyncedStrWithY(um_abline_y,
                                                          extra_fse,
                                                          abline_st[fstart:fend],
                                                          asmapper)
                        out_abbyy_frag_list.append(out_abbyy_frag)
                if asmapper.extra_tse_list:  # pdfbox has this, but not abbyy
                    for extra_tse in asmapper.extra_tse_list:
                        tstart, tend = extra_tse
                        out_pbox_frag = UnsyncedStrWithY(pbox_y,
                                                         extra_tse,
                                                         pbox_text[tstart-pstart:tend-pstart],
                                                         asmapper)
                        out_pbox_frag_list.append(out_pbox_frag)
        else:
            out_pbox_lines.append(um_pbox_line)


    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(as_mapper_list) == 1:
        matched_str_mapper = as_mapper_list[0]
        adj_from_se_list = alignedstr.adjust_list_offset(matched_str_mapper.from_se_list,
                                                         frag_start)
        adj_to_se_list = matched_str_mapper.to_se_list
        abbyy_line_abbyy_pbox_as_mapper.update_with_mapper(adj_from_se_list,
                                                           adj_to_se_list)
        if out_abbyy_frag_list:
            abbyy_more_extra_frags.extend(out_abbyy_frag_list)
        if out_pbox_frag_list:
            pbox_extra_se_list.extend(out_pbox_frag_list)
        # pbox_extra_se_list is NOT modified
        return True, out_pbox_lines, abbyy_more_extra_frags, pbox_extra_se_list

    as_mapper_list = []
    out_abbyy_frag, out_pbox_frag = None, None
    out_pbox_extra_se_list = []  # type: List[UnsyncedStrWithY]
    for unused_i, pbox_extra_se in enumerate(pbox_extra_se_list):
        pbox_y, (pstart, unused_end), pbox_text, unused_as_mapper = pbox_extra_se.to_tuple()
        # print("   extra_pbox_str #{}: {} ({}, {}) [{}]".format(i, pbox_y, start, end, pbox_text))

        asmapper = MatchedStrMapper(abline_st,
                                    pbox_text,
                                    pstart)

        if asmapper.is_aligned:
            as_mapper_list.append(asmapper)

            if IS_DEBUG_SYNC:
                print("matched str 22 !! abline_st [{}]".format(abline_st))
                print("                  pdfbox_st [{}]".format(pbox_text))

            if not asmapper.is_fully_synced:
                # logger.info("not fully synced: %s", str(asmapper))
                if asmapper.extra_fse_list:  # abbyy has this, but not pdfbox
                    for extra_fse in asmapper.extra_fse_list:
                        fstart, fend = extra_fse
                        out_abbyy_frag = UnsyncedStrWithY(um_abline_y,
                                                          extra_fse,
                                                          abline_st[fstart:fend],
                                                          asmapper)
                        out_abbyy_frag_list.append(out_abbyy_frag)
                if asmapper.extra_tse_list:  # pdfbox has this, but not abbyy
                    for extra_tse in asmapper.extra_tse_list:
                        tstart, tend = extra_tse
                        out_pbox_frag = UnsyncedStrWithY(pbox_y,
                                                         extra_tse,
                                                         pbox_text[tstart-pstart:tend-pstart],
                                                         asmapper)
                        out_pbox_frag_list.append(out_pbox_frag)
        else:
            out_pbox_extra_se_list.append(pbox_extra_se)

    # Must only have only str has the ab_line,
    # otherwise, too ambiguous and return None
    if len(as_mapper_list) == 1:
        matched_str_mapper = as_mapper_list[0]
        adj_from_se_list = alignedstr.adjust_list_offset(matched_str_mapper.from_se_list,
                                                         frag_start)
        adj_to_se_list = matched_str_mapper.to_se_list
        abbyy_line_abbyy_pbox_as_mapper.update_with_mapper(adj_from_se_list,
                                                           adj_to_se_list)
        # abbyy_line_abbyy_pbox_as_mapper.update_with_mapper(matched_str_mapper.from_se_list,
        #                                                  matched_str_mapper.to_se_list)
        if out_abbyy_frag_list:
            abbyy_more_extra_frags.extend(out_abbyy_frag_list)
        if out_pbox_frag_list:
            out_pbox_extra_se_list.extend(out_pbox_frag_list)
        # pbox_extra_se_list is NOT modified
        return True, unsync_pbox_lines, abbyy_more_extra_frags, out_pbox_extra_se_list

    return False, unsync_pbox_lines, abbyy_more_extra_frags, pbox_extra_se_list



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


def find_abbyy_extra_in_pbox_lines_by_y_align_str(abbyy_ypoint: int,
                                                  extra_text: str,
                                                  pbox_lines: List[UnsyncedPBoxLine]) \
                                                  -> Optional[Tuple[AlignedStrMapper,
                                                                    Optional[UnsyncedStrWithY],
                                                                    Optional[UnsyncedStrWithY],
                                                                    UnsyncedPBoxLine]]:
    """Find 'fully synced' abbyy extra se in um_pbox_xy_list.

    The requirement is
        - the abbyy_y matches an unused_pbox_xy
        - the abbyy extra text matched an unused_pdbox text

    Returns:
        A tuple of 4 parts:
          - the matched PBoxLine
          - the abbyy frags
          - the pbox frags
          - the unmatched PBoxLine
    """

    as_mapper_list = []  # type: List[AlignedStrMapper]
    synced_pbox_lines = []  # type: List[UnsyncedPBoxLine]
    out_abbyy_frag = None  # type: Optional[UnsyncedStrWithY]
    out_pbox_frag = None  # type: Optional[UnsyncedStrWithY]
    for pbox_line in pbox_lines:
        pbox_xypair, (pstart, unused_pend), pbox_text = pbox_line.to_tuple()
        unused_pbox_x, pbox_y = pbox_xypair

        # use y coordinate and extract string to perform the matching
        # This probably is too strict:
        #   if pbox_y == abbyy_ypoint and extra_text == pbox_text:
        if pbox_y - Y_BOX_CHECK_MIN_2 <= abbyy_ypoint and \
           abbyy_ypoint < pbox_y + Y_BOX_CHECK_MAX_2:

            asmapper = AlignedStrMapper(extra_text,
                                        pbox_text,
                                        pstart)

            # *.is_aligned means that some prefix match is found, but
            # not be complete match.
            if asmapper.is_aligned:
                as_mapper_list.append(asmapper)
                synced_pbox_lines.append(pbox_line)

                if IS_DEBUG_SYNC:
                    # pylint: disable=line-too-long
                    print("aligned matched, abbyy_extra_in_pbox_lines !! abline_st [{}]".format(extra_text))

                if not asmapper.is_fully_synced:
                    # logger.info("not fully synced: %s", str(asmapper))
                    if asmapper.extra_fse:  # abbyy has this, but not pdfbox
                        fstart, fend = asmapper.extra_fse
                        out_abbyy_frag = UnsyncedStrWithY(abbyy_ypoint,
                                                          asmapper.extra_fse,
                                                          extra_text[fstart:fend],
                                                          asmapper)
                    if asmapper.extra_tse:  # pdfbox has this, but not abbyy
                        tstart, tend = asmapper.extra_tse
                        out_pbox_frag = UnsyncedStrWithY(pbox_y,
                                                         asmapper.extra_tse,
                                                         pbox_text[tstart-pstart:tend-pstart],
                                                         asmapper)

    # Must only have exact matching only once in a page, otherwise ambiguous and return None.
    if len(as_mapper_list) == 1:
        matched_str_mapper = as_mapper_list[0]
        return matched_str_mapper, out_abbyy_frag, out_pbox_frag, synced_pbox_lines[0]
    return None


# This doesn't handle pbox_fragments
# It find only pbox strs with x, y info known
def find_pbox_lines_in_abbyy_frags_by_y_align_str(pbox_lines: List[UnsyncedPBoxLine],
                                                  abbyy_extra_se_list: List[UnsyncedStrWithY],
                                                  pbox_extra_se_list: List[UnsyncedStrWithY]) \
                                                  -> Tuple[List[UnsyncedPBoxLine],
                                                           List[UnsyncedStrWithY],
                                                           List[UnsyncedStrWithY]]:
    """Filter the um_pbox_xy_list by using abbyy_extra_se_list.

    Returns
        first: unsync_pbox_line: pbox lines that are not found.
        second: abbyy_extra_se_list
    """
    if not pbox_lines:
        return [], abbyy_extra_se_list, pbox_extra_se_list

    out_abbyy_frags = []  # type: List[UnsyncedStrWithY]
    out_pbox_frags = list(pbox_extra_se_list)  # type: List[UnsyncedStrWithY]
    to_remove_pbox_lines = []  # type: List[UnsyncedPBoxLine]
    for unused_i, abbyy_extra_se in enumerate(abbyy_extra_se_list):
        abbyy_y, (fstart, unused_fend), extra_text, asmapper = abbyy_extra_se.to_tuple()

        found_pbox_line_tuple = \
            find_abbyy_extra_in_pbox_lines_by_y_align_str(abbyy_y,
                                                          extra_text,
                                                          pbox_lines)
        if found_pbox_line_tuple:
            found_as_mapper, found_abbyy_frag, found_pbox_frag, found_pbox_line = \
                found_pbox_line_tuple

            adj_from_se_list = alignedstr.adjust_list_offset(found_as_mapper.from_se_list,
                                                             fstart)

            # add the new found pdfbox offsets to abline.asmapper.from/to_se_list
            asmapper.update_with_mapper(adj_from_se_list,
                                        found_as_mapper.to_se_list)
            to_remove_pbox_lines.append(found_pbox_line)
            if found_abbyy_frag:
                out_abbyy_frags.append(found_abbyy_frag)
            if found_pbox_frag:
                out_pbox_frags.append(found_pbox_frag)
        else:
            out_abbyy_frags.append(abbyy_extra_se)

    out_pbox_lines = []  # type: List[UnsyncedPBoxLine]
    for pbox_line in pbox_lines:
        if pbox_line not in to_remove_pbox_lines:
            out_pbox_lines.append(pbox_line)

    return out_pbox_lines, out_abbyy_frags, out_pbox_frags


# Note:
# The offset mapping between abbyy and pdfbox differs only when
# There are '_', '-', or space misalignment for 'aligned' str.

# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def sync_page_offsets(abbyy_page: AbbyyPage,
                      pbox_page: PageInfo3,
                      doc_text: str) -> None:

    # pbox_xy_map: the coordinates all str in pdfbox's output
    # pbox_str_mapped_tracker: all the xy pairs in pdfbox.  Will be used to figure
    #                     which str are not mapped.
    # pylint: disable=line-too-long
    pbox_xy_map, pbox_str_mapped_tracker = \
        setup_pdfbox_xy_maps(pbox_page, doc_text)  # type: Dict[Tuple[int, int], UnsyncedPBoxLine], StrMappedTracker

    unmatched_ablines = []  # type: List[AbbyyLine]
    ab_line_list = abbyyxmlparser.get_page_abbyy_lines(abbyy_page)

    """
    if pbox_page.page_num == 79:
        # for debug purpose
        for x, y in pbox_xy_map:
            print("sync_page_offsets, xy = {}, {} || {}".format(x, y, pbox_xy_map[(x, y)]))
        print()
        for ab_line in ab_line_list:
            print("sync_page_offsets, ab_line = {}".format(ab_line))
    """

    # These are leftover from matched 'str' between abbyy and pdfbox
    # Because these are left overs, so we don't know the exact x location of those extras.
    # As a result, the first 'int' is the y coordinate only.
    # The 2nd xy pair are the offsets in the pdfbox.
    # The 3rd str is the text in abbyy or pdfbox
    abbyy_extra_se_list = []  # type: List[UnsyncedStrWithY]
    pbox_extra_se_list = []  # type: List[UnsyncedStrWithY]

    # Here are all the data structure trying to keep track of what is used in abbyy and pdfbox
    # abbyy:
    #     unmatched_ablines
    #     abbyy_extra_se_list
    # pdfbox:
    #     pbox_str_mapped_tracker: all xy coordinates of used str in pdfbox
    #     um_pbox_xy_list: unmatched pbox xy list
    #     pbox_extra_se_list: leftover of partial matched pdfbox str
    #
    # All the matched or aligned strs will have synched
    #     - abline.abbyy_pbox_offset_mapper.from_se_list
    #     - abline.abbyy_pbox_offset_mapper.to_selist

    # Go through all abbyy lines and find the corresponding pdfbox offsets.
    #   page_unmatch_ab_lines will have all abline that's not even partially matched.
    #       For those partially matched ablines, their from/to_se_list are set.
    #   pbox_str_mapped_tracker will have all partially aligned pbox strs
    #   abbyy_extra_se_list will have the extra abbyy str's
    #   pbox_extra_se_list will have the extra pbox str's
    for ab_line in ab_line_list:
        um_pbox_line = find_unsync_pbox_line_by_xy(ab_line.infer_attr_dict['x'],
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
                ab_line.abbyy_pbox_offset_mapper = asmapper
                # as long as there is a partial match, mark
                # the pdfbox str x, y as used.  (NOTE, not abbyy's str)
                pbox_str_mapped_tracker.set_used(xypair)
                if not asmapper.is_fully_synced:
                    # logger.info("not fully synced: %s", str(asmapper))
                    if asmapper.extra_fse:  # abbyy has this, but not pdfbox
                        fstart, fend = asmapper.extra_fse
                        abbyy_extra_se_list.append(UnsyncedStrWithY(xypair[-1],
                                                                    asmapper.extra_fse,
                                                                    ab_line.text[fstart:fend],
                                                                    ab_line.abbyy_pbox_offset_mapper))
                    if asmapper.extra_tse:  # pdfbox has this, but not abbyy
                        tstart, tend = asmapper.extra_tse
                        pbox_extra_se_list.append(UnsyncedStrWithY(xypair[-1],
                                                                   asmapper.extra_tse,
                                                                   pbox_text[tstart - start:tend - start],
                                                                   ab_line.abbyy_pbox_offset_mapper))
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
            #    - the abbyy str is pointing to the middle pdfbox str
            #          pdfbox str: 'I16606'
            #          abbyy str: 'I' and '16606'
            #    - In the above case, the abbyy str, '16606', will not have matching
            #      pdfbox str.
            if IS_DEBUG_SYNC_L2:
                print("  - cannot find pdfbox str for ab_line by x, y: {}".format(ab_line))
            # print("adding unmatched_ablines... page {}, {}".format(pbox_page.page_num,
            #                                                   ab_line))
            # raise Exception("cannot find ab_line '%r' in pbox" % (ab_line, ))
            unmatched_ablines.append(ab_line)

    unsync_pbox_lines = pbox_xy_list_to_pbox_lines(pbox_str_mapped_tracker.get_unused_xy_list(),
                                                   pbox_xy_map)

    # TODO, I believe um_pbox_lines should be used to replace
    # unmatched_pbox_xy_list and pbox_xy_map here, but
    # will deal with this later.
    if IS_DEBUG_SYNC:
        if not unmatched_ablines and \
           not abbyy_extra_se_list and \
           not unsync_pbox_lines and \
           not pbox_extra_se_list:
            return

        print("\n\n====== print_unsynced, page #{}====\n".format(pbox_page.page_num))
        print("\n----- {}\n".format("after sync on abbyy_line xy with pbox_xy_list, has abbyy_frags, pbox_frags"))
        print_abbyy_page_unsynced_aux(unmatched_ablines,
                                      abbyy_extra_se_list,
                                      unsync_pbox_lines,
                                      pbox_extra_se_list)
        print('^^^^^\n')

    # filter unmatched_pbox_line by applying filter using abbyy_extra_se_list.
    # This only handle pbox_lines, not pbox_frags.
    # Because of partial sync'ing, abbyy_frags and pbox_frags might be updated.
    unsync_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list = \
        find_pbox_lines_in_abbyy_frags_by_y_align_str(unsync_pbox_lines,
                                                      abbyy_extra_se_list,
                                                      pbox_extra_se_list)

    if IS_DEBUG_SYNC:
        print("\n----- {}\n".format("after sync pbox_lines by abbyy_frags, y_align_str"))
        print_abbyy_page_unsynced_aux(unmatched_ablines,
                                      abbyy_extra_se_list,
                                      unsync_pbox_lines,
                                      pbox_extra_se_list)
        print('^^^^^\n')

    # At this point,
    #   unmatched_pbox_xy_list has all pbox str that's not found by
    #     abline xy, partial match
    #     abline extra str, based on y and complete match
    # We have done all we can looking at things from pdfbox's str
    # list view point

    # try match abbyy_lines with both pbox_line and pbox_frags by
    #   1. y_align_str
    #   2. strmatch only
    #
    # Look for sync'ing from Abbyy's str list view point
    if unmatched_ablines:
        out_unmatched_ablines = []  # type: List[AbbyyLine]
        for um_abline in unmatched_ablines:
            is_um_abline_found, unsync_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list = \
                find_unsync_abline_in_pbox_strs_by_y_align_str(um_abline,
                                                               unsync_pbox_lines,
                                                               pbox_extra_se_list,
                                                               abbyy_extra_se_list)
            if not is_um_abline_found:
                out_unmatched_ablines.append(um_abline)
        unmatched_ablines = out_unmatched_ablines

    if IS_DEBUG_SYNC:
        print("\n----- {}\n".format("after synch abbyy_lines with pbox_lines and pbox_frag by y_align_str"))
        print_abbyy_page_unsynced_aux(unmatched_ablines,
                                      abbyy_extra_se_list,
                                      unsync_pbox_lines,
                                      pbox_extra_se_list)
        print('^^^^^\n')

    if unmatched_ablines:
        out_unmatched_ablines = []
        for um_abline in unmatched_ablines:
            is_um_abline_found, unsync_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list = \
                find_unsync_abline_in_pbox_strs_by_y_align_str(um_abline,
                                                               unsync_pbox_lines,
                                                               pbox_extra_se_list,
                                                               abbyy_extra_se_list,
                                                               is_skip_y_check=True)
            if not is_um_abline_found:
                out_unmatched_ablines.append(um_abline)
        unmatched_ablines = out_unmatched_ablines


    if IS_DEBUG_SYNC:
        print("\n----- {}\n".format("after synch abbyy_lines with pbox_lines and pbox_frag by alignstr"))
        print_abbyy_page_unsynced_aux(unmatched_ablines,
                                      abbyy_extra_se_list,
                                      unsync_pbox_lines,
                                      pbox_extra_se_list)
        print('^^^^^\n')

    if unmatched_ablines:
        out_unmatched_ablines = []
        for um_abline in unmatched_ablines:
            is_um_abline_found, unsync_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list = \
                find_unsync_abline_in_pbox_strs_by_match_str(um_abline,
                                                             unsync_pbox_lines,
                                                             pbox_extra_se_list,
                                                             abbyy_extra_se_list)
            if not is_um_abline_found:
                out_unmatched_ablines.append(um_abline)
        unmatched_ablines = out_unmatched_ablines

    if unmatched_ablines:
        out_unmatched_ablines = []
        for um_abline in unmatched_ablines:
            is_um_abline_found, unsync_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list = \
                find_unsync_abline_in_pbox_strs_by_match_str(um_abline,
                                                             unsync_pbox_lines,
                                                             pbox_extra_se_list,
                                                             abbyy_extra_se_list,
                                                             allow_partial_match=True)
            if not is_um_abline_found:
                out_unmatched_ablines.append(um_abline)
        unmatched_ablines = out_unmatched_ablines


    if IS_DEBUG_SYNC:
        print("\n----- {}\n".format("after synch abbyy_lines with pbox_lines and pbox_frag by match_str"))
        print_abbyy_page_unsynced_aux(unmatched_ablines,
                                      abbyy_extra_se_list,
                                      unsync_pbox_lines,
                                      pbox_extra_se_list)
        print('^^^^^\n')

    if abbyy_extra_se_list:
        out_abbyy_frag_list = []
        for unsync_frag in list(abbyy_extra_se_list):
            is_unsync_frag_found, unsync_pbox_lines, abbyy_more_extra_frags, pbox_extra_se_list = \
                find_unsync_abbyy_frag_in_pbox_strs_by_match_str(unsync_frag,
                                                                 unsync_pbox_lines,
                                                                 pbox_extra_se_list)

            if not is_unsync_frag_found:
                out_abbyy_frag_list.append(unsync_frag)
            else:
                out_abbyy_frag_list.extend(abbyy_more_extra_frags)

        abbyy_extra_se_list = out_abbyy_frag_list


    if IS_DEBUG_SYNC:
        print("\n----- {}\n".format("after synch abbyy_frags with pbox_lines and pbox_frag by match_str"))
        print_abbyy_page_unsynced_aux(unmatched_ablines,
                                      abbyy_extra_se_list,
                                      unsync_pbox_lines,
                                      pbox_extra_se_list)
        print('^^^^^\n')


    # now do the aggressive version, even if two or more matches are found, take the first one
    # any way.
    if unmatched_ablines:
        out_unmatched_ablines = []
        for um_abline in unmatched_ablines:
            is_um_abline_found, unsync_pbox_lines, abbyy_extra_se_list, pbox_extra_se_list = \
                find_unsync_abline_first_in_pbox_strs_by_match_str(um_abline,
                                                                   unsync_pbox_lines,
                                                                   pbox_extra_se_list,
                                                                   abbyy_extra_se_list,
                                                                   allow_partial_match=True)
            if not is_um_abline_found:
                out_unmatched_ablines.append(um_abline)
        unmatched_ablines = out_unmatched_ablines

    if IS_DEBUG_SYNC:
        print("\n----- {}\n".format("after synch abbyy_lines first  with pbox_lines and pbox_frag by match_str"))
        print_abbyy_page_unsynced_aux(unmatched_ablines,
                                      abbyy_extra_se_list,
                                      unsync_pbox_lines,
                                      pbox_extra_se_list)
        print('^^^^^\n')


    # maybe remove in future, but still not sure if this is
    # ever needed.
    # pylint: disable=pointless-string-statement
    """
    # abbyydoc.unmatched_ablines are not found in pbox_doc
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
            um_abline.abbyy_pbox_offset_mapper = \
                alignedstr.make_aligned_str_mapper(sorted(fromto_selist))
        for xypair in to_remove_unmatched_pbox_xy_list:
            unmatched_pbox_xy_list.remove(xypair)

    if unmatched_pbox_xy_list:
        logger.debug("unused pbox strs in page #%d", pbox_page.page_num)
        for xypair in unmatched_pbox_xy_list:
            unused_xypair2, se_pair, stext = pbox_xy_map[xypair]
            logger.debug("    unused str: xy=%r, %r [%s]", xypair, se_pair, stext)
    """

    if IS_REMOVE_SPECIAL_CHAR_ONLY_FRAGS:
        unmatched_ablines = remove_special_char_only_abbyy_lines(unmatched_ablines)
        abbyy_extra_se_list = remove_special_char_only_frags(abbyy_extra_se_list)
        unsync_pbox_lines = remove_special_char_only_pbox_lines(unsync_pbox_lines)
        pbox_extra_se_list = remove_special_char_only_frags(pbox_extra_se_list)


    # now record unsync stuff, per page
    abbyy_page.unsync_abbyy_lines = unmatched_ablines
    abbyy_page.unsync_abbyy_frags = abbyy_extra_se_list
    abbyy_page.unsync_pbox_lines = unsync_pbox_lines
    abbyy_page.unsync_pbox_frags = pbox_extra_se_list

    if IS_DEBUG_SYNC:
        print("\n----- {}\n".format("after page sync finished 33540"))
        print_abbyy_page_unsynced(abbyy_page,
                                  file=sys.stdout)
        print('^^^^^\n')


def sync_doc_offsets(abbyy_doc: AbbyyXmlDoc,
                     pbox_doc: PDFTextDoc) -> None:
    """Update lines in abbyy_doc withe offsets in pbox_doc lines.

    The synching process does ONE page at a time.  Because of this 1
    page limit, there are opportunities to perform stray matching
    based on the fact that there are only limited unmatched candidates
    in a page.
    """
    # doc_text = pbox_doc.doc_text
    # for special hyphen char
    doc_text = re.sub('[­¬]', '-', pbox_doc.doc_text)
    for page_num, ab_page in enumerate(abbyy_doc.ab_pages):
        pbox_page = pbox_doc.page_list[page_num]
        sync_page_offsets(ab_page,
                          pbox_page,
                          doc_text)


def print_abbyy_pbox_unsync(abbyy_doc: AbbyyXmlDoc,
                            file: TextIO = sys.stdout) -> int:
    count_diff = 0
    for abbyy_page in abbyy_doc.ab_pages:
        if abbyy_page.has_unsynced_strs():
            print("\n========= page  #{:3d} ========".format(abbyy_page.num), file=file)

        count_diff += print_abbyy_page_unsynced(abbyy_page,
                                                file=file)
    if count_diff:
        print('{} has {} unsynched strs'.format(abbyy_doc.file_id, count_diff))
    else:
        print('All synched.')
    return count_diff


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def print_abbyy_pbox_sync(abbyy_doc: AbbyyXmlDoc,
                          doc_text: str,
                          file: TextIO) -> None:
    count_diff = 0
    out_st_buf = []  # type: List[str]
    # pylint: disable=too-many-nested-blocks
    for abbyy_page in abbyy_doc.ab_pages:

        print("========= page  #{:3d} ========".format(abbyy_page.num), file=file)

        for ab_block in abbyy_page.ab_blocks:

            if isinstance(ab_block, AbbyyTextBlock):
                ab_text_block = ab_block
                for ab_par in ab_text_block.ab_pars:

                    for ab_line in ab_par.ab_lines:
                        st_list = []
                        amapper = ab_line.abbyy_pbox_offset_mapper
                        print("\nabbyy_line: [{}]".format(ab_line.text), file=file)

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
                                # print("\nabbyy_line: [{}]".format(ab_line.text), file=file)
                                print("pbox_line: [{}]".format(to_st), file=file)
                                # count_diff += 1
                                # pass
                        else:
                            # pylint: disable=line-too-long
                            out_st_buf.append("\n#{}, page {}, text ab_line: [{}]".format(count_diff,
                                                                                          abbyy_page.num,
                                                                                          ab_line.text))
                            count_diff += 1
            elif isinstance(ab_block, AbbyyTableBlock):
                ab_table_block = ab_block

                for ab_row in ab_table_block.ab_rows:
                    for ab_cell in ab_row.ab_cells:
                        for ab_par in ab_cell.ab_pars:

                            for ab_line in ab_par.ab_lines:
                                amapper = ab_line.abbyy_pbox_offset_mapper
                                print("\ntable abbyy_line: [{}]".format(ab_line.text),
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
                                    abbyy_st = ''.join(st_list)

                                    # if ab_line.text == to_st:
                                    if abbyy_st == pbox_st:
                                        print("table pbox_line: [{}]".format(abbyy_st),
                                              file=file)
                                        # pass
                                    else:
                                        # slight differ due to space or '_', not important enough
                                        # pylint: disable=line-too-long
                                        # out_st_buf.append("\ntable ab_line: [{}]".format(abbyy_st))
                                        # out_st_buf.append("table   line: [{}]".format(pbox_st))
                                        # count_diff += 1
                                        pass
                                else:
                                    # pylint: disable=line-too-long
                                    out_st_buf.append("\n#{}, page {}, table ab_line: [{}]".format(count_diff,
                                                                                                   abbyy_page.num,
                                                                                                   ab_line.text))
                                    print("\ntable ab_line: [{}]".format(ab_line.text),
                                          file=file)
                                    print("table ---Not found in PDFBox---", file=file)
                                    count_diff += 1

    if count_diff:
        print("\n\n===== Abbyy lines not found in pdfbox =====", file=file)
        print("\n".join(out_st_buf), file=file)
        print("\ncount_diff = {}".format(count_diff), file=file)
