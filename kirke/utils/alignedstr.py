from collections import defaultdict
import logging
import re
# pylint: disable=unused-import
from typing import Dict, DefaultDict, List, Optional, Tuple

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IS_DEBUG = False

def is_space_huline(line: str) -> bool:
    return line.isspace() or is_hyphen_underline(line)


def is_hyphen_underline(line: str) -> bool:
    return line == '_' or line == '-' or line == '.'


def is_all_hyphen_underline(line: str) -> bool:
    if not line:
        return False
    for xchar in line:
        if not is_hyphen_underline(xchar):
            return False
    return True


def adjust_list_offset(se_list: List[Tuple[int, int]],
                       offset: int) \
                       -> List[Tuple[int, int]]:
    return [(offset + start, offset + end)
            for start, end in se_list]


def adjust_pair_offset(se_pair: Optional[Tuple[int, int]],
                       offset: int) \
                       -> Optional[Tuple[int, int]]:
    if se_pair:
        return (offset + se_pair[0], offset + se_pair[1])
    return None


def to_str_list(line: str, se_list: List[Tuple[int, int]]) -> List[str]:
    return [line[start:end] for start, end in se_list]


def make_aligned_charmap(line: str) -> Dict[str, int]:
    adict = defaultdict(int)  # type: DefaultDict[str, int]
    for achar in line:
        if achar not in set([' ', '-', '_', '.']):
            adict[achar] += 1
    return adict

# Note: nobody is calling this now.
def align_aligned_strs(line1: str,
                       line2: str,
                       line2_span_list: List[Tuple[int, int]]) \
                       -> List[Tuple[int, int]]:
    """Align the line1's offsets to line2's span list.

    It is assumed that line1 and line2 are already completely aligned.

    If there is any failure, an empty list is returned
    """

    # print("align_aligned_strs()")
    # print("         str1: [{}]".format(line1))
    # print("         str2: [{}]".format(line2))
    # print("   str2_spans: [{}]".format(line2_span_listxs))

    idx1 = 0
    len1 = len(line1)
    out_span_list = []  # type: List[Tuple[int, int]]
    try:
        for start, unused_end in line2_span_list:
            idx1_span_start = idx1
            idx2 = start
            while line1[idx1] == line2[idx2]:
                idx1 += 1
                idx2 += 1

            out_span_list.append((idx1_span_start, idx1))
            # now line1 must have some extra spaces that
            # are not in line2_span_list
            while idx1 < len1 and is_space_huline(line1[idx1]):
                idx1 += 1
        return out_span_list
    except IndexError:
        logger.warning('align_aligned_strs failed on')
        logger.warning('      str1 = [%s]', line1)
        logger.warning('      str2 = [%s]', line2)
        logger.warning('str2 spans = %r', line2_span_list)
        return []


def is_aligned_leftover_char_maps(map1: Dict[str, int],
                                  map2: Dict[str, int]):
    """Determine if the leftover character are similar enought to
    claim 2 strings are aligned.

    The code here is intensionally not very precise.  Ideally,
    something closer to a cosine measure is probably better, but
    too slow.  This a much faster, but less accurate replacement.
    """
    count_diff = 0
    for achar, count in map1.items():
        count2 = map2.get(achar, 0)
        count_diff += abs(count - count2)

        if count_diff > 4:
            return False
    # try from the other perspective
    for achar, count in map2.items():
        count2 = map1.get(achar, 0)
        # if not found in the other map, then
        # this must be a diff
        if count2 == 0:
            count_diff += count
        if count_diff > 4:
            return False
    return True


def is_leftover_chars_mostly_overlap(line1: str, line2: str) -> bool:
    # if the leftover has no chars, then there is enough overlap before
    if not line1 or not line2:
        return True
    char1_map = make_aligned_charmap(line1)
    char2_map = make_aligned_charmap(line2)

    return is_aligned_leftover_char_maps(char1_map,
                                         char2_map)


# pylint: disable=too-many-branches, too-many-locals, too-many-statements
def compute_se_list(from_line: str,
                    to_line: str,
                    offset: int) \
    -> Optional[Tuple[List[Tuple[int, int]],
                      List[Tuple[int, int]],
                      Optional[Tuple[int, int]],
                      Optional[Tuple[int, int]]]]:
    """When return the extra fextra or textra, adjust for spaceprefix."""
    flen, tlen = len(from_line), len(to_line)
    # fse = from_line's extra start-end, tse=to_line's start-end
    extra_fse, extra_tse = None, None

    from_se_list, to_se_list = [], []
    fstart, tstart = 0, 0
    prev_matched_char = ' '

    # to handle the case where a string starts with special chars
    if is_space_huline(from_line[fstart]):
        fstart += 1
        while fstart < flen and is_space_huline(from_line[fstart]):
            fstart += 1
    if is_space_huline(to_line[tstart]):
        tstart += 1
        while tstart < tlen and is_space_huline(to_line[tstart]):
            tstart += 1

    fstart_00, tstart_00 = fstart, tstart
    fidx, tidx = fstart, tstart

    # in case a line1 or line2 == '_' only
    if fstart_00 == flen or \
        tstart_00 == tlen:
        return None

    while fidx < flen and tidx < tlen:
        if from_line[fidx] == to_line[tidx]:
            prev_matched_char = from_line[fidx]
            fidx += 1
            tidx += 1
        else:
            # if fstart == fidx or tstart == tidx:
            if fidx == fstart_00 or tidx == tstart_00:
                # if fidx < 3 or tidx < 3:
                if IS_DEBUG:
                    # even the first character didn't match
                    print("Character1 diff at %d, char '%s'" %
                          (offset + fidx, from_line[fidx]))
                return None

            # try to see if the mismatch is due to space or underline
            if fstart != fidx and \
                tstart != tidx:
                from_se_list.append((fstart, fidx))
                to_se_list.append((tstart, tidx))
                fstart, tstart = fidx, tidx
            else:
                break

            # from_line has underline, while the other has space
            # if from_line[fstart].isspace() or \
            #    (is_hyphen_underline(from_line[fstart]) and
            #    to_line[tstart].isspace()):
            """
            if from_line[fstart].isspace():
                fstart += 1
                while fstart < flen and from_line[fstart].isspace():
                    fstart += 1
            if to_line[tstart].isspace():
                tstart += 1
                while tstart < tlen and to_line[tstart].isspace():
                    tstart += 1

            if is_hyphen_underline(prev_matched_char):
                if is_space_huline(from_line[fstart]):
                    fstart += 1
                    while fstart < flen and is_space_huline(from_line[fstart]):
                        fstart += 1
                if is_space_huline(to_line[tstart]):
                    tstart += 1
                    while tstart < tlen and is_space_huline(to_line[tstart]):
                        tstart += 1
            """
            if is_space_huline(from_line[fstart]):
                fstart += 1
                while fstart < flen and is_space_huline(from_line[fstart]):
                    fstart += 1
            if is_space_huline(to_line[tstart]):
                tstart += 1
                while tstart < tlen and is_space_huline(to_line[tstart]):
                    tstart += 1

            # if either advanced, then there is a reason to move forward
            # otherwise, don't other
            if fstart != fidx or tstart != tidx:
                fidx, tidx = fstart, tstart
            else:
                break

    # if two lines are diff by two much, we don't bother claim they are matched.
    # We currently allow partial match of two strings due to some strings might be
    # concatenated randomly.

    # We can check for character overlap.  We currently assume x,y coordinate limit
    # such candidates.
    # if fidx < len(from_line) / 3.0 and \
    #   tidx < len(to_line) / 3.0:
    if not is_leftover_chars_mostly_overlap(from_line[fidx:],
                                            to_line[tidx:]):
        if IS_DEBUG:
            # even the first character didn't match
            print("Character5 diff at %d, char '%s'" %
                  (offset + fidx, from_line[fidx]))
        return None

    # if there was any match, add them
    if fidx != fstart:
        # must have saw some new text after last fstart reset
        from_se_list.append((fstart, fidx))
        to_se_list.append((tstart, tidx))

    # print("fidx = {}, tidx = {}".format(fidx, tidx))
    # it's either a mismatch or eoln is reached for both line1 and line2
    if fidx == flen and tidx == tlen:
        pass
    else:

        # check if the excessed on are all spaces
        # We are using fi2 and ti2 because we don't want to change
        # fidx and tidx if the ends are not reached
        fi2, ti2 = fidx, tidx
        if fi2 < flen:
            while fi2 < flen and \
                  (from_line[fi2].isspace() or \
                   (is_hyphen_underline(prev_matched_char) and
                    is_hyphen_underline(from_line[fi2]))):
                fi2 += 1

        if ti2 < tlen:
            while ti2 < tlen and \
                  (to_line[ti2].isspace() or \
                   (is_hyphen_underline(prev_matched_char) and
                    is_hyphen_underline(to_line[ti2]))):
                ti2 += 1

        if fi2 == flen and ti2 == tlen:
            pass
        elif fidx < flen and tidx < tlen:
            # this shouldn't happen.
            # usually one of the lines should be finished.
            if IS_DEBUG:
                print("Character2 diff at %d, char '%s', weird" %
                      (offset + fidx, from_line[fidx]))
            return (from_se_list,
                    adjust_list_offset(to_se_list, offset),
                    (fidx, flen),
                    adjust_pair_offset((tidx, tlen), offset))
        elif fi2 < flen:
            extra_fse = (fi2, flen)
        elif ti2 < tlen:
            extra_tse = (ti2, tlen)
        else:
            if IS_DEBUG:
                print("Character3 diff at %d, eoln" %
                      (offset + fidx, ))
            return None

    # print('flen = {}, tlen = {}'.format(flen, tlen))
    # print("from_se_list: {}".format(from_se_list))
    # print("to_se_list: {}".format(to_se_list))
    return (from_se_list,
            adjust_list_offset(to_se_list, offset),
            extra_fse,
            adjust_pair_offset(extra_tse, offset))


class AlignedStrMapper:

    def __init__(self, from_line: str, to_line: str, offset: int = 0) -> None:
        if IS_DEBUG:
            print("AlignedStrMapper(), offset = {}".format(offset))
            print("     from_line: [{}]".format(from_line))
            print("       to_line: [{}]".format(to_line))

        self.is_aligned = False
        self.is_fully_synced = False
        self.from_se_list = []  # type: List[Tuple[int, int]]
        self.to_se_list = []  # type: List[Tuple[int, int]]
        self.extra_fse = None  # type: Optional[Tuple[int, int]]
        self.extra_tse = None  # type: Optional[Tuple[int, int]]

        align_result = compute_se_list(from_line, to_line, offset)
        if align_result:
            self.from_se_list, self.to_se_list, \
                self.extra_fse, self.extra_tse = align_result
            self.is_aligned = True
            if not self.extra_fse and not self.extra_tse:
                self.is_fully_synced = True

        if IS_DEBUG:
            if self.is_aligned:
                print("    return from_se_list: {}".format(self.from_se_list))
                print("             to_se_list: {}".format(self.to_se_list))
                print("              extra_fse: {}".format(self.extra_fse))
                print("              extra_tse: {}".format(self.extra_tse))
            else:
                print("    return is_aligned is False")


    def __str__(self):
        return 'AlignedStrMapper(is_aligned={}\n'.format(self.is_aligned) + \
            '                 {},\n'.format(self.from_se_list) + \
            '                 {},\n'.format(self.to_se_list) + \
            '                 {},\n'.format(self.extra_fse) + \
            '                 {})'.format(self.extra_tse)

    # pylint: disable=invalid-name
    def get_to_offset(self, x: int) -> int:
        for i, from_se in enumerate(self.from_se_list):
            fstart, fend = from_se
            if x <= fend:
                if x >= fstart:
                    to_se = self.to_se_list[i]
                    tstart, unused_tend = to_se
                    diff = x - fstart
                    return tstart + diff
                return -1
        return -1

    def add_aligned_se_pair(self, aligned_se_pair: Tuple[int, int]) -> None:
        self.from_se_list.append(aligned_se_pair)  # the offset is not correct now
        self.to_se_list.append(aligned_se_pair)

    # TODO, why is the line below not valid?
    # def update_with_mapper(self, other: AlignedStrMapper) -> None:
    def update_with_mapper(self,
                           other_from_se_list: List[Tuple[int, int]],
                           other_to_se_list: List[Tuple[int, int]]) -> None:
        self.from_se_list.extend(other_from_se_list)
        self.to_se_list.extend(other_to_se_list)


# nobody is calling this now?
def unused_make_aligned_str_mapper(fromto_se_pair_list: List[Tuple[Tuple[int, int],
                                                                   Tuple[int, int]]]) \
                                                                   -> AlignedStrMapper:
    amapper = AlignedStrMapper('', '', offset=-1)
    for fromto_se_pair in fromto_se_pair_list:
        from_se, to_se = fromto_se_pair
        amapper.from_se_list.append(from_se)
        amapper.to_se_list.append(to_se)
    return amapper


# from_line is abbyy_line
# to_line is pbox_line
# pylint: disable=too-many-return-statements
def compute_matched_se_list(from_line: str,
                            to_line: str,
                            offset: int) \
    -> Optional[Tuple[List[Tuple[int, int]],
                      List[Tuple[int, int]],
                      List[Tuple[int, int]],
                      List[Tuple[int, int]]]]:

    if len(from_line) == len(to_line):
        if from_line == to_line:
            return [(0, len(from_line))], [(offset, offset + len(to_line))], [], []
        return None

    # if either from_line is all special char, it will match anything
    if is_all_hyphen_underline(from_line) or \
       is_all_hyphen_underline(to_line):
        return None

    # originally tried to use length to decide which line is
    # used for matching, but this is not reliable.  Example:
    #   str1 [aaa bb ccc]
    #   str2 [aaabbccc]
    # These would not match is the shorter version, str2,, is always
    # applied.

    mat_st = re.escape(to_line)
    # allow multiple '-', '_', and ' '
    mat_st = mat_st.replace(' ', ' *')
    mat_st = mat_st.replace('-', '-*')
    mat_st = mat_st.replace('_', '_*')
    abbyy_big_mat_list = list(re.finditer(mat_st, from_line))

    if len(abbyy_big_mat_list) == 1:
        # pbox's match, offset is based on abbyy_line or from_line
        mat = abbyy_big_mat_list[0]
        mstart, mend = mat.start(), mat.end()
        before_start, before_end = 0, mstart
        after_start, after_end = mend, len(from_line)

        len_matched = mend - mstart
        len_to = len(to_line)
        if len_matched != len_to:
            line3 = from_line[mstart:mend]
            as_mapper = AlignedStrMapper(line3, to_line)
            # print('as_mapper: {}'.format(as_mapper))
            out_to_se_list = adjust_list_offset(as_mapper.to_se_list, offset)
            out_from_se_list = adjust_list_offset(as_mapper.from_se_list, offset + mstart)

            # can be removed in future, same behavior has above shorter version
            # out_from_se_list = align_aligned_strs(from_line[mstart:mend],
            #                                      to_line,
            #                                       as_mapper.to_se_list)
            # out_from_se_list = adjust_list_offset(out_from_se_list,
            #                                       mstart)
        else:
            out_to_se_list = [(offset, offset + len(to_line))]
            out_from_se_list = [(0, len(from_line))]

        # There should be possible either before frag and after frag,
        # but the API only allow returning 1 frag back.
        # Since previously, the front has being checked in aligned_str_mapper.
        # Assume front one has priority.
        from_extra_se_list = []
        # take the suffix extra first
        if before_start != before_end:
            from_extra_se_list.append((before_start, before_end))
        if after_start != after_end:
            from_extra_se_list.append((after_start, after_end))

        return out_from_se_list, out_to_se_list, from_extra_se_list, []
    elif not abbyy_big_mat_list:
        mat_st = re.escape(from_line)
        # allow multiple '-', '_', and ' '
        mat_st = mat_st.replace(' ', ' *')
        mat_st = mat_st.replace('-', '-*')
        mat_st = mat_st.replace('_', '_*')
        pbox_big_mat_list = list(re.finditer(mat_st, to_line))

        if len(pbox_big_mat_list) == 1:
            # abby's match, offset is based on pbox_line or to_line
            mat = pbox_big_mat_list[0]
            mstart, mend = mat.start(), mat.end()
            before_start, before_end = 0, mstart
            after_start, after_end = mend, len(to_line)

            len_matched = mend - mstart
            len_from = len(from_line)
            if len_matched != len_from:
                line3 = to_line[mstart:mend]
                as_mapper = AlignedStrMapper(line3, from_line)
                # print('as_mapper: {}'.format(as_mapper))
                out_to_se_list = adjust_list_offset(as_mapper.from_se_list, offset + mstart)
                out_from_se_list = as_mapper.to_se_list
            else:
                out_to_se_list = [(offset + mstart, offset + mend)]
                out_from_se_list = [(0, len(from_line))]

            to_extra_se_list = []
            if before_start != before_end:
                to_extra_se_list.append((before_start + offset, before_end + offset))
            if after_start != after_end:
                to_extra_se_list.append((after_start + offset, after_end + offset))

            return out_from_se_list, out_to_se_list, [], to_extra_se_list

        # 0 or more than 1, no match
        return None

    # not 1 or 0 matches, too ambiguous, fail
    return None



# pylint: disable=too-few-public-methods
class MatchedStrMapper:

    def __init__(self, from_line: str, to_line: str, offset: int = 0) -> None:
        if IS_DEBUG:
            print("MatchedStrMapper(), offset = {}".format(offset))
            print("     from_line: [{}]".format(from_line))
            print("       to_line: [{}]".format(to_line))

        self.is_aligned = False
        self.is_fully_synced = False
        self.from_se_list = []  # type: List[Tuple[int, int]]
        self.to_se_list = []  # type: List[Tuple[int, int]]
        # there can be something before or after the matched str
        # minimum 0, to maximum 2
        self.extra_fse_list = []  # type: List[Tuple[int, int]]
        self.extra_tse_list = []  # type: List[Tuple[int, int]]

        align_result = compute_matched_se_list(from_line, to_line, offset)
        if align_result:
            self.from_se_list, self.to_se_list, \
                self.extra_fse_list, self.extra_tse_list = align_result
            self.is_aligned = True
            if not self.extra_fse_list and not self.extra_tse_list:
                self.is_fully_synced = True

        if IS_DEBUG:
            if self.is_aligned:
                print("    return from_se_list: {}".format(self.from_se_list))
                print("             to_se_list: {}".format(self.to_se_list))
                print("         extra_fse_list: {}".format(self.extra_fse_list))
                print("         extra_tse_list: {}".format(self.extra_tse_list))
            else:
                print("    return is_aligned is False")


    def __str__(self):
        return 'MatchedStrMapper({},\n'.format(self.from_se_list) + \
            '                 {},\n'.format(self.to_se_list) + \
            '                 {},\n'.format(self.extra_fse_list) + \
            '                 {})'.format(self.extra_tse_list)

    # pylint: disable=invalid-name
    def get_to_offset(self, x: int) -> int:
        for i, from_se in enumerate(self.from_se_list):
            fstart, fend = from_se
            if x <= fend:
                if x >= fstart:
                    to_se = self.to_se_list[i]
                    tstart, unused_tend = to_se
                    diff = x - fstart
                    return tstart + diff
                return -1
        return -1

    # TODO, why is the line below not valid?
    # def update_with_mapper(self, other: AlignedStrMapper) -> None:
    def update_with_mapper(self,
                           other_from_se_list: List[Tuple[int, int]],
                           other_to_se_list: List[Tuple[int, int]]) -> None:
        self.from_se_list.extend(other_from_se_list)
        self.to_se_list.extend(other_to_se_list)
