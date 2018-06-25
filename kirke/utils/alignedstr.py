
from typing import List, Optional, Tuple

IS_DEBUG = False

def is_space_uline(line: str) -> bool:
    return line.isspace() or line == '_'

def is_underline(line: str) -> bool:
    return line == '_'

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


class AlignedStrMapper:

    def __init__(self, from_line: str, to_line: str, offset: int = 0) -> None:
        if IS_DEBUG:
            print("AlignedStrMapper(), offset = {}".format(offset))
            print("     from_line: [{}]".format(from_line))
            print("       to_line: [{}]".format(to_line))

        self.is_fully_synced = False
        # pylint: disable=line-too-long
        self.extra_fse, self.extra_tse = None, None  # type: Optional[Tuple[int, int]], Optional[Tuple[int, int]],

        if offset == -1:
            self.from_se_list = []  # type: List[Tuple[int, int]]
            self.to_se_list = []  # type: List[Tuple[int, int]]
        else:
            (self.from_se_list, self.to_se_list,
             self.extra_fse, self.extra_tse) = \
                self.compute_se_list(from_line, to_line, offset)
            if not self.extra_fse and not self.extra_tse:
                self.is_fully_synced = True

        if IS_DEBUG:
            print("    return from_se_list: {}".format(self.from_se_list))
            print("             to_se_list: {}".format(self.to_se_list))
            print("              extra_fse: {}".format(self.extra_fse))
            print("              extra_tse: {}".format(self.extra_tse))


    def __str__(self):
        return 'AlignedStrMapper({},\n'.format(self.from_se_list) + \
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
                else:
                    return -1
        return -1

    def add_aligned_se_pair(self, aligned_se_pair: Tuple[int, int]) -> None:
        self.from_se_list.append(aligned_se_pair)  # the offset is not correct now
        self.to_se_list.append(aligned_se_pair)

    # pylint: disable=too-many-branches
    def compute_se_list(self,
                        from_line: str,
                        to_line: str,
                        offset: int) \
        -> Tuple[List[Tuple[int, int]],
                 List[Tuple[int, int]],
                 Optional[Tuple[int, int]],
                 Optional[Tuple[int, int]]]:
        fidx, tidx = 0, 0
        flen, tlen = len(from_line), len(to_line)
        # fse = from_line's extra start-end, tse=to_line's start-end
        extra_fse, extra_tse = None, None

        from_se_list, to_se_list = [], []
        fstart, tstart = 0, 0
        prev_matched_char = ' '
        while fidx < flen and tidx < tlen:
            if from_line[fidx] == to_line[tidx]:
                prev_matched_char = from_line[fidx]
                fidx += 1
                tidx += 1
            else:
                if fstart == fidx or tstart == tidx:
                    # even the first character didn't match
                    raise Exception("Character1 diff at %d, char '%s'" %
                                    (offset + fidx, from_line[fidx]))

                # try to see if the mismatch is due to space or underline
                from_se_list.append((fstart, fidx))
                to_se_list.append((tstart, tidx))
                fstart, tstart = fidx, tidx

                if from_line[fstart].isspace():
                    fstart += 1
                    while fstart < flen and from_line[fstart].isspace():
                        fstart += 1
                elif to_line[tstart].isspace():
                    tstart += 1
                    while tstart < tlen and to_line[tstart].isspace():
                        tstart += 1

                if is_underline(prev_matched_char):
                    if is_space_uline(from_line[fstart]):
                        fstart += 1
                        while fstart < flen and is_space_uline(from_line[fstart]):
                            fstart += 1
                    if is_space_uline(to_line[tstart]):
                        tstart += 1
                        while tstart < tlen and is_space_uline(to_line[tstart]):
                            tstart += 1

                # if either advanced, then there is a reason to move forward
                # otherwise, don't other
                if fstart != fidx or tstart != tidx:
                    fidx, tidx = fstart, tstart
                else:
                    break


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
                       (is_underline(prev_matched_char) and
                        is_underline(from_line[fi2]))):
                    fi2 += 1

            if ti2 < tlen:
                while ti2 < tlen and \
                      (to_line[ti2].isspace() or \
                       (is_underline(prev_matched_char) and
                        is_underline(to_line[ti2]))):
                    ti2 += 1

            if fi2  == flen and ti2 == tlen:
                pass
            elif fidx < flen and tidx < tlen:
                # this shouldn't happen.
                # usually one of the lines should be finished.
                raise Exception("Character2 diff at %d, char '%s', weird" %
                                (offset + fidx, from_line[fidx]))
            elif fidx < flen:
                extra_fse = (fidx, flen)
            elif tidx < tlen:
                extra_tse = (tidx, tlen)
            else:
                raise Exception("Character3 diff at %d, eoln" %
                                (offset + fidx, ))

        # print('flen = {}, tlen = {}'.format(flen, tlen))
        # print("from_se_list: {}".format(from_se_list))
        # print("to_se_list: {}".format(to_se_list))
        return (adjust_list_offset(from_se_list, offset),
                adjust_list_offset(to_se_list, offset),
                adjust_pair_offset(extra_fse, offset),
                adjust_pair_offset(extra_tse, offset))


def make_aligned_str_mapper(fromto_se_pair_list: List[Tuple[Tuple[int, int],
                                                            Tuple[int, int]]]) \
                                                            -> AlignedStrMapper:
    amapper = AlignedStrMapper('', '', offset=-1)
    for fromto_se_pair in fromto_se_pair_list:
        from_se, to_se = fromto_se_pair
        amapper.from_se_list.append(from_se)
        amapper.to_se_list.append(to_se)
    return amapper
