
from typing import List, Optional, Tuple


def is_space_uline(line: str) -> bool:
    return line.isspace() or line == '_'


class AlignedStrMapper:

    def __init__(self, from_line: str, to_line: str, offset: int = 0) -> None:
        self.is_fully_synced = False
        self.extra_fse, self.extra_tse = None, None

        if offset == -1:
            self.from_se_list = []
            self.to_se_list = []
        else:
            from_se_list, to_se_list, extra_fse, extra_tse = \
                self.compute_se_list(from_line, to_line, offset)
            self.from_se_list = from_se_list
            self.to_se_list = to_se_list
            if not extra_fse and not extra_tse:
                self.is_fully_synced = True
            else:
                self.extra_fse = extra_fse
                self.extra_tse = extra_tse

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

    # pylint: disable=no-self-use
    def add_offset(self, se_list: List[Tuple[int, int]], offset: int) -> List[Tuple[int, int]]:
        return [(offset + start, offset + end)
                for start, end in se_list]

    def add_aligned_se_pair(self, aligned_se_pair: Tuple[int, int]) -> None:
        self.from_se_list.append(aligned_se_pair)  # the offset is not correct now
        self.to_se_list.append(aligned_se_pair)

    # pylint: disable=too-many-branches
    def compute_se_list(self, from_line: str, to_line: str, offset: int) \
        -> Tuple[List[Tuple[int, int]],
                 List[Tuple[int, int]],
                 Optional[Tuple[int, int]],
                 Optional[Tuple[int, int]]]:
        fidx, tidx = 0, 0
        flen, tlen = len(from_line), len(to_line)
        # fse = from_line's extra start-end, tse=to_line's start-end
        extra_fse, extra_tse = None, None

        from_se_list = []
        to_se_list = []
        fstart, tstart = 0, 0
        while fidx < flen and tidx < tlen:
            if from_line[fidx] == to_line[tidx]:
                fidx += 1
                tidx += 1
            else:
                if fstart == fidx:
                    raise Exception("Character diff at %d, char '%s'" %
                                    (offset + fidx, from_line[fidx]))
                from_se_list.append((fstart, fidx))
                to_se_list.append((tstart, tidx))
                fstart, tstart = fidx, tidx

                if is_space_uline(from_line[fstart]):
                    fstart += 1
                    while fstart < flen and is_space_uline(from_line[fstart]):
                        fstart += 1
                elif is_space_uline(to_line[tstart]):
                    tstart += 1
                    while tstart < tlen and is_space_uline(to_line[tstart]):
                        tstart += 1
                else:
                    raise Exception("Character diff at %d, char '%s'" %
                                    (offset + fidx, from_line[fidx]))
                fidx, tidx = fstart, tstart
            # print("fidx = {}, tidx = {}".format(fidx, tidx))
        if fidx == flen and tidx == tlen:
            if fidx != fstart:
                from_se_list.append((fstart, fidx))
                to_se_list.append((tstart, tidx))
        else:
            # check if the excessed on are all spaces
            fi2, ti2 = fidx, tidx
            if fidx < flen:
                while fi2 < flen and is_space_uline(from_line[fi2]):
                    fi2 += 1
            if tidx < tlen:
                while ti2 < tlen and is_space_uline(to_line[ti2]):
                    ti2 += 1

            if fi2 == flen and ti2 == tlen:
                if fidx != fstart:
                    from_se_list.append((fstart, fidx))
                    to_se_list.append((tstart, tidx))
            else:
                if fi2 < flen:
                    # print("flen = {}, fi2 = {}, fidx = {}, fstart= {}".format(flen, fi2, fidx, fstart))
                    # print("tlen = {}, ti2 = {}, tidx = {}, tstart= {}".format(tlen, ti2, tidx, tstart))
                    if flen - fidx <= 3 and \
                       fidx > 10 and \
                       ti2 == tlen and \
                       fidx == fstart:
                        # Probably in toc.  Don't have the number at the end
                        # Don't add empty spans
                        # from_se_list.append((fstart, fidx))
                        # to_se_list.append((tstart, tidx))
                        extra_fse = (fi2, flen)
                        # pass
                    else:
                        raise Exception("Character diff at %d, char '%s'" %
                                        (offset + fidx, from_line[fidx]))
                else:
                    # fi2 >= flen
                    # print("flen = {}, fi2 = {}, fidx = {}, fstart= {}".format(flen, fi2, fidx, fstart))
                    # print("tlen = {}, ti2 = {}, tidx = {}, tstart= {}".format(tlen, ti2, tidx, tstart))
                    if tlen - ti2 <= 3 and \
                       fidx > 10:
                        extra_tse = (ti2, tlen)
                        # pass
                    else:
                        raise Exception("Character diff at %d, eoln" %
                                        (offset + fidx, ))

        # print('flen = {}, tlen = {}'.format(flen, tlen))
        # print("from_se_list: {}".format(from_se_list))
        # print("to_se_list: {}".format(to_se_list))
        return self.add_offset(from_se_list, offset), self.add_offset(to_se_list, offset), extra_fse, extra_tse


def make_aligned_str_mapper(fromto_se_pair_list: List[Tuple[Tuple[int, int],
                                                            Tuple[int, int]]]) \
                                                            -> AlignedStrMapper:
    amapper = AlignedStrMapper('', '', offset=-1)
    for fromto_se_pair in fromto_se_pair_list:
        from_se, to_se = fromto_se_pair
        amapper.from_se_list.append(from_se)
        amapper.to_se_list.append(to_se)
    return amapper


