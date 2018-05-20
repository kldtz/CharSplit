
from typing import List, Tuple


def is_space_uline(line: str) -> bool:
    return line.isspace() or line == '_'


class AlignedStrMapper:

    def __init__(self, from_line: str, to_line: str, offset: int = 0) -> None:
        from_se_list, to_se_list = self.compute_se_list(from_line, to_line, offset)
        self.from_se_list = from_se_list
        self.to_se_list = to_se_list
        self.offset = offset

    # pylint: disable=invalid-name
    def get_to_offset(self, x: int) -> int:
        for i, from_se in enumerate(self.from_se_list):
            fstart, fend = from_se
            if x < fend:
                to_se = self.to_se_list[i]
                tstart, unused_tend = to_se
                diff = x - fstart
                return tstart + diff
        return -1

    # pylint: disable=no-self-use
    def add_offset(self, se_list: List[Tuple[int, int]], offset: int) -> List[Tuple[int, int]]:
        return [(offset + start, offset + end)
                for start, end in se_list]

    # pylint: disable=too-many-branches
    def compute_se_list(self, from_line: str, to_line: str, offset: int) \
        -> Tuple[List[Tuple[int, int]],
                 List[Tuple[int, int]]]:
        fidx, tidx = 0, 0
        flen, tlen = len(from_line), len(to_line)

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
                if fidx < flen:
                    raise Exception("Character diff at %d, char '%s'" %
                                    (offset + fidx, from_line[fidx]))
                else:
                    raise Exception("Character diff at %d, eoln" %
                                    (offset + fidx, ))

        # print('flen = {}, tlen = {}'.format(flen, tlen))
        # print("from_se_list: {}".format(from_se_list))
        # print("to_se_list: {}".format(to_se_list))
        return self.add_offset(from_se_list, offset), self.add_offset(to_se_list, offset)
