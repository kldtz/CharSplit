from typing import Dict, Tuple

# from collections import namedtuple
#
# LnPos = namedtuple('LnPos', ['start', 'end', 'line_num', 'is_gap'])
# # set the defautls for line_num and is_gap
# LnPos.__new__.__defaults__ = (-1, False)

# pylint: disable=too-few-public-methods
class LnPos:
    __slots__ = ['start', 'end', 'line_num']

    def __init__(self, start: int, end: int, line_num: int = -1) -> None:
        self.start = start  # type: int
        self.end = end  # type: int
        self.line_num = line_num  # type: int

    def __str__(self) -> str:
        alist = ['{}, {}'.format(self.start, self.end)]
        if self.start != self.end:
            alist.append('{}'.format(self.line_num))
        return '({})'.format(', '.join(alist))

    def __repr__(self) -> str:
        return self.__str__()

    def __lt__(self, other) -> bool:
        if self.start == other.start:
            return self.end < other.end
        return self.start < other.start

    # similar to namedtuple._asdict()
    def _asdict(self) -> Dict[str, int]:
        adict = {'start': self.start,
                 'end': self.end,
                 'line_num': self.line_num}
        return adict

    def to_tuple(self) -> Tuple[int, int, int]:
        return self.start, self.end, self.line_num
