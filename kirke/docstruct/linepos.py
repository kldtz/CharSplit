
from typing import Dict, Union


class LnPos:
    __slots__ = ['start', 'end', 'line_num', 'is_gap']
    
    def __init__(self, start: int, end: int, line_num: int=-1, is_gap: bool=False) -> None:
        self.start = start
        self.end = end
        self.line_num = line_num
        self.is_gap = is_gap

    def __str__(self) -> str:
        """
        alist = ['s={}, e={}'.format(self.start, self.end)]
        if self.start != self.end:
            alist.append('ln={}'.format(self.line_num))
        if self.is_gap:
            alist.append('gap')
        return '({})'.format(' ,'.join(alist))
        """
        alist = ['{}, {}'.format(self.start, self.end)]
        if self.start != self.end:
            alist.append('{}'.format(self.line_num))
        if self.is_gap:
            alist.append('gap')
        return '({})'.format(', '.join(alist))        

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> Dict[str, Union[int, bool]]:
        adict = {'start': self.start,
                 'end': self.end,
                 'line_num': self.line_num}
        if self.is_gap:
            adict['gap'] = True
        return adict
