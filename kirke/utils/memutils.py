#!/usr/bin/env python3

from collections import namedtuple
from pympler import asizeof

from typing import Any

# https://stackoverflow.com/questions/33978/find-out-how-much-memory-is-being-used-by-an-object-in-python

# pylint: disable=too-few-public-methods
class StartEndPair(object):
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

# pylint: disable=too-few-public-methods
class StartEndPair2(object):
    __slots__ = ['start', 'end']
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

StartEndNamedPair = namedtuple('StartEndNamedPair', 'start end')

def get_size(obj: Any) -> int:
    return asizeof.asizeof(obj);


get_size_bytes = get_size


def get_size_kbytes(obj: Any) -> float:
    return asizeof.asizeof(obj) / 1000.0;


def get_size_mbytes(obj: Any) -> float:
    return asizeof.asizeof(obj) / 100000.0;


if __name__ == '__main__':

    print('psize(SE_pair) = %d' % asizeof.asizeof(StartEndPair(1, 2)))

    print('psize(SE_pair2) = %d' % asizeof.asizeof(StartEndPair2(1, 2)))

    print('psize(tuple) = %d' % asizeof.asizeof((1, 2)))

    print('psize(int) = %d' % asizeof.asizeof(1))
    print('psize(tuple1) = %d' % asizeof.asizeof((1,)))
    print('psize(tuple2) = %d' % asizeof.asizeof((1, 2)))
    print('psize(tuple3) = %d' % asizeof.asizeof((1, 2, 3)))

    print('psize(namedtuple) = %d' % asizeof.asizeof(StartEndNamedPair(1, 2)))
    print('psize(list) = %d' % asizeof.asizeof([1, 2]))
