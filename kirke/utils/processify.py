# From https://gist.github.com/schlamar/2311116
# pylint: disable=line-too-long
# Fecommended from http://chase-seibert.github.io/blog/2013/08/03/diagnosing-memory-leaks-python.html

import os
import sys
import traceback
from functools import wraps
from multiprocessing import Process, Queue
import logging

# pylint: disable=invalid-name
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    '''

    def process_func(qxx, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        # pylint: disable=broad-except
        except Exception:
            ex_type, ex_value, tbk = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tbk))
            ret = None
        else:
            error = None

        qxx.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        qxx = Queue()
        proc = Process(target=process_func, args=[qxx] + list(args), kwargs=kwargs)
        proc.start()
        ret, error = qxx.get()
        proc.join()

        if error:
            # ex_value can be null, then it causes the process to crash
            logger.error("error: %r", error)
            # ex_type, ex_value, tb_str = error
            # message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            # raise ex_type(message)
            raise Exception("error: {}".format(error))

        return ret
    return wrapper


@processify
def test_function():
    return os.getpid()


@processify
def test_deadlock():
    return range(30000)


@processify
def test_exception():
    raise RuntimeError('xyz')


def test():
    print(os.getpid())
    print(test_function())
    print(len(test_deadlock()))
    test_exception()

if __name__ == '__main__':
    test()
