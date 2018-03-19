from collections import Mapping, Container
import os
import fcntl
from fcntl import LOCK_EX, LOCK_SH, LOCK_NB
import sys
from sys import getsizeof
from typing import Any, List, Optional, Set


# Create a directory and any missing ancestor directories.
# If the directory already exists, do nothing.
# similar to distutils.dir_util import mkpath
def mkpath(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_last_cmd_line_arg() -> str:
    prefix = ''
    if len(sys.argv) > 1:
        prefix = sys.argv[-1]
    return prefix

# Examples of model example file names:
#   jurisdiction_scutclassifier.pkl
#   cust_3_scutclassifier.pk
def get_model_files(dir_name: str) -> List[str]:
    return [f for f in os.listdir(dir_name)
            if (os.path.isfile(os.path.join(dir_name, f))
                and not 'docclassifier' in f
                and 'classifier' in f and f.endswith('.pkl'))]

def get_size(obj: Any, seen: Optional[Set] = None) -> int:
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size

def deep_getsizeof(obj: Any, ids: Set) -> int:
    """Find the memory footprint of a Python object

    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    :param obj: the object
    :param ids:
    :return:
    """
    dsize = deep_getsizeof
    if id(obj) in ids:
        return 0

    rval = getsizeof(obj)
    ids.add(id(obj))

    # if isinstance(obj, str) or isinstance(0, unicode):
    if isinstance(obj, str):
        return rval

    if isinstance(obj, Mapping):
        return rval + sum(dsize(k, ids) + dsize(v, ids)
                          for k, v in obj.iteritems())

    if isinstance(obj, Container):
        return rval + sum(dsize(xobj, ids) for xobj in obj)

    return rval


# https://www.safaribooksonline.com/library/view/python-cookbook/0596001673/ch04s25.html
def lock(file, flags):
    fcntl.flock(file.fileno(), flags)

def unlock(file):
    fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def increment_file(file_name: str) -> int:
    int_val = 1000
    try:
        with open(file_name, 'r+') as fin:
            lock(fin, LOCK_EX)

            line = fin.read().strip()
            if line:
                int_val = int(line)
            int_val += 1
            fin.seek(0)
            fin.write("%d\n" % (int_val, ))
            fin.truncate()

            unlock(fin)
    except FileNotFoundError:
        int_val += 1 
        with open(file_name, 'w') as fout:
            lock(fout, LOCK_EX)
            fout.write("%d\n" % (int_val, ))
            unlock(fout)
    return int_val

def read_version_file(file_name: str) -> int:
    int_val = -1
    try:
        with open(file_name, 'r') as fin:
            lock(fin, LOCK_SH)
            int_val = int(fin.read().strip())
            unlock(fin)
    except FileNotFoundError:
        return -1

    return int_val


def save_locked_file(file_name: str, text: str) -> None:
    with open(file_name, 'w') as fout:
        lock(fout, LOCK_EX)
        fout.write("%s\n" % (text, ))
        unlock(fout)

def read_locked_file(file_name: str) -> Optional[str]:
    line = None
    try:
        with open(file_name, 'r') as fin:
            lock(fin, LOCK_SH)
            line = fin.read().strip()
            unlock(fin)
    except FileNotFoundError:
        return None

    return line

    
def increment_model_version(model_dir: str) -> int:
    version = increment_file('%s/kirke_model_count.txt' % (model_dir, ))
    return version

def read_model_version(model_dir: str) -> int:
    version = read_version_file('%s/kirke_model_count.txt' % (model_dir, ))
    return version

def set_cluster_name(line: str, model_dir: str) -> None:
    save_locked_file('%s/kirke_cluster_name.txt' % (model_dir, ), line)

def read_cluster_name(model_dir: str) -> int:
    line = read_locked_file('%s/kirke_cluster_name.txt' % (model_dir, ))
    return line


if __name__ == '__main__':
    XOBJ = '1234567'
    print(get_size(XOBJ))



