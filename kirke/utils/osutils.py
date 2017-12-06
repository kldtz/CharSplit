from collections import Mapping, Container
import os
import sys
from sys import getsizeof


# Create a directory and any missing ancestor directories.
# If the directory already exists, do nothing.
# similar to distutils.dir_util import mkpath
def mkpath(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_last_cmd_line_arg():
    prefix = ''
    if len(sys.argv) > 1:
        prefix = sys.argv[-1]
    return prefix

# Examples of model example file names:
#   jurisdiction_scutclassifier.pkl
#   cust_3_scutclassifier.pk
def get_model_files(dir_name):
    return [f for f in os.listdir(dir_name)
            if (os.path.isfile(os.path.join(dir_name, f))
                and 'classifier' in f and f.endswith('.pkl'))]

def get_size(obj, seen=None):
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

def deep_getsizeof(obj, ids):
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


if __name__ == '__main__':
    XOBJ = '1234567'
    print(get_size(XOBJ))
