from collections import Mapping, Container, defaultdict
import fcntl
from fcntl import LOCK_EX, LOCK_SH
import os
import re
import sys
from sys import getsizeof
# pylint: disable=unused-import
from typing import Any, Dict, List, Optional, Set, Tuple


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


def _find_fname_with_lang(lang: str, lang_fname_list: List[Tuple[str, str]]) -> str:
    """Return the model file name that matches the language.

    Returns None if not even EN is available.
    """
    en_fname = None
    for xlang, fname in lang_fname_list:
        if xlang == lang:
            return fname
        if xlang == 'en':
            en_fname = fname
    # return the EN if lang has no match;  None if EN not available
    return en_fname

def _get_highest_version(ver_lang_fname_list: List[Tuple[int, Tuple[str, str]]]) \
    -> List[Tuple[str, str]]:
    ver_lang_fnames_map = defaultdict(list)  # type: DefaultDict[int, List[Tuple[str, str]]]
    ver_list = []
    for ver, lang_fname in ver_lang_fname_list:
        ver_lang_fnames_map[ver].append(lang_fname)
        ver_list.append(ver)
    highest_ver = sorted(ver_list)[-1]
    return ver_lang_fnames_map[highest_ver]



CUSTOM_MODEL_PREFIX_PAT = re.compile(r'((cust_\d+)(\.(\d+))?)_(..)(.)')

# When get all the file names related to the provisions, there is what
# happens:
#    If cust_prov has version specified, take that version.
#    In a version, there can be multilple language for that provision.
#      - If the document language
#           - match with a versioned model with the language, return it.
#           - not match, take the EN version, if available
#    If cust_prov has NO version specified, take the highest version
#      - Once highest version is chosen, the rest of logic is followed
# Clearly, cust_prov with no version is much more costly because we have
# to figure out which is the latest version.  This should be avoided.
# pylint: disable=too-many-locals
def get_custom_model_files(dir_name: str,
                           cust_prov_set: Set[str],
                           lang: str) -> List[Tuple[str, str]]:
    """Return the set of file names that matched cust_prov_set and lang.

    Return a list of tuples, with provision_name, model_file_name.
    The provision name is exactly the same as those from cust_prov_set.
    """
    maybe_fnames = [f for f in os.listdir(dir_name)
                    if (os.path.isfile(os.path.join(dir_name, f))
                        and 'classifier' in f and f.endswith('.pkl'))]

    # the key is the cust_id_ver; 1st tuple str is 'lang', 2nd value is fname
    cust_idver_set = set([])  # type: Set[str]
    cust_idver_fnames_map = defaultdict(list)  # DefaultDict[str, Tuple[str, str]]
    # the key is the cust_id;
    # 1st tuple str is int(version), 2nd str is 'lang', 3rd value is fname
    cust_id_set = set([])  # type: Set[str]
    cust_id_fnames_map = defaultdict(list)  # DefaultDict[str, Tuple[int, str, str]]
    # print("cust_prov_set: {}".format(cust_prov_set))
    for cust_prov in cust_prov_set:
        if '.' in cust_prov:
            cust_idver_set.add(cust_prov)
        else:
            cust_id_set.add(cust_prov)

    # print("cust_idver_set = {}".format(cust_idver_set))
    # print("cust_id_set = {}".format(cust_id_set))

    for maybe_fname  in maybe_fnames:
        mat = CUSTOM_MODEL_PREFIX_PAT.match(maybe_fname)
        if mat:
            maybe_prov_ver = mat.group(1)
            maybe_prov_name = mat.group(2)
            maybe_ver = mat.group(4)
            maybe_lang = mat.group(5)
            maybe_dash = mat.group(6)

            if maybe_dash != '_':
                maybe_lang = 'en'
            if not maybe_ver:
                maybe_ver = '0'

            # if maybe_prov_name == cust_prov:
            #     print("maybe_prov_ver = [{}], {}, lang={}".format(maybe_prov_ver,
            #                                                       maybe_ver,
            #                                                       maybe_lang))

            # for a version, there can be multiple language still
            # for that particular version, pick the best language.
            # if no language match, take the "en"
            if maybe_prov_ver in cust_idver_set:
                cust_idver_fnames_map[maybe_prov_ver].append((maybe_lang, maybe_fname))

            if maybe_prov_name in cust_id_set:
                # for non-version, we need to add all versions
                # elif maybe_prov_ver in cust_id_set:
                cust_id_fnames_map[maybe_prov_name].append((int(maybe_ver),
                                                            (maybe_lang, maybe_fname)))

    cust_prov_fnames = []  # type: List[Tuple[str, str]]
    for cust_idver, lang_fname_list in cust_idver_fnames_map.items():
        if lang == 'all':  # this is for custom-train-export
            for xlang, fname in lang_fname_list:
                cust_prov_fnames.append((cust_idver, fname))
        else:
            cust_prov_fnames.append((cust_idver, _find_fname_with_lang(lang, lang_fname_list)))

    for cust_id, ver_lang_fname_list in cust_id_fnames_map.items():
        lang_fname_list = _get_highest_version(ver_lang_fname_list)
        if lang == 'all':  # this is for custom-train-export
            for xlang, fname in lang_fname_list:
                cust_prov_fnames.append((cust_id, fname))
        else:        
            cust_prov_fnames.append((cust_id, _find_fname_with_lang(lang, lang_fname_list)))

    return cust_prov_fnames


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

def deep_getsizeof(obj, ids: Set) -> int:
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
                          for k, v in obj.items())

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

def read_cluster_name(model_dir: str) -> Optional[str]:
    line = read_locked_file('%s/kirke_cluster_name.txt' % (model_dir, ))
    return line


if __name__ == '__main__':
    XOBJ = '1234567'
    print(get_size(XOBJ))
