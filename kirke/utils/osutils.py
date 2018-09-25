import configparser
from datetime import datetime
import fcntl
from fcntl import LOCK_EX, LOCK_SH
import os
import re
import shutil
import sys
import tempfile
import time
# pylint: disable=unused-import
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

from hashlib import md5

from sklearn.externals import joblib

# pylint: disable=invalid-name
config = configparser.ConfigParser()
config.read('kirke.ini')


# Create a directory and any missing ancestor directories.
# If the directory already exists, do nothing.
# similar to distutils.dir_util import mkpath
def mkpath(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

EB_FILES = os.environ['EB_FILES']
KIRKE_TMP_DIR = EB_FILES + config['ebrevia.com']['KIRKE_TMP']
# make sure that KIRKE_TMP_DIR is setup correctly
mkpath(KIRKE_TMP_DIR)


def get_last_cmd_line_arg() -> str:
    prefix = ''
    if len(sys.argv) > 1:
        prefix = sys.argv[-1]
    return prefix


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


def joblib_atomic_dump(an_object: Any,
                       file_name: str,
                       tmp_dir: str = KIRKE_TMP_DIR) -> None:
    tmpFileName = tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False)
    joblib.dump(an_object, tmpFileName.name)
    shutil.move(tmpFileName.name, file_name)


def atomic_dumps(text: str,
                 file_name: str,
                 tmp_dir: str = KIRKE_TMP_DIR) -> None:
    tmpFileName = tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False)
    with open(tmpFileName.name, 'wt') as fout:
        fout.write(text)
    shutil.move(tmpFileName.name, file_name)


def get_minute_timestamp_str() -> str:
    """Return a timestamp string, down to minutes."""
    timestamp = int(time.time())
    # datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    aline = datetime.fromtimestamp(timestamp).strftime('%Y%m%d-%H%M')
    return aline


def get_text_md5(doc_nlp_text: str) -> str:
    nlptxt_hash = md5()
    nlptxt_hash.update(doc_nlp_text.encode('utf-8'))
    return nlptxt_hash.hexdigest()


DOCID_MD5_PAT = re.compile(r'^(\d+)\-([a-f0-9]{32})(.*)$', re.I)
MD5_DOCID_PAT = re.compile(r'^([a-f0-9]{32})\-(\d+)(.*)$', re.I)

def split_docid_md5(base_file_name: str) -> Optional[Tuple[str, str, str]]:
    mat = re.match(MD5_DOCID_PAT, base_file_name)
    if mat:
        return mat.group(2), mat.group(1), mat.group(3)

    mat = re.match(DOCID_MD5_PAT, base_file_name)
    if mat:
        return mat.group(1), mat.group(2), mat.group(3)
    return None


def get_docid(file_name: str) -> Optional[str]:
    """Return docId if it satisfied either
         - DOCID_MD5_PAT
         - MD5_DOCID_PAT
    Otherwise, it return None
    """
    base_file_name = os.path.basename(file_name)
    result = split_docid_md5(base_file_name)
    if result:
        return result[0]
    return None

def get_docid_or_basename_prefix(file_name: str) -> str:
    """Return docId if it satisfied either
         - DOCID_MD5_PAT
         - MD5_DOCID_PAT
    Otherwise, it return base file name without '.txt' extension
    """
    base_file_name = os.path.basename(file_name)
    result = split_docid_md5(base_file_name)
    if result:
        return result[0]
    return base_file_name.replace('.txt', '')


def get_knorm_base_file_name(base_file_name: str) -> str:
    result = split_docid_md5(base_file_name)
    if result:
        docid, md5x, rest = result
        return '{}-{}{}'.format(docid, md5x, rest)
    return base_file_name


def get_knorm_file_name(full_file_name: str) -> str:
    dir_path, bname = os.path.split(full_file_name)
    knorm_bname = get_knorm_base_file_name(bname)
    return os.path.join(dir_path, knorm_bname)


def get_md5docid_base_file_name(base_file_name: str) -> str:
    result = split_docid_md5(base_file_name)
    if result:
        docid, md5x, rest = result
        return '{}-{}{}'.format(md5x, docid, rest)
    return base_file_name


def get_md5docid_file_name(full_file_name: str) -> str:
    dir_path, bname = os.path.split(full_file_name)
    md5docid_bname = get_md5docid_base_file_name(bname)
    return os.path.join(dir_path, md5docid_bname)
