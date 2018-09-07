import configparser
from datetime import datetime
import fcntl
from fcntl import LOCK_EX, LOCK_SH
import os
import shutil
import sys
import tempfile
import time
# pylint: disable=unused-import
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

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
