import os
import sys


# Create a directory and any missing ancestor directories.
# If the directory already exists, do nothing.
# similar to distutils.dir_util import mkpath
def mkpath(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_last_cmd_line_arg():
    prefix = ''
    if len(sys.argv) > 1:
        prefix = sys.argv[-1]
    return prefix


def get_model_files(dir):
    return [f for f in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, f)) and f.endswith('.pkl')]
