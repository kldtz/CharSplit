import os
import sys


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


def get_model_files(dir_name):
    return [f for f in os.listdir(dir_name)
            if os.path.isfile(os.path.join(dir_name, f)) and f.endswith('.pkl')]
