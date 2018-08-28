#!/bin/bash

# https://github.com/scikit-learn/scikit-learn/issues/5115,
# https://github.com/joblib/joblib/blob/master/doc/parallel.rst#bad-interaction-of-multiprocessing-and-third-party-libraries

export EB_FILES=eb_files_test/
export EB_MODELS=dir-scut-model

# if ran before in this directory, to remove the cache
# rm -rf eb_files_test/kirke_tmp/ dir-work

export JOBLIB_START_METHOD="forkserver"
gunicorn --timeout 9600 app:app
