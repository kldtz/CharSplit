#!/bin/bash

# https://github.com/scikit-learn/scikit-learn/issues/5115,
# https://github.com/joblib/joblib/blob/master/doc/parallel.rst#bad-interaction-of-multiprocessing-and-third-party-libraries
export JOBLIB_START_METHOD="forkserver"

gunicorn --workers 4 --timeout 9600 --preload app:app
