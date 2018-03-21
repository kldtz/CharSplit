#!/bin/bash

export JOBLIB_START_METHOD="forkserver"
gunicorn --workers 4 --timeout 9600 --preload app:app
