#!/bin/bash

export EB_MODELS='dir-scut-model'
# gunicorn --workers 2 --timeout 600 app:app
# gunicorn --timeout 600 app:app

gunicorn --timeout 1200 -k gevent --worker-connections 4 app:app
