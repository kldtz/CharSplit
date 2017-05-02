#!/bin/bash

export EB_MODELS='dir-scut-model'
gunicorn --workers 4 --timeout 600 app:app
