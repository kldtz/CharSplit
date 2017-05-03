#!/bin/bash

export EB_MODELS='dir-scut-model'
gunicorn --workers 2 --timeout 600 app:app
