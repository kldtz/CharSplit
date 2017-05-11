#!/bin/bash

export EB_MODELS='dir-scut-model'
gunicorn --workers 4 --timeout 1200 --preload app:app
