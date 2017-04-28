#!/bin/bash

export EB_MODELS='dir-scut-model'
gunicorn --timeout 600 app:app
