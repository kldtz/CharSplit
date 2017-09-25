#!/bin/bash

gunicorn --workers 4 --timeout 1200 --preload app:app
