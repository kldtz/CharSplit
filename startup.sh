#!/bin/bash

gunicorn --workers 4 --timeout 9600 --preload app:app
