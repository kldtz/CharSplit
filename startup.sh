#!/bin/bash

gunicorn --timeout 600 app:app
