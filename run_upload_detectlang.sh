#!/bin/bash

python kirke/client/postfile2.py --url http://localhost:8000/detect-lang $1

python kirke/client/postfile2.py --url http://localhost:8000/detect-langs $1
