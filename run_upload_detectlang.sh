#!/bin/bash

python3 kirke/client/postfile2.py --url http://localhost:8000/detect-lang $1

python3 kirke/client/postfile2.py --url http://localhost:8000/detect-langs $1
