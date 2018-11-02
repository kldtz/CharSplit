#!/bin/bash

# $1 = tmp/fail1.txt

python3 kirke/client/postfile2.py --url http://localhost:8000/annotate-doc $1

# python3 kirke/client/postfile2.py --url http://localhost:8000/annotate-doc --lang $1
