#!/bin/bash

# $1 = tmp/fail1.txt

python kirke/client/postfile2.py --url http://localhost:8000/annotate-doc $1

# python kirke/client/postfile2.py --url http://localhost:8000/annotate-doc --lang $1

# python kirke/client/postfileDocCatLang.py --url http://localhost:8000/annotate-doc --lang --doccat $1


# python kirke/client/postfile2.py --url http://localhost:8000/classify-doc $1

# python kirke/client/postfile2.py --url http://localhost:8000/detect-lang $1

# python kirke/client/postfile2.py --url http://localhost:8000/detect-langs $1

# python kirke/client/postfile2.py --url http://localhost:8000/annotate-doc sample_data2/19400.txt
# python kirke/client/postfile2.py -u http://localhost:8000/annotate-doc sample_data2/19600.txt
