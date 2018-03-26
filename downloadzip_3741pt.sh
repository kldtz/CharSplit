#!/bin/bash

# $1 = tmp/fail1.txt

python3 kirke/client/downloadzip.py --url http://localhost:8000/custom-train-export/cust_3741

# python3 kirke/client/postfile2.py --url http://localhost:8000/annotate-doc --lang $1

# python3 kirke/client/postfile2.py --url http://localhost:8000/annotate-doc --lang --doccat $1

# python3 kirke/client/postfile2.py --url http://localhost:8000/classify-doc $1

# python3 kirke/client/postfile2.py --url http://localhost:8000/detect-lang $1

# python3 kirke/client/postfile2.py --url http://localhost:8000/detect-langs $1

# python3 kirke/client/postfile2.py --url http://localhost:8000/annotate-doc sample_data2/19400.txt
# python3 kirke/client/postfile2.py -u http://localhost:8000/annotate-doc sample_data2/19600.txt
