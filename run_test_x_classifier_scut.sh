#!/bin/bash

run_test_prov_classifier.py --provision $1 --docs "sample_data2.scut.model/$1_test_doclist.txt" --work_dir "sample_data2.feat" --model_file "sample_data2.scut.model/$1_scutclassifier.pkl"
