#!/bin/bash

main.py --cmd test_one_annotator --provision $1 --docs "sample_data2.scut.model/$1_test_doclist.txt" --work_dir "sample_data2.feat" --custom_model_dir "sample_data2.custmodel" --model_file "sample_data2.scut.model/$1_scutclassifier.pkl"
