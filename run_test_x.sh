#!/bin/bash

main.py --cmd test_annotators --provisions $1 --docs "sample_data2.model/$1_test_doclist.txt" --work_dir "sample_data2.feat" --model_dir "sample_data2.model" --custom_model_dir "sample_data2.custmodel"
