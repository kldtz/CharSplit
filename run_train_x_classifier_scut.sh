#!/bin/bash

run_train_prov_classifier.py --provision $1 --docs "sample_data2.scut.model/$1_train_doclist.txt" --work_dir "sample_data2.feat" --model_dir "sample_data2.scut.model" --scut
