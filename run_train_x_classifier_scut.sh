#!/bin/bash

./main.py --cmd train_classifier --provision $1 --docs "sample_data2.scut.model/$1_train_doclist.txt" --work_dir "sample_data2.feat" --model_dir "sample_data2.scut.model"  --custom_model_dir "sample_data2.custmodel" --scut
