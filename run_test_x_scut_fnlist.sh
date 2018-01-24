#!/bin/bash

WORK_DIR=dir-work
MODEL_DIR=dir-model
SCUT_MODEL_DIR=dir-scut-model
CUSTOM_MODEL_DIR=dir-custom-model

if [ $# -eq 2 ]
then
    ./main.py --cmd test_annotators --provisions $1 --docs $2 --work_dir $WORK_DIR --model_dir $SCUT_MODEL_DIR --custom_model_dir $CUSTOM_MODEL_DIR
else
    echo "usage run_test_x_scut_fnlist.sh [prov] [prov_test_doclist.txt]"
fi

