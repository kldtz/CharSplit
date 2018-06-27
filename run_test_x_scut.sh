#!/bin/bash

WORK_DIR=dir-work
MODEL_DIR=dir-scut-model
SCUT_MODEL_DIR=dir-scut-model
CUSTOM_MODEL_DIR=dir-custom-model
OUT_DIR=test-out

if [ $# -eq 2 ]
then
    ./main.py --cmd test_annotators --provisions $1 --docs $SCUT_MODEL_DIR/$1_test_doclist.txt --work_dir $WORK_DIR --model_dir $MODEL_DIR --custom_model_dir $CUSTOM_MODEL_DIR --threshold $2 --out_dir $OUT_DIR
else
    ./main.py --cmd test_annotators --provisions $1 --docs $SCUT_MODEL_DIR/$1_test_doclist.txt --work_dir $WORK_DIR --model_dir $MODEL_DIR --custom_model_dir $CUSTOM_MODEL_DIR --out_dir $OUT_DIR
fi

