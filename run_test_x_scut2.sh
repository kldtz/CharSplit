#!/bin/bash

WORK_DIR=dir-work
MODEL_DIR=dir-model
SCUT_MODEL_DIR=dir-scut-model
CUSTOM_MODEL_DIR=dir-custom-model

./main.py --cmd test_one_annotator --provision $1 --docs "$SCUT_MODEL_DIR/$1_test_doclist.txt" --work_dir $WORK_DIR --custom_model_dir $CUSTOM_MODEL_DIR --model_file "$SCUT_MODEL_DIR/$1_scutclassifier.v1.2.1.pkl"
