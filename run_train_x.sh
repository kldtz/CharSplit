#!/bin/bash

# DATA_DIR=sample_data2.model
# WORK_DIR=sample_data2.feat
# MODEL_DIR=sample_data2.model
# CUSTOM_MODEL_DIR=sample_data2.custmodel

WORK_DIR=dir-work
MODEL_DIR=dir-model
SCUT_MODEL_DIR=dir-scut-model
CUSTOM_MODEL_DIR=dir-custom-model

main.py --cmd train_annotator --provision $1 --docs $MODEL_DIR/$1.doclist.txt --work_dir $WORK_DIR --model_dir $MODEL_DIR --custom_model_dir $CUSTOM_MODEL_DIR
