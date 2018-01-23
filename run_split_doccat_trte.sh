#!/bin/bash

PROVFILE_DIR=dir-provfiles
WORK_DIR=dir-work
MODEL_DIR=dir-model
SCUT_MODEL_DIR=dir-scut-model
CUSTOM_MODEL_DIR=dir-custom-model

./main.py --cmd split_doccat_trte --docs export-train.filelist --work_dir $WORK_DIR --model_dirs "$MODEL_DIR,$SCUT_MODEL_DIR" --custom_model_dir $CUSTOM_MODEL_DIR
