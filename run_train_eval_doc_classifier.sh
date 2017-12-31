#!/bin/bash

# PROVFILE_DIR=dir-provfiles
WORK_DIR=dir-work
MODEL_DIR=dir-scut-model
SCUT_MODEL_DIR=dir-scut-model
CUSTOM_MODEL_DIR=dir-custom-model

./main.py --cmd train_eval_doc_classifier --docs export-train.valid.filelist --model_dir $MODEL_DIR --work_dir $WORK_DIR --custom_model_dir $CUSTOM_MODEL_DIR
