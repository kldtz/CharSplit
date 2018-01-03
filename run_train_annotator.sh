#!/bin/bash

WORK_DIR=dir-work
MODEL_DIR=dir-scut-model
CUSTOM_MODEL_DIR=dir-custom-model

./main.py --cmd train_span_annotator --provision $1 --work_dir $WORK_DIR --model_dir $MODEL_DIR
