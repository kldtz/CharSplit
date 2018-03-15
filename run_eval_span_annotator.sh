#!/bin/bash

WORK_DIR=dir-work
MODEL_DIR=dir-scut-model
CUSTOM_MODEL_DIR=dir-custom-model

./main.py --cmd eval_span_annotator --provision $1 --candidate_type $2 --docs $3 --work_dir $WORK_DIR --model_dir $MODEL_DIR
