#!/bin/bash

WORK_DIR=dir-work
MODEL_DIR=dir-scut-model
CUSTOM_MODEL_DIR=dir-custom-model

# usage: run_train_annotator.sh effectivedate DATE
# usage: run_train_annotator.sh l_tenant_notice ADDRESS

if [ $# -eq 3 ]; then
    ./main.py --cmd train_span_annotator --provision $1 --candidate_types $2 --docs $3 --work_dir $WORK_DIR --model_dir $MODEL_DIR
elif [ $# -eq 2 ]; then
    ./main.py --cmd train_span_annotator --provision $1 --candidate_types $2 --work_dir $WORK_DIR --model_dir $MODEL_DIR
else
    echo "usage: run_train_annotator.sh effectivedate DATE"
    echo "usage: run_train_annotator.sh l_tenant_notice ADDRESS"
fi

