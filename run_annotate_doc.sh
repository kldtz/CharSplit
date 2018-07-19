#!/bin/bash

#!/bin/bash

# PROVFILE_DIR=dir-provfiles
WORK_DIR=dir-work
MODEL_DIR=dir-model
SCUT_MODEL_DIR=dir-scut-model
CUSTOM_MODEL_DIR=dir-custom-model

if [ $# -eq 2 ]
then
    ./main.py --cmd annotate_document --doc $1 --provisions $2 --work_dir $WORK_DIR --model_dir $SCUT_MODEL_DIR --custom_model_dir $CUSTOM_MODEL_DIR
else
    ./main.py --cmd annotate_document --doc $1 --work_dir $WORK_DIR --model_dir $SCUT_MODEL_DIR --custom_model_dir $CUSTOM_MODEL_DIR
fi
