#!/bin/bash

#!/bin/bash

# PROVFILE_DIR=dir-provfiles
WORK_DIR=dir-work
MODEL_DIR=dir-model
SCUT_MODEL_DIR=dir-party-model
CUSTOM_MODEL_DIR=dir-custom-model

main.py --cmd annotate_doc_party --docs $1 --work_dir $WORK_DIR --model_dir $SCUT_MODEL_DIR --custom_model_dir $CUSTOM_MODEL_DIR