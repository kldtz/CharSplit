#!/bin/bash

# run with caching will create all the cached files which are good for later components of the pipeline

# PROVFILE_DIR=dir-provfiles
# DATA_DIR=sample_data2.model
# WORK_DIR=sample_data2.feat
# MODEL_DIR=sample_data2.model
# SCUT_MODEL_DIR=sample_data2.scut.model
# CUSTOM_MODEL_DIR=sample_data2.custmodel

PROVFILE_DIR=dir-provfiles
WORK_DIR=dir-work
# MODEL_DIR=dir-model
SCUT_MODEL_DIR=dir-scut-model
CUSTOM_MODEL_DIR=dir-custom-model

./main.py --cmd split_provision_trte --provfiles_dir $PROVFILE_DIR --work_dir $WORK_DIR --model_dirs "$SCUT_MODEL_DIR"


rm -rf dir-provfiles

# below is doing this without caching

# split_provision_files.py --provisions "amending_agreement,arbitration,assign,change_control,choiceoflaw,confidentiality,date,equitable_relief,events_default,exclusivity,indemnify,insurance,jurisdiction,limliability,nonsolicit,party,preamble,renewal,sublicense,survival,term,termination" --docs sample_data2.txt.files --model_dirs "sample_data2.model"
