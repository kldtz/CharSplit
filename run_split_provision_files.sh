#!/bin/bash

# run with caching will create all the cached files which are good for later components of the pipeline

split_provision_files.py --provisions "amending_agreement,arbitration,assign,change_control,choiceoflaw,confidentiality,date,equitable_relief,events_default,exclusivity,indemnify,insurance,jurisdiction,limliability,nonsolicit,party,preamble,renewal,sublicense,survival,term,termination" --work_dir "sample_data2.feat" --docs sample_data2.txt.files --model_dir "sample_data2.model"


# below is doing this without caching

# split_provision_files.py --provisions "amending_agreement,arbitration,assign,change_control,choiceoflaw,confidentiality,date,equitable_relief,events_default,exclusivity,indemnify,insurance,jurisdiction,limliability,nonsolicit,party,preamble,renewal,sublicense,survival,term,termination" --docs sample_data2.txt.files --model_dir "sample_data2.model"
