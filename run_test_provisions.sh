#!/bin/bash

# We use party test file for now since we have no general test file.
# This test all the annotators in ebrunner.
run_test_provisions.py --docs "sample_data2.model/party_test_doclist.txt" --work_dir "sample_data2.feat" --model_dir "sample_data2.model"

