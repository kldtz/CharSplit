#!/bin/bash

# To add new unit test data and create the kirke_unit_test.v.0.8.9.tar.gz
#
#   - Download the latest kirke_unit_test.v.0.8.9.tar.gz into kirke_unit_test/ directory
#   - cd krike_unit_test/data/
#   - copy the new unit test files to kirke_unit_test/data/
#   - tar cvfz kirke_uni_test.v.0.8.9.tar.gz *
#   - upload kirke_uni_test.v.0.8.9.tar.gz to AWS S3 through GUI

# remove all cached files
rm -rf eb_files_test/kirke_tmp/ dir-work dir-scut-model *tar.gz*

wget https://s3.amazonaws.com/repo.ebrevia.com/repository/kirke_unit_test.v.0.8.9.tar.gz

tar xvfz kirke_unit_test.v.0.8.9.tar.gz

# Download the following file:
#
# s3://repo.ebrevia.com/repository/dir-sent-check.v.0.1.tar.gz
#
# tar xvfz dir-sent-check.v.0.1.tar.gz
#
# In the future, add S3 key so we can download this directly from S3 instead of
# manually.
