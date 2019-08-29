#!/bin/bash

# remove all cached files
rm -rf eb_files_test/kirke_tmp/ dir-work dir-scut-model *tar.gz*

wget https://s3.amazonaws.com/repo.ebrevia.com/repository/kirke_unit_test.v.0.8.11.tar.gz

tar xvfz kirke_unit_test.v.0.8.11.tar.gz

# Download the following file:
#
# s3://repo.ebrevia.com/repository/dir-sent-check.v.0.1.tar.gz
#
# tar xvfz dir-sent-check.v.0.1.tar.gz
#
# In the future, add S3 key so we can download this directly from S3 instead of
# manually.
