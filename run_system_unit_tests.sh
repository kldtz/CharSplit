#!/bin/bash

# assume corenlp server is running

# wget https://s3.amazonaws.com/repo.ebrevia.com/repository/kirke_unit_test.v.0.1.tar.gz
# tar xvfz kirke_unit_test.v.0.1.tar.gz

# assume startup_unittest.sh is running

nosetests tests
nosetests tests-ml

