# to create the xxx.tar.gz
# tar cvfz kirke_unit_test.v.0.2.3.tar.gz demo-pdf demo-txt export-train demo-validate dir-test-doc dir-scut-model eb_files_test cust_12345 cust_9 cands_json cust_10 cust_21 cust_22 cust_42 data-myparty

wget https://s3.amazonaws.com/repo.ebrevia.com/repository/kirke_unit_test.v.0.2.3.tar.gz
tar xvfz kirke_unit_test.v.0.2.3.tar.gz
# nosetests tests
# nosetests tests-ml

