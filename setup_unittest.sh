# to create the xxx.tar.gz
# tar cvfz kirke_unit_test.v.0.1.tar.gz demo-pdf demo-txt export-train demo-validate dir-test-doc dir-scut-model eb_files_test

wget kirke_unit_test.v.0.1.tar.gz
tar xvfz kirke_unit_test.v.0.1.tar.gz
mkdir demo-out
# nosetests tests
# nosetests tests-ml

