# to create the xxx.tar.gz
# tar cvfz kirke_unit_test.v.0.8.5.tar.gz demo-pdf demo-txt export-train demo-validate dir-test-doc dir-scut-model eb_files_test cust_12345 cust_9 cands_json cust_10 cust_21 cust_22 cust_39 cust_42 data-myparty data-rate-table paragraph-tests paragraphs twoColTxt cust_555 cust_555-too-few cust_10-1best-div-zero data-myparty-fail dir-paracand

rm -rf eb_files_test/kirke_tmp/ dir-work dir-scut-model *tar.gz*

wget https://s3.amazonaws.com/repo.ebrevia.com/repository/kirke_unit_test.v.0.8.5.tar.gz

tar xvfz kirke_unit_test.v.0.8.5.tar.gz

# download the following file:
# s3://repo.ebrevia.com/repository/dir-sent-check.v.0.1.tar.gz

# tar xvfz dir-sent-check.v.0.1.tar.gz

