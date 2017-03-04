# Kirke: eBrevia Document Annotation System

# What is this?

XXX is a document annotation system that annotate documents based on previous annotated examples.

## How to Building Classifier Models

Here are the steps to build classifier models:

1. Collect an annotated corpus and list the files into a file.  Let's call it '*change_control_file_list.txt*'.  The annotations related to these files are expect to end in '_.ant_' and should be in the same directory as those '*.txt' files.

2. Train the model.

```bash
run_train_x.sh change_control
```

## Top Level Commands

# Create the file lists that contain specific provisions
# We want to train a provision only on files annotated for that provision.
run_split_provision_files.sh

# now we have provision file lists, such as
sample_data2.model/party.doclist.txt
sample_data2.model/term.doclist.txt


# build the classifier, which will evaluate based on annotations, not just classifiers
run_train_x.sh party


# to simply test the annotators without training



# calls run_train_eval_provision.py, which calls ebtrainer.train_eval_annotator()
# 
run_train_x.sh party


# calls run_train_prov_classifier.py, which calls ebtrainer._train_classifier()
# this only trains the classifier on ALL data
run_train_x_classifier.sh party


# calls run_test_provision.py, which calls ebrunner.test_annotators()
# which loads all dev models
run_test_x.sh party
# this runs on all provision.
run_test_provisions.sh


# annotate one particular document using ebrunner
# annotate_file.py calls eb_runner.annotate_document()
run_annotate_doc.sh

## NOTE:
When the app server says:
```raise ValueError('unable to infer matrix dimensions')```
The likely reason is that the size of the matrix is 0, which is caused by not finding any positive examples in the data.  This is likely caused by mismatch in "provision" name.  Watch out for caching issue, which might use the old file, which caused the mismatch.


## Tests

Because of outdated ```ebrevia/learn``` code, we can only do

```nosetests tests/test1.py```

for now.

