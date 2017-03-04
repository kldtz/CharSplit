# Kirke
eBrevia Document Annotator

![kirke image](http://repo.ebrevia.com/repository/kirke.jpg)

Kirke is a document annotator that annotates documents based on previous annotated examples.

## Setting up the development environment

1. Check out the code from github.com

```
git clone git@github.com:eBrevia/kirke.git Kirke
cd Kirke
```

2. Setup the virtual environment

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements
```

## How to annotate a document using Kirke?

Simple question, but it takes a while to get to that point.  We need to first train a few models first before we can annotate a document.  We have to first run a few commands in a terminal.

1. Download or copy the training corpus.  In my system, I copied them to "sample_data2" directory.

2. Get hold of the files for building each provision.  In this repo, these files can be generated from running the following command on "sample_data2.txt.files":
```
run_split_provision_trte.sh
```
3. We want to first train 3 provision annotators, "party", "date", and "change_control".

```
run_train_x.sh party
run_train_x.sh date
run_train_x.sh change_control
```

Each of the above commands can 5 to 15 minutes, depending on the parameters we passed to GridSearchCV.  The model files are create in "sample_data2.model"


4. Now we can finally annotate a document.

```
run_annoate_doc.sh
```

Note: We also have a faster training mode, which I called _shortcut_ models.  Many of the above scripts have corresponding files with "._scut" version, which take only up to 3 minutes to train each provision.  The created model files are stored in "sample_data2.scut.model"

## How to run Kieke in server mode?

After you have setup Kieke using command line mode, now we are ready to run Kirke in server mode.

1. Make sure you have a corenlp web server running.  In my system, I put the web server in ~/tools/corenlp-server directory.  Please note that you should first read the Confluence page on this topic on how to set this up. 

```
cd ~/tools/corenlp-server
startup.sh
```

1. Start the Kirke server.

```
cd Kirke
startup.sh
```

The server is runnig.

2. Annoate a document.  Go to a new terminal window, do

```
run_upload_annotate_doc.sh
```


## How to do custom training?

1. Obtain the data needed for training.  On my machine, there are in Kirke/custom_train.  The file list is in custom_train.txt.files.  These files will uploaded to server for training later.

2. The command line version is

```
run_train_cust_12345.sh
```

3. Please make sure Kirke server, "startup.sh", is still running.  The server-client version is

```
run_upload_train.sh
```

## How to test built models?

1. We can test the annotators using the following scripts:

```
run_test_x.sh change_control
```

or 

```
run_test_x_scut.sh change_control
```
for shortcut models.

## Can I build models with ALL the data and not do testing?

1. Yes, you can.

```
run_train_x_classifier.sh [party|date|change_control]
```

Normal training using "run_train_x.sh [party|date|change_control]" takes 1/5 of the data for testing.



## NOTE:
When the app server says:
```
raise ValueError('unable to infer matrix dimensions')
```
The likely reason is that the size of the matrix is 0, which is caused by not finding any positive examples in the data.  This is likely caused by mismatch in "provision" name.  Watch out for caching issue, which might use the old file, which caused the mismatch.


## Tests

Because of outdated "ebrevia/learn" code, we can only do

```
nosetests tests/test1.py
```

for now.

