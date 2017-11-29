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

2. Install the dependencies

This is for Ubuntu 14.04
```
sudo apt-get install libmysqlclient-dev
sudo apt-get install python3.4-dev python3-pip libxml2-dev libxslt1-dev python3-numpy python3-scipy 
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran libatlas-dev libatlas3gf-base

```

On Ubuntu 16.04, libatlas3gf-base can be skipped.  In general, we want the *optimized* libblas and liblapack.  Otherwise, scipy will be mucher slower than expected.

In order to verify which version of liblapack is activated, please do
```
sudo update-alternatives --config libblas.so.3
sudo update-alternatives --config liblapack.so.3
```

It's also possible to set the versions of liblapack directly
```
sudo update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3
sudo update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3
```

3. Setup the virtual environment

```
virtualenv -p python3 env
source env/bin/activate
pip install numpy
pip install scipy
pip install -r requirements.txt
python download_nltk.py
```

The need to install numpy and scip earlier than requirements.txt is because we haven't merged https://github.com/eBrevia/kirke/pull/18 .

4. running CoreNLP server

You can follow the instruction on xxx.

Here is another way to get thing up and running faster.

```
# go to a directory at the same level as kirke
cd ..
mkdir corenlp
wget https://s3.amazonaws.com/repo.ebrevia.com/repository/stanford-corenlp-3.7.0-models.jar
wget https://s3.amazonaws.com/repo.ebrevia.com/repository/stanford-corenlp-3.7.0.jar
cp ../extractor/docker/service/corenlp/run .
# remove "> /dev/null 2>&1" from the end of 'run' command file
./run
```

That terminal will be used for corenlp.

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
If you are getting an Extraction Error in the UI while running Kirke on OSX, try updating your startup.sh to
```
#!/bin/bash

export EB_MODELS='dir-scut-model'
gunicorn --timeout 1200 app:app
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
The likely reason is that the size of the matrix is 0, which is caused by not finding any positive examples in the data.  This is likely caused by mismatch in "provision" name.  Watch out for caching issue, which might use the old file, which can cause the mismatch.


## Tests

Because of outdated "ebrevia/learn" code, we can only do specific test, not the global one yet.  Use

```
nosetests tests/test1.py
```

for now.

