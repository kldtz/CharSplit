# CharSplit - An *ngram*-based compound splitter for German

Splits a German compound into its body and head, e.g.
> Autobahnraststätte -> Autobahn - Raststätte

Implementation of the method described in the appendix of the thesis:

Tuggener, Don (2016). *Incremental Coreference Resolution for German.* University of Zurich, Faculty of Arts.

### TL;DR
The method calculates probabilities of ngrams occurring at the beginning, end and in the middle of words and identifies the most likely position for a split.

The method achieves ~95% accuracy for head detection on the [Germanet compound test set](http://www.sfs.uni-tuebingen.de/lsd/compounds.shtml).

A model is provided, trained on 1 Mio. German nouns from Wikipedia.

### Usage 
### Train a new model:
```
$ python char_split_train.py <your_train_file>
```
where `<your_train_file>` contains one word (noun) per line.

### Compound splitting

From command line:
```
$ python char_split.py <word>
```
Outputs all possible splits, ranked by their score, e.g.
```
$ python char_split.py Autobahnraststätte
0.84096566854	Autobahn	Raststätte
-0.54568851959	Auto	Bahnraststätte
-0.719082070993	Autobahnrast	Stätte
...
```


As a module:
```
$ python
>>> import char_split
>>> char_split.split_compound('Autobahnraststätte')
[[0.7945872450631273, 'Autobahn', 'Raststätte'],
 [-0.7143290887876655, 'Auto', 'Bahnraststätte'],  
 [-1.1132332878581173, 'Autobahnrast', 'Stätte'],  
 [-1.4010051533086552, 'Aut', 'Obahnraststätte'],  
 [-2.3447843979244944, 'Autobahnrasts', 'Tätte'],  
 [-2.4761904761904763, 'Autobahnra', 'Ststätte'],  
 [-2.4761904761904763, 'Autobahnr', 'Aststätte'],  
 [-2.5733333333333333, 'Autob', 'Ahnraststätte'],  
 [-2.604651162790698, 'Autobahnras', 'Tstätte'],  
 [-2.7142857142857144, 'Autobah', 'Nraststätte'],  
 [-2.730248306997743, 'Autobahnrastst', 'Ätte'],  
 [-2.8033113109925973, 'Autobahnraststä', 'Tte'],  
 [-3.0, 'Autoba', 'Hnraststätte']]
```

### Document splitting

From command line:
```
$ python doc_split.py <dict>
```
Reads everything from standard input
and writes out the same, with the best splits
separated by the middle dot character `·`.

Each word is split as many times as possible based
on the file <dict>, which contains German words
one per line (comment lines beginning with # are allowed).

The name of the default dictionary is in the file `doc_config.py`.

Note that the `doc_split` module retains a cache of words already split,
so long documents will typically be processed proportionately faster
than short ones.
The cache is discarded when the program ends.
```
$ python sentence1.txt
Um die in jeder Hinsicht zufriedenzustellen, tüftelt er einen Weg aus,
sinnlose Bürokratie wie Ladenschlußgesetz und Nachtbackverbot auszutricksen.  
$ python doc_split.py <sentence1.txt  
Um die in jeder Hinsicht zufriedenzustellen, tüftelt er einen Weg aus,
sinnlose Bürokratie wie Laden·schluß·gesetz und Nacht·back·verbot auszutricksen.  
```

As a module:
```
$ python
>>> import doc_split
>>> # Constant containing a middle dot
>>> doc_split.MIDDLE_DOT
'·'
>>> # Split a word as much as possible, return a list
>>> doc_split.maximal_split('Verfassungsschutzpräsident')
['Verfassungs', 'Schutz', 'Präsident']
>>> # Split a word as much as possible, return a word with middle dots
'Verfassungs·schutz·präsident'
>>> # Split all splittable words in a sentence
>>> doc_split.doc_split('Der Marquis schlug mit dem Handteller auf sein Regiepult.')
Der Marquis schlug mit dem Hand·teller auf sein Regie·pult.
```
### Document splitting server

Because of the startup time, you can run the document splitter
as a simple server, and the responses will be quicker.
```
$ python doc_server <dict> <port>
```
The server will load `<dict>` and listen on `<port>`.
The client must
send the raw data in UTF-8 encoding to the port
and close the write side of the port, and the
server will return the split data.
Because of Python restrictions, the server is single-threaded.

The default dictionary and port are in the file `doc_config.py`.

A trivial client is provided:
```
$ python doc_client <port> <host>
```
Reads a document from standard input,
send it to the server running on `<host>` and `<port>`,
and send the server's output to standard output.
Thus it has the same interface as `doc_split`
(except that the dictionary cannot be specified),
but should run somewhat faster.

The default host and port are in the file `doc_config.py`.

## Downloading dictionaries
To download German dictionaries for `doc_split` and `doc_server`:
```
$ sh getdicts
```
This will download the German spelling plugins from the LibreOffice site,
extract the wordlists, and write five files into the current directory.
It leaves a good many files in `/tmp`, which are not needed further.
  * The dictionaries `de-DE.dic`, `de-AT.dic`, and `de-CH.dic` are
    fairly extensive (about 250,000 words each)
    and provide current German, Austrian, and Swiss spelling.
  * The file `de-1901.dic` provides the spelling used between 1901 and 1996.
  * The file `de-mixed.dic` is a merger of all four of the other files.

If the program is not splitting hard enough for your purposes,
you may want to find and use a smaller dictionary.
