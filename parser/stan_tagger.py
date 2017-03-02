def stan_tagger(text):
    import os
    import nltk
    java_path = "C:/Program Files/Java/jdk1.8.0_25/bin/java.exe"
    nltk.internals.config_java(java_path)
    os.environ['JAVAHOME'] = java_path
    first_file= 'C:/Users/LouVacca/stanford_ner/english.all.3class.distsim.crf.ser.gz'
    sec_file = 'C:/Users/LouVacca/stanford_ner/stanford-ner.jar'
    from nltk.tag.stanford import NERTagger
    st = NERTagger(first_file,sec_file,encoding='utf-8')
    a = st.tag(text.split())
    return a[0]
