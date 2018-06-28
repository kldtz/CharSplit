FROM phusion/baseimage:0.10.0

# update package repository sources and accept oracle license
RUN apt-get update \
  && apt-get -y install software-properties-common \
  && add-apt-repository ppa:webupd8team/java \
  && echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true \
  | /usr/bin/debconf-set-selections

RUN apt-get update \
  && apt-get -y install python3 python3-pip \
  libxml2-dev libxslt1-dev python3-dev python3-setuptools \
  libatlas-dev libatlas3-base oracle-java8-installer \
  libatlas-base-dev libblas-dev liblapack-dev gfortran \
  zlib1g-dev \
  && rm -rf /var/cache/oracle-jdk8-installer \
  && apt-get clean

# After remove python3-numpy python3-scipy, there is 2x speed penalty probably due to
# less optimized operations.  As a result, need to include following packages to
# enable the optimizations.
RUN pip3 install numpy==1.13.0 pandas==0.20.1 scipy==0.19.0 scikit-learn==0.18.1 lxml==3.4.4

# configure java 8 oracle jdk
RUN update-alternatives --display java \
  && echo "JAVA_HOME=/usr/lib/jvm/java-8-oracle" >> /etc/environment

#------------------------------------------------
# install dependencies that are unlikely to change as often in separate step to cache

# python
RUN update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3
RUN update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3

# install UTF locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8

# add CoreNLP models
WORKDIR /var/www/ebrevia-corenlp
RUN wget https://s3.amazonaws.com/repo.ebrevia.com/repository/stanford-corenlp-3.7.0-models.jar \
    && wget https://s3.amazonaws.com/repo.ebrevia.com/repository/stanford-corenlp-3.7.0.jar \
    && wget https://s3.amazonaws.com/repo.ebrevia.com/repository/StanfordCoreNLP.properties \
    && wget https://s3.amazonaws.com/repo.ebrevia.com/repository/stanford-spanish-corenlp-2016-10-31-models.jar \
    && wget https://s3.amazonaws.com/repo.ebrevia.com/repository/stanford-french-corenlp-2016-10-31-models.jar \
    && wget https://s3.amazonaws.com/repo.ebrevia.com/repository/portuguese-ner.ser.gz \
    && wget https://s3.amazonaws.com/repo.ebrevia.com/repository/stanford-chinese-corenlp-2016-10-31-models.jar 


#------------------------------------------------
# all downloading is done, start configuration
# NO SLOW STUFF BELOW THIS LINE, should all be setting or copying

ENV EB_FILES=/eb_files/
ENV EB_MODELS=/eb_models/

#------------------------------------------------
# add specified config for local or production

# install python app
COPY target/kirke/requirements.txt /var/www/ebrevia-python/
WORKDIR /var/www/ebrevia-python/
RUN pip3 install -r requirements.txt
COPY target/kirke/download_nltk.py /var/www/ebrevia-python/
RUN python3 download_nltk.py

WORKDIR $EB_MODELS
RUN wget https://s3.amazonaws.com/repo.ebrevia.com/repository/eb_models_2.0.6.tar.gz \
    && tar -zxf eb_models_2.0.6.tar.gz \
    && rm eb_models_2.0.6.tar.gz


# install service start scripts for phusion/baseimage
COPY docker/service /etc/service/
RUN mkdir -p /etc/my_init.d
ADD docker/bin/startup.sh /etc/my_init.d/startup.sh

COPY target/kirke /var/www/ebrevia-python/

#------------------------------------------------

ENV OCR_SERVER_URL=http://ocrrecognitionserve-env.us-east-1.elasticbeanstalk.com/Recognition4WS/RSSoapService.asmx?WSDL

EXPOSE 8000

VOLUME /eb_files

CMD ["/sbin/my_init"]
