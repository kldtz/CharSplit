#!/bin/bash

docker rm -f kirke
docker run --name kirke \
       -v /eb_files:/eb_files:Z \
       -p 8000:8000 -d \
       --restart always \
       ebreviainc/kirke:$1

#        ebreviainc/kirke:kirke-docker-5
