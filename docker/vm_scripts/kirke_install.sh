#!/bin/sh
docker login docker.io
docker pull ebreviainc/kirke:onprem
docker rm -f kirke
docker run --name kirke \
       -e ONPREM=1 \
       -v /eb_files:/eb_files:Z \
       -p 8000:8000 -d \
       --restart always \
       ebreviainc/kirke:onprem
