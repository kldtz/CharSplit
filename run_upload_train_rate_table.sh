#!/bin/bash

python3 -m kirke.client.postfileutils \
        --cmd uploaddir \
        --provision rate_table \
        --candidate_types TABLE \
        data-rate-table 
