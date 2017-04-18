#!/usr/bin/env perl

use strict;
use warnings;

print("#!/bin/bash\n\n");

while (<>) {
    chomp;

    print("echo \"generate model $_\"\n");
    print("run_train_x_scut.sh $_ > $_.stat.0415\n\n");
}
