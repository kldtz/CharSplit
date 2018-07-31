#!/usr/bin/env perl

use strict;
use warnings;

while (<>) {
    chomp;

    my $fn = $_;

    my $fout = $fn;
    $fout =~ s/.ebdata/.ant/;

    print("\necho \"ebdata_to_ant.sh $fn $fout\"\n");
    print("run_ebdata_to_ant.sh $fn $fout\n");
}
