#!/usr/bin/env python

import argparse
import logging
from pprint import pprint
import sys
import warnings
import re

from pathlib import Path

from collections import defaultdict
import os

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

from kirke.eblearn import ebrunner, ebtrainer, provclassifier, scutclassifier
from kirke.eblearn import ebtext2antdoc, ebannotator
from kirke.utils import osutils, splittrte, ebantdoc, entityutils

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

st = 'AGREEMENT, made as of August , 1998, between CEDAR BROOK 11 CORPORATE CENTER, L.P., 1000 Eastpark Blvd., Cranbury, New Jersey 08512 “Landlord”; and CHRYSALIS DNX TRANSGENIC SCIENCE CORP., 301 College Road East, Princeton, New Jersey 08540, “Tenant”.' 

st2 = 'THIS INITIAL TERM ADVISORY AGREEMENT, effective as of July 1, 2012 (the “Agreement”), is between WELLS REAL ESTATE INVESTMENT TRUST II, INC., a Maryland corporation (the “Company”), and WELLS REAL ESTATE ADVISORY SERVICES II, LLC, a Georgia limited liability corporation (the “Advisor”).'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")

    args = parser.parse_args()

    pat_list = entityutils.extract_define_party(st)
    for pat in pat_list:
        print("pat = {}, {}, {}".format(pat[1], pat[2], pat[0]))

    pat_list = entityutils.extract_define_party(st2)
    for pat in pat_list:
        print("pat2 = {}, {}, {}".format(pat[1], pat[2], pat[0]))        
        
