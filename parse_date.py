#!/usr/bin/env python3

import argparse

from kirke.ebrules import dates

if __name__ == '__main__':
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('file', help='input file')

    # pylint: disable=invalid-name
    args = parser.parse_args()

    # fname = args.file

    # pylint: disable=line-too-long
    st = 'THIS CONTRACT ORDER is issued pursuant to Agreement No. 10321404 effective July 22,  2011 between Entergy Nuclear Operations, Inc. (Entergy Nuclear Operations) and Sargent & Lundy Lie  ("Contractor").'

# pylint: disable=invalid-name
    result = dates.extract_dates_v2(st, 0)

    print(result)

    # pylint: disable=line-too-long
    st = 'BY THIS POWER OF ATTORNEY made on U kfWWf 2012, I CHRISTOPHER TUKE of 182 Lancaster Road, London Wll 1QU (the "Appointor") hereby  appoint PHILIP PRICE of 60 Lessar Avenue, London SW4 9HQ and MATTHEW HARRISON of  9a Woodland Avenue, Windsor, Berkshire SL4 4AG, each to act severally as my true and lawful attorney, agent and proxy (the "Attorney") with  full power and authority in my name and on my beh'

    result = dates.extract_dates_from_party_line(st)

    print(result)

    # pylint: disable=line-too-long
    st = '1. This is a lease made and entered into this ____ day of _______, 2008, by and between the Town  of Concrete, a Washington municipal corporation, hereinafter referred to as “Landlord”, and the East  Valley Community Care Team and Oasis Teen Shelter, both Was'

    result = dates.extract_dates_from_party_line(st)

    print(result)
