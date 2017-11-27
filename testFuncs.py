#!/usr/bin/env python

import argparse
import logging
import re

from kirke.utils import mathutils


CORPPAT = r'(company|borrowers?|tenant|landlords?|tenent|parent|holders?|licensee|licensor|buyers?|holdings|issuers?|corporation|banks?|lenders?|administrative agents?|merger sub|operating partnership|advisor|advisers?|purchasers?|trustees?|agents?|executive|seller|consultant|employee|partnership|general partner|partner|partnership|contractor|service provider|marketing agent|supplier|lessee|subordination agent|escrow agent|transaction entities|guarantors?|producers?|assignee|swingline lender|liquidity provider|assignor|maker|investors?|lessor|manager|collateral agent|distributors?|servicer|payee|independent director|tenant|sub|op general partner|operating partnership general partner|paying agent|reit|employer|syndication agent|operator|owner|mutual holding company|mid-tier holding company|depositors?|manufacturers?|subsidiaries|loan parties|debtors?|customer|original lender|clients?|sales manager|transferor|originator|collateral custodian|backup servicer|undersigned|subsidiary|affiliates?|credit parties|credit party|representative|indenture trustee|management investors|grantee|sponsor|secured party|director|publisher|payor|class a pass through trustee|arrangers?|documentation agents?|hospital?|grantor|obligors?|noteholders?|master servicer|co-documentation agents?|designated borrower|buyer parent|party b|party a|pledgor|user|selling stockholders?|participants?|you|mortgagee|initial purchaser|syndication agent|subsidiary guarantors?|\S+ borrowers?|escrow issuers?|companies|\S+ resolver borrowers?|subscribers?|promissor|parties|holding company|shareholders?|note issuers?|guarantor company|guarantor companies|stockholders?|sub-advisers?|sub-advisors?)'

CORPPAT1 = re.compile(r'(\([^"“”\(]*(["“”](?=the )?{}\.?["“”])\))'.format(CORPPAT), re.IGNORECASE)
#	my $comp = $2;

CORPPAT2 = re.compile(r'(\((?=the )?{}\))'.format(CORPPAT), re.IGNORECASE)

#my $comp = $1;

CORPPAT4 = re.compile(r'(["“”](?=the )?{}\.?["“”])'.format(CORPPAT), re.IGNORECASE)

CORPPAT3 = re.compile(r'^\s*{}:\s*(.*)$'.format(CORPPAT), re.IGNORECASE)


def extract_define_party(line: str, start_offset=0):
    mat = CORPPAT3.search(line)
    pat1list = CORPPAT1.finditer(line)
    pat2list = CORPPAT2.finditer(line)
    pat4list = CORPPAT4.finditer(line)

    result = []
    if mat:
        result.append((mat.group(2), start_offset + mat.start(2), start_offset + mat.end(2)))
    if pat1list:
        for pat1 in pat1list:
            print("xxx1\t{}\t{}\t{}".format(pat1.start(1), pat1.end(1), pat1.group(1)))
            result.append((pat1.group(1), start_offset + pat1.start(1), start_offset + pat1.end(1)))
    if pat2list:
        for pat1 in pat2list:
            print("xxx2\t{}\t{}\t{}".format(pat1.start(1), pat1.end(1), pat1.group(1)))
            result.append((pat1.group(1), start_offset + pat1.start(1), start_offset + pat1.end(1)))
    if pat4list:
        for pat1 in pat4list:
            # print("pat1\t{}\t{}\t{}".format(pat1.start(1), pat1.end(1), pat1.group(1)))
            is_overlap = False
            for oldpat in result:
                if mathutils.start_end_overlap((oldpat[1], oldpat[2]), (start_offset + pat1.start(1), start_offset + pat1.end(1))):
                    is_overlap = True
            if not is_overlap:
                print("xxx4\t{}\t{}\t{}".format(pat1.start(1), pat1.end(1), pat1.group(1)))
                result.append((pat1.group(1), start_offset + pat1.start(1), start_offset + pat1.end(1)))
            

    #for atuple in result:
    #    print("xxx {}".format(atuple))

    return result


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

st = 'AGREEMENT, made as of August , 1998, between CEDAR BROOK 11 CORPORATE CENTER, L.P., 1000 Eastpark Blvd., Cranbury, New Jersey 08512 “Landlord”; and CHRYSALIS DNX TRANSGENIC SCIENCE CORP., 301 College Road East, Princeton, New Jersey 08540, “Tenant”.' 

st2 = 'THIS INITIAL TERM ADVISORY AGREEMENT, effective as of July 1, 2012 (the “Agreement”), is between WELLS REAL ESTATE INVESTMENT TRUST II, INC., a Maryland corporation (the “Company”), and WELLS REAL ESTATE ADVISORY SERVICES II, LLC, a Georgia limited liability corporation (the “Advisor”).'

st3 = 'AGREEMENT entered into effective 2/23/09 (start date) by and between Comverse Technology, lnc., a New York Corporation, on behalf of itself and its subsidiaries (the “Company”) and Joel Legon (“Employee”).'

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training models.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")

    args = parser.parse_args()

    """
    # pat_list = entityutils.extract_define_party(st)
    pat_list = extract_define_party(st)
    for pat in pat_list:
        print("pat = {}, {}, {}".format(pat[1], pat[2], pat[0]))

    # pat_list = entityutils.extract_define_party(st2)
    pat_list = extract_define_party(st2)
    for pat in pat_list:
        print("pat2 = {}, {}, {}".format(pat[1], pat[2], pat[0]))
"""

    # pat_list = entityutils.extract_define_party(st3)
    pat_list = extract_define_party(st3)
    for pat in pat_list:
        print("pat3 = {}, {}, {}".format(pat[1], pat[2], pat[0]))                
        
