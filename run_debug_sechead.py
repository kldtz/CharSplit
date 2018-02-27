#!/usr/bin/env python3

import argparse
import logging
from pprint import pprint
import sys
import warnings
import re

import nltk

from kirke.utils import strutils
from kirke.docstruct import secheadutils, footerutils
# import extract_sechead2, extract_sechead3, split_subsection_head3


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_nbsp():
    st = "3.4.   Changes to Purchase Orders. If Arrayit Diagnostics requests a change to a Purchase Order (a \"Change Order\") after such Purchase Order is accepted by Arrayit, Arrayit Diagnostics shall inform Arrayit about such Change Order as soon as possible. Arrayit shall use commercially reasonable efforts to accommodate such Change Order."

    pat = re.compile(r'4\.\s+Changes')

    m = pat.search(st)
    if m:
        print("found:")
    else:
        print("not found:")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Section Headings.')
    parser.add_argument("-v", "--verbosity", help="increase output verbosity")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    # parser.add_argument('--dir', default='data-300-txt', help='input directory for .txt files')
    # parser.add_argument('file', help='input file')

    args = parser.parse_args()

    # words = strutils.split_words('Hi, James.')
    words = strutils.split_words('Hi, James B. Scott.')
    print('words = {}'.format(words))
    xxx = footerutils.classify_line_page_number('11-23-1923')
    print('xxx = {}'.format(xxx))

    word = 'Capitalized'
    print('{}.istitle() = {}'.format(word, word.istitle()))
    word = 'CAP'
    print('{}.istitle() = {}'.format(word, word.istitle()))

    # fname = args.file

    # line = 'ARTICLES ARTICLE 1— BASIC TERMS'
    # line = 'Exhibit 10.1'
    # line = '16.02 Landlord’s Consent. Tenant will pay Landlord'

    # line = '16.02 Landlord’s Consent. Tenant will pay Landlord its reasonable fees and expenses paid to third parties incurred in connection with any act by Tenant which requires Landlord’s consent or approval under this Lease. Landlord will notify Tenant prior to spending sums in excess of $1,000 and afford Tenant the opportunity to withdraw the request.     '

    # line = '5.5. Manufacturing Changes.'
    # line = '2.1.   Obligation to Supply. Pursuant to the terms of this Agreement, Arrayit shall supply Arrayit'
    # line = '5.3.   Supply of Arrayit Diagnostics Supplied Raw Materials. Arrayit Diagnostics shall be '
    # line = '12000 Westheimer Rd'
    # line = '524 East Weddell'
    # line = '4. Reference to and Effect on the Loan Documents. '
    line = 'I.  Background, Definitions, Representations and Findings'
    line = 'II.  The Project Facilities.'
    line = 'III. Loan By Issuer; Loan Payments; Other Payments'
    line = 'IV. Additional Covenants Of Company'
    line = 'V. Redemption of Bonds'
    line = 'VI. Events Of Default And Remedies'
    line = 'VII. Miscellaneous'
    # line = '(xi)           Description of the Securities and the Indenture.'
    line = 'STELLARIS LLC and Q3 CONTRACTING , INC . jointly and severally , as Borrower    '
    line = '11.0% Senior Notes Due 2014'
    line = 'TABLE OF CONTENTS Page'
    line = 'U.S. BANK NATIONAL ASSOCIATION,'
    line = 'Section 3.4 Definitions 1'
    line = 'Indentured'
    line = 'V. Redemption of Bonds'
    line = 'V Redemption of Bonds'
    line = '(vii) Redemption of Bonds'    # semi-fail, detect, but incorrect prefix

    """
    _, prefix, sechead, split_idx = extract_sechead3(line)
    if prefix or sechead:
        print('[{}]\t[{}]\t||\t[{}]'.format(prefix, sechead,
                                      line[:split_idx]))
"""

    line = '12.4   Validity    If any term or conditio            '
    line = 'Section 1.14b: Lease Cancellation: This section of the June 1, 1994 lease is deleted from this lease.'
    line = '1.2   Product    “Products” shall mean any product, '

    line = line.replace('\xa0', ' ')

    """
    split_idx = split_subsection_head3(line.replace('\xa0', ' '))
    print("split_idx = {}".format(split_idx))
    print("before: [{}]".format(line[:split_idx]))
    print("after: [{}]".format(line[split_idx:]))
"""
    line = '3.2.2           Percentage Adjustment. The Monthly Base Rent shall...'

    line = 'H is a character'
    line = 'H Exhibit'
    line = '7.7. Complete Agreement.'
    prev_line = None
    prev_line = '16. Pari Passu Notes. xxx'
    line = '9'
    prev_line_idx = 22

    prev_line = '386-462-6801'
    prev_line_idx = -1
    line = 'Agreed to and Accepted:'

    prev_line = None
    line = 'Appendix A:'

    prev_line = ''
    prev_line_idx = -1
    line = 'G&I V Midwest Residential LLC (2)'
    line = 'C.3 STRATEGIC SALES & MARKETING AGREEMENT'

    line = 'Section 7. Reservation of Rights; Effect on Insolvency Proceeding. Nothing herein shall be construed...'

    prev_line = 'License Agreement'
    line = '1'

    prev_line = 'this is not going to match.'
    line = 'Recitals'

    prev_line = ''
    line = 'Exhibit H—Tenant Estoppel Certificate.'

    prev_line = '1.02'
    line = 'Annexes'

    prev_line = '1.2 Other Definitions'
    line = 'Term'

    prev_line = ''
    line = 'ARTICLE II. CERTAIN COVENANTS'

    prev_line = '1.'
    line = 'Engagement.  Subject to the Terms and Conditions of this...'

    prev_line = ''
    line = 'Exhibit B-2'

    prev_line = 'EXHIBITS'
    line = 'Exhibit A'

    prev_line = 'Exhibit'
    line = 'Exhibit A – Notice of Conversion'

    prev_line = ''
    line = 'A Navada Corporation'

    line = '(512) 944-6464 Mobile'

    prev_line = 'Background'
    line = 'A.'

    prev_line = '5.'
    # line = 'Operating     Requirements / Performance'
    line = 'OPERATING       REQUIREMENTS / PERFORMANCE'
    line = 'Compliance       with Applicable Statutes'

    prev_line = '1.6'
    line = 'DISTRIBUTOR       shall purchase the PRODUCTS from either ESK, ESK affiliates or from third...'

    prev_line = ''
    line = '(e)        Regulation S. The Securities will be offered and sold'

    line = '(d)   Severability.  In the event that any one or more of'
    line = '8.1 Indemnification by Tenant. Subject to the...'

    prev_line = "Definitions and Construction"
    line = '1.1 Definitions. As used in this Agreement, the following terms shall have the following definitions:'


    prev_line = 'ARTICLE       32'
    line = 'LANGUAGE,       EFFECTIVENESS OF CONTRACTAND MISCELLANEOUS   PROVISIONS'

    prev_line = '1.'
    # line = 'RECITALS'
    line = 'Security Instrument'

    # prev_line = 'Backgroun. xxx'
    # line = '2. RECITALS'

    prev_line = ''
    line = '5- Forecast, Ordering and Delivery....................................................................................................................................................................................................................................11'

    print('any section head?')
    sechead_type, prefix, sechead, split_idx = secheadutils.extract_sechead_v4(line, prev_line=prev_line, prev_line_idx=prev_line_idx)
    if prefix or sechead:
        if split_idx >= 0:
            print('<{}>\t[{}]\t[{}]\t[{}]\t{}'.format(line[:split_idx], sechead_type, prefix, sechead, split_idx))
        else:
            print('<{}>\t[{}]\t[{}]\t[{}]\t{}'.format(line, sechead_type, prefix, sechead, split_idx))            
    
    """
    _, prefix, sechead, split_idx = extract_sechead3(line)
    if prefix or sechead:
        print('[{}]\t[{}]\t||\t[{}]'.format(sechead_type, prefix, sechead,
                                            line[:split_idx]))       
"""
        
    
    """
    category, prefix, sechead, split_idx = extract_sechead2(line)

    if category and split_idx == -1:
        print('{}\t{}\t{}\t||\t{}'.format(category, prefix, sechead,
                                          line))
    elif category and split_idx != -1:
        print('{}\t{}\t{}\t||\t{}'.format(category, prefix, sechead,
                                          line[:split_idx]))        
    elif not category and len(line) < 40:
        #print('{}\t{}\t{}\t||\t{}'.format(category, prefix, sechead,
        #                                  line))
        pass
    else:
        pass
    """

    logging.info('Done.')
