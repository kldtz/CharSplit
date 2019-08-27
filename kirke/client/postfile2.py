#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import sys
# pylint: disable=unused-import
from typing import Any, Dict

import requests


# pylint: disable=C0103
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='identify the language')
    parser.add_argument('-v', '--verbosity', help='increase output verbosity')
    parser.add_argument('-d', '--debug', action='store_true', help='print debug information')
    parser.add_argument('-u', '--url', help='url to post the file')
    parser.add_argument('-l', '--lang', action='store_true', help='to detect lang')
    parser.add_argument('--header', action='store_true', help='to print header')
    parser.add_argument('--doccat', action='store_true', help='to classify document')

    parser.add_argument('filename')

    args = parser.parse_args()
    if args.verbosity:
        print('verbosity turned on')
    if args.debug:
        isDebug = True

    url = 'http://127.0.0.1:8000/detect-lang'
    # use url='http://127.0.0.1:8000/detect-langs' to detect top langs with probabilities
    if args.url:
        url = args.url


    # payload = {}
    # payload = {'types': 'party'}
    # payload = {'types': 'party,change_control'}
    # payload = {'types': 'termination,term,confidentiality,cust_3566'}
    # payload = {'types': 'term'}
    # payload = {'types': 'l_tenant_notice'}
    # payload = {'types': 'term,cust_2253'}

    # pylint: disable=line-too-long
    # prov_list = ['bbb', 'affiliates', 'amending_agreement', 'arbitration', 'assign', 'audit_rights', 'change_control', 'choiceoflaw', 'confidentiality', 'date', 'ea_base_salary', 'ea_employee', 'ea_employer', 'ea_employment_termiantion', 'ea_employment_term', 'ea_engagement', 'ea_excess_parachute', 'ea_expense', 'ea_full_part_time', 'ea_performance_bonus', 'ea_perks', 'ea_position', 'ea_reasonableness_rc', 'ea_standard_benefits', 'ea_termination_rights', 'effectivedate', 'equitable_relief', 'events_default', 'exclusivity', 'force_majeure', 'guaranty', 'indemnify', 'insurance', 'jurisdiction', 'jury_trial', 'la_agent_appointment', 'la_agent_delegation', 'la_agent_fees', 'la_agent_no_other_duties', 'la_agent_reliance', 'la_agent_resignation', 'la_agent_rights', 'la_agent_trustee', 'la_borrower_indemnification', 'la_borrower', 'l_address_only', 'l_address', 'la_funding_loss_payments', 'la_illegality', 'la_increased_costs', 'la_lc_fees', 'la_lender_indemnification', 'la_lender', 'la_letter_credit', 'la_mitigation_obligations', 'la_optional_prepayment', 'la_patriot_act', 'la_payment_dates', 'la_reduction_increase', 'la_set_off', 'la_sharing_payment', 'la_swing_line', 'la_taxes', 'la_use_proceeds', 'la_waiver_consequential_damages', 'l_brokers', 'l_building_services', 'l_commencement_date', 'l_condemnation_term', 'l_damage_term', 'l_default_rate', 'l_early_term', 'l_electric_charges', 'l_estoppel_cert_only', 'l_estoppel_cert', 'l_execution_date', 'l_expansion_opt', 'l_expiration_date', 'l_holdover', 'lic_grant_license', 'lic_licensee', 'lic_licensor', 'limliability', 'l_landlord_lessor', 'l_landlord_repair', 'l_lessee_eod', 'l_lessor_eod', 'l_mutual_waiver_subrogation', 'l_no_lien', 'l_no_recordation_lease', 'l_operating_escalation', 'l_parking', 'l_permitted_use', 'l_premises', 'l_quiet_enjoy', 'l_renewal_opt', 'l_rent', 'l_re_tax', 'l_security', 'l_snda', 'l_square_footage', 'l_tenant_assignment', 'l_tenant_lessee', 'l_tenant_repair', 'l_term', 'l_time_essence', 'noncompete', 'nonsolicit', 'notice', 'party', 'pricing_salary', 'remedy', 'renewal', 'securities_transfer', 'sigdate', 'sublet', 'sublicense', 'survival', 'termination', 'term', 'third_party_bene', 'title', 'warranty', 'l_tenant_notice', 'cust_9', 'aaa', 'cust_300']

    # pylint: disable=line-too-long
    prov_list = ['affiliates', 'amending_agreement', 'arbitration', 'assign', 'audit_rights', 'change_control', 'choiceoflaw', 'confidentiality', 'date', 'ea_base_salary', 'ea_employee', 'ea_employer', 'ea_employment_termiantion', 'ea_employment_term', 'ea_engagement', 'ea_excess_parachute', 'ea_expense', 'ea_full_part_time', 'ea_performance_bonus', 'ea_perks', 'ea_position', 'ea_reasonableness_rc', 'ea_standard_benefits', 'ea_termination_rights', 'effectivedate', 'equitable_relief', 'events_default', 'exclusivity', 'force_majeure', 'guaranty', 'indemnify', 'insurance', 'jurisdiction', 'jury_trial', 'la_agent_appointment', 'la_agent_delegation', 'la_agent_fees', 'la_agent_no_other_duties', 'la_agent_reliance', 'la_agent_resignation', 'la_agent_rights', 'la_agent_trustee', 'la_borrower_indemnification', 'la_borrower', 'l_address_only', 'l_address', 'la_funding_loss_payments', 'la_illegality', 'la_increased_costs', 'la_lc_fees', 'la_lender_indemnification', 'la_lender', 'la_letter_credit', 'la_mitigation_obligations', 'la_optional_prepayment', 'la_patriot_act', 'la_payment_dates', 'la_reduction_increase', 'la_set_off', 'la_sharing_payment', 'la_swing_line', 'la_taxes', 'la_use_proceeds', 'la_waiver_consequential_damages', 'l_brokers', 'l_building_services', 'l_commencement_date', 'l_condemnation_term', 'l_damage_term', 'l_default_rate', 'l_early_term', 'l_electric_charges', 'l_estoppel_cert_only', 'l_estoppel_cert', 'l_execution_date', 'l_expansion_opt', 'l_expiration_date', 'l_holdover', 'lic_grant_license', 'lic_licensee', 'lic_licensor', 'limliability', 'l_landlord_lessor', 'l_landlord_repair', 'l_lessee_eod', 'l_lessor_eod', 'l_mutual_waiver_subrogation', 'l_no_lien', 'l_no_recordation_lease', 'l_operating_escalation', 'l_parking', 'l_permitted_use', 'l_premises', 'l_quiet_enjoy', 'l_renewal_opt', 'l_rent', 'l_re_tax', 'l_security', 'l_snda', 'l_square_footage', 'l_tenant_assignment', 'l_tenant_lessee', 'l_tenant_repair', 'l_term', 'l_time_essence', 'noncompete', 'nonsolicit', 'notice', 'party', 'pricing_salary', 'remedy', 'renewal', 'securities_transfer', 'sigdate', 'sublet', 'sublicense', 'survival', 'termination', 'term', 'third_party_bene', 'title', 'warranty', 'l_tenant_notice', 'cust_9.1247', 'korean']
    # 'l_tenant_notice',
    payload = {'types': ','.join(prov_list),
               'dev-mode': True}  # type: Dict[str, Any]

    """
    payload = {'types': 'term,cust_1089,cust_1102,cust_1115,cust_1116,cust_1117'
               'cust_1118,cust_1141,cust_1148,cust_1157,cust_1158,cust_1159'
               'cust_1161,cust_1176,cust_1177,cust_1178,cust_1229,cust_1254,cust_12'
               'cust_1392,cust_1394,cust_1399,cust_1460,cust_1473,cust_1474'
               'cust_1478,cust_1481,cust_1482,cust_1483,cust_1484,cust_3741.1,cust_12345'}

    payload = {'types': 'term,cust_9,cust_28'}
    """
    # payload = {'types': 'term,cust_9.1051,cust_28,cust_12345'}
    # cust_3741 has '.2'
    if args.lang:
        payload['detect-lang'] = True
    if args.doccat:
        payload['classify-doc'] = True

    txt_file = Path(args.filename)
    if txt_file.is_file() and args.filename.endswith('.txt'):
        offset_filename = args.filename.replace('.txt', '.offsets.json')
        if os.path.exists(offset_filename):
            files = [('file', open(args.filename, 'rt', encoding='utf-8')),
                     ('file', open(offset_filename, 'rt', encoding='utf-8'))]
        else:
            files = {'file': open(args.filename, 'rt', encoding='utf-8')}  # type: ignore

        resp = requests.post(url, files=files, data=payload)

        if args.header:
            print('status: [{}]'.format(resp.status_code))
            print(resp.headers)
        print(resp.text)
    else:
        print("file '{}' is not a valid file".format(args.filename), file=sys.stderr)
