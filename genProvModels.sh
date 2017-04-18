#!/bin/bash

echo "generate model affiliates"
run_train_x_scut.sh affiliates > affiliates.stat.0415

echo "generate model agent_trustee_assign"
run_train_x_scut.sh agent_trustee_assign > agent_trustee_assign.stat.0415

echo "generate model agent_trustee_coc"
run_train_x_scut.sh agent_trustee_coc > agent_trustee_coc.stat.0415

echo "generate model agreementdate"
run_train_x_scut.sh agreementdate > agreementdate.stat.0415

echo "generate model amending_agreement"
run_train_x_scut.sh amending_agreement > amending_agreement.stat.0415

echo "generate model appraisal"
run_train_x_scut.sh appraisal > appraisal.stat.0415

echo "generate model arbitration"
run_train_x_scut.sh arbitration > arbitration.stat.0415

echo "generate model asset_transfer"
run_train_x_scut.sh asset_transfer > asset_transfer.stat.0415

echo "generate model assign"
run_train_x_scut.sh assign > assign.stat.0415

echo "generate model assign_landlord_owner"
run_train_x_scut.sh assign_landlord_owner > assign_landlord_owner.stat.0415

echo "generate model assign_lender"
run_train_x_scut.sh assign_lender > assign_lender.stat.0415

echo "generate model audit_rights"
run_train_x_scut.sh audit_rights > audit_rights.stat.0415

echo "generate model change_control"
run_train_x_scut.sh change_control > change_control.stat.0415

echo "generate model choiceoflaw"
run_train_x_scut.sh choiceoflaw > choiceoflaw.stat.0415

echo "generate model closing_date"
run_train_x_scut.sh closing_date > closing_date.stat.0415

echo "generate model coc_assign_other"
run_train_x_scut.sh coc_assign_other > coc_assign_other.stat.0415

echo "generate model confidentiality"
run_train_x_scut.sh confidentiality > confidentiality.stat.0415

echo "generate model consulting_fees"
run_train_x_scut.sh consulting_fees > consulting_fees.stat.0415

echo "generate model date"
run_train_x_scut.sh date > date.stat.0415

echo "generate model dispute_resolution"
run_train_x_scut.sh dispute_resolution > dispute_resolution.stat.0415

echo "generate model ea_agreement_termination"
run_train_x_scut.sh ea_agreement_termination > ea_agreement_termination.stat.0415

echo "generate model ea_base_salary"
run_train_x_scut.sh ea_base_salary > ea_base_salary.stat.0415

echo "generate model ea_confidential_info"
run_train_x_scut.sh ea_confidential_info > ea_confidential_info.stat.0415

echo "generate model ea_employee"
run_train_x_scut.sh ea_employee > ea_employee.stat.0415

echo "generate model ea_employer"
run_train_x_scut.sh ea_employer > ea_employer.stat.0415

echo "generate model ea_employment_term"
run_train_x_scut.sh ea_employment_term > ea_employment_term.stat.0415

echo "generate model ea_employment_termiantion"
run_train_x_scut.sh ea_employment_termiantion > ea_employment_termiantion.stat.0415

echo "generate model ea_engagement"
run_train_x_scut.sh ea_engagement > ea_engagement.stat.0415

echo "generate model ea_excess_parachute"
run_train_x_scut.sh ea_excess_parachute > ea_excess_parachute.stat.0415

echo "generate model ea_expense"
run_train_x_scut.sh ea_expense > ea_expense.stat.0415

echo "generate model ea_full_part_time"
run_train_x_scut.sh ea_full_part_time > ea_full_part_time.stat.0415

echo "generate model ea_inventions"
run_train_x_scut.sh ea_inventions > ea_inventions.stat.0415

echo "generate model ea_performance_bonus"
run_train_x_scut.sh ea_performance_bonus > ea_performance_bonus.stat.0415

echo "generate model ea_perks"
run_train_x_scut.sh ea_perks > ea_perks.stat.0415

echo "generate model ea_position"
run_train_x_scut.sh ea_position > ea_position.stat.0415

echo "generate model ea_reasonableness_rc"
run_train_x_scut.sh ea_reasonableness_rc > ea_reasonableness_rc.stat.0415

echo "generate model ea_restrictive_covenants"
run_train_x_scut.sh ea_restrictive_covenants > ea_restrictive_covenants.stat.0415

echo "generate model ea_section_409a"
run_train_x_scut.sh ea_section_409a > ea_section_409a.stat.0415

echo "generate model ea_signing_bonus"
run_train_x_scut.sh ea_signing_bonus > ea_signing_bonus.stat.0415

echo "generate model ea_standard_benefits"
run_train_x_scut.sh ea_standard_benefits > ea_standard_benefits.stat.0415

echo "generate model ea_stock_compensation"
run_train_x_scut.sh ea_stock_compensation > ea_stock_compensation.stat.0415

echo "generate model ea_termination_rights"
run_train_x_scut.sh ea_termination_rights > ea_termination_rights.stat.0415

echo "generate model effectivedate"
run_train_x_scut.sh effectivedate > effectivedate.stat.0415

echo "generate model equitable_relief"
run_train_x_scut.sh equitable_relief > equitable_relief.stat.0415

echo "generate model escrow"
run_train_x_scut.sh escrow > escrow.stat.0415

echo "generate model events_default"
run_train_x_scut.sh events_default > events_default.stat.0415

echo "generate model exclusivity"
run_train_x_scut.sh exclusivity > exclusivity.stat.0415

echo "generate model exhibit_appendix_complete"
run_train_x_scut.sh exhibit_appendix_complete > exhibit_appendix_complete.stat.0415

echo "generate model exhibit_appendix"
run_train_x_scut.sh exhibit_appendix > exhibit_appendix.stat.0415

echo "generate model force_majeure"
run_train_x_scut.sh force_majeure > force_majeure.stat.0415

echo "generate model guaranty"
run_train_x_scut.sh guaranty > guaranty.stat.0415

echo "generate model indemnify"
run_train_x_scut.sh indemnify > indemnify.stat.0415

echo "generate model insurance"
run_train_x_scut.sh insurance > insurance.stat.0415

echo "generate model jurisdiction"
run_train_x_scut.sh jurisdiction > jurisdiction.stat.0415

echo "generate model jury_trial"
run_train_x_scut.sh jury_trial > jury_trial.stat.0415

echo "generate model la_agent_appointment"
run_train_x_scut.sh la_agent_appointment > la_agent_appointment.stat.0415

echo "generate model la_agent_delegation"
run_train_x_scut.sh la_agent_delegation > la_agent_delegation.stat.0415

echo "generate model la_agent_exculpatory"
run_train_x_scut.sh la_agent_exculpatory > la_agent_exculpatory.stat.0415

echo "generate model la_agent_fees"
run_train_x_scut.sh la_agent_fees > la_agent_fees.stat.0415

echo "generate model la_agent_indemnification"
run_train_x_scut.sh la_agent_indemnification > la_agent_indemnification.stat.0415

echo "generate model la_agent_no_other_duties"
run_train_x_scut.sh la_agent_no_other_duties > la_agent_no_other_duties.stat.0415

echo "generate model la_agent_reliance"
run_train_x_scut.sh la_agent_reliance > la_agent_reliance.stat.0415

echo "generate model la_agent_resignation"
run_train_x_scut.sh la_agent_resignation > la_agent_resignation.stat.0415

echo "generate model la_agent_rights"
run_train_x_scut.sh la_agent_rights > la_agent_rights.stat.0415

echo "generate model la_agent_trustee"
run_train_x_scut.sh la_agent_trustee > la_agent_trustee.stat.0415

echo "generate model la_borrower"
run_train_x_scut.sh la_borrower > la_borrower.stat.0415

echo "generate model la_borrower_indemnification"
run_train_x_scut.sh la_borrower_indemnification > la_borrower_indemnification.stat.0415

echo "generate model la_borrowing_base"
run_train_x_scut.sh la_borrowing_base > la_borrowing_base.stat.0415

echo "generate model la_collateral"
run_train_x_scut.sh la_collateral > la_collateral.stat.0415

echo "generate model la_commitment_fees"
run_train_x_scut.sh la_commitment_fees > la_commitment_fees.stat.0415

echo "generate model l_ada"
run_train_x_scut.sh l_ada > l_ada.stat.0415

echo "generate model l_addl_rent"
run_train_x_scut.sh l_addl_rent > l_addl_rent.stat.0415

echo "generate model l_address"
run_train_x_scut.sh l_address > l_address.stat.0415

echo "generate model l_address_only"
run_train_x_scut.sh l_address_only > l_address_only.stat.0415

echo "generate model la_defaulting_lender"
run_train_x_scut.sh la_defaulting_lender > la_defaulting_lender.stat.0415

echo "generate model la_default_rate"
run_train_x_scut.sh la_default_rate > la_default_rate.stat.0415

echo "generate model la_"
run_train_x_scut.sh la_ > la_.stat.0415

echo "generate model la_expenses_payment"
run_train_x_scut.sh la_expenses_payment > la_expenses_payment.stat.0415

echo "generate model la_expiration_termination"
run_train_x_scut.sh la_expiration_termination > la_expiration_termination.stat.0415

echo "generate model la_financial_covenants"
run_train_x_scut.sh la_financial_covenants > la_financial_covenants.stat.0415

echo "generate model la_funding_loss_payments"
run_train_x_scut.sh la_funding_loss_payments > la_funding_loss_payments.stat.0415

echo "generate model la_guarantor"
run_train_x_scut.sh la_guarantor > la_guarantor.stat.0415

echo "generate model la_illegality"
run_train_x_scut.sh la_illegality > la_illegality.stat.0415

echo "generate model la_inability_determine_rates"
run_train_x_scut.sh la_inability_determine_rates > la_inability_determine_rates.stat.0415

echo "generate model la_increased_costs"
run_train_x_scut.sh la_increased_costs > la_increased_costs.stat.0415

echo "generate model la_information_reporting"
run_train_x_scut.sh la_information_reporting > la_information_reporting.stat.0415

echo "generate model la_interest_rate_margin"
run_train_x_scut.sh la_interest_rate_margin > la_interest_rate_margin.stat.0415

echo "generate model la_interest_rate_options"
run_train_x_scut.sh la_interest_rate_options > la_interest_rate_options.stat.0415

echo "generate model la_interlender"
run_train_x_scut.sh la_interlender > la_interlender.stat.0415

echo "generate model la_judgment_currency"
run_train_x_scut.sh la_judgment_currency > la_judgment_currency.stat.0415

echo "generate model la_lc_fees"
run_train_x_scut.sh la_lc_fees > la_lc_fees.stat.0415

echo "generate model la_lc_issuer"
run_train_x_scut.sh la_lc_issuer > la_lc_issuer.stat.0415

echo "generate model la_lender"
run_train_x_scut.sh la_lender > la_lender.stat.0415

echo "generate model la_lender_indemnification"
run_train_x_scut.sh la_lender_indemnification > la_lender_indemnification.stat.0415

echo "generate model la_letter_credit"
run_train_x_scut.sh la_letter_credit > la_letter_credit.stat.0415

echo "generate model la_loan_maturity"
run_train_x_scut.sh la_loan_maturity > la_loan_maturity.stat.0415

echo "generate model l_alterations"
run_train_x_scut.sh l_alterations > l_alterations.stat.0415

echo "generate model la_mitigation_obligations"
run_train_x_scut.sh la_mitigation_obligations > la_mitigation_obligations.stat.0415

echo "generate model la_optional_prepayment"
run_train_x_scut.sh la_optional_prepayment > la_optional_prepayment.stat.0415

echo "generate model la_patriot_act"
run_train_x_scut.sh la_patriot_act > la_patriot_act.stat.0415

echo "generate model la_payment_dates"
run_train_x_scut.sh la_payment_dates > la_payment_dates.stat.0415

echo "generate model la_payment_schedule"
run_train_x_scut.sh la_payment_schedule > la_payment_schedule.stat.0415

echo "generate model la_reduction_increase"
run_train_x_scut.sh la_reduction_increase > la_reduction_increase.stat.0415

echo "generate model la_revolving_loan_commitment"
run_train_x_scut.sh la_revolving_loan_commitment > la_revolving_loan_commitment.stat.0415

echo "generate model la_set_off"
run_train_x_scut.sh la_set_off > la_set_off.stat.0415

echo "generate model la_sharing_payment"
run_train_x_scut.sh la_sharing_payment > la_sharing_payment.stat.0415

echo "generate model la_swing_line"
run_train_x_scut.sh la_swing_line > la_swing_line.stat.0415

echo "generate model la_taxes"
run_train_x_scut.sh la_taxes > la_taxes.stat.0415

echo "generate model la_term_loan_commitment"
run_train_x_scut.sh la_term_loan_commitment > la_term_loan_commitment.stat.0415

echo "generate model la_use_proceeds"
run_train_x_scut.sh la_use_proceeds > la_use_proceeds.stat.0415

echo "generate model la_usury"
run_train_x_scut.sh la_usury > la_usury.stat.0415

echo "generate model la_voluntary_conversion"
run_train_x_scut.sh la_voluntary_conversion > la_voluntary_conversion.stat.0415

echo "generate model la_waiver_consequential_damages"
run_train_x_scut.sh la_waiver_consequential_damages > la_waiver_consequential_damages.stat.0415

echo "generate model l_brokers"
run_train_x_scut.sh l_brokers > l_brokers.stat.0415

echo "generate model l_building_naming"
run_train_x_scut.sh l_building_naming > l_building_naming.stat.0415

echo "generate model l_building_services"
run_train_x_scut.sh l_building_services > l_building_services.stat.0415

echo "generate model l_cam"
run_train_x_scut.sh l_cam > l_cam.stat.0415

echo "generate model l_commencement_date"
run_train_x_scut.sh l_commencement_date > l_commencement_date.stat.0415

echo "generate model l_competitor_restrictions"
run_train_x_scut.sh l_competitor_restrictions > l_competitor_restrictions.stat.0415

echo "generate model l_condemnation_term"
run_train_x_scut.sh l_condemnation_term > l_condemnation_term.stat.0415

echo "generate model l_contraction_opt"
run_train_x_scut.sh l_contraction_opt > l_contraction_opt.stat.0415

echo "generate model l_co_tenancy"
run_train_x_scut.sh l_co_tenancy > l_co_tenancy.stat.0415

echo "generate model l_cpi_adjust"
run_train_x_scut.sh l_cpi_adjust > l_cpi_adjust.stat.0415

echo "generate model l_credit"
run_train_x_scut.sh l_credit > l_credit.stat.0415

echo "generate model l_damage_term"
run_train_x_scut.sh l_damage_term > l_damage_term.stat.0415

echo "generate model l_default_rate"
run_train_x_scut.sh l_default_rate > l_default_rate.stat.0415

echo "generate model l_early_term"
run_train_x_scut.sh l_early_term > l_early_term.stat.0415

echo "generate model l_electric_charges"
run_train_x_scut.sh l_electric_charges > l_electric_charges.stat.0415

echo "generate model l_estoppel_cert"
run_train_x_scut.sh l_estoppel_cert > l_estoppel_cert.stat.0415

echo "generate model l_estoppel_cert_only"
run_train_x_scut.sh l_estoppel_cert_only > l_estoppel_cert_only.stat.0415

echo "generate model l_execution_date"
run_train_x_scut.sh l_execution_date > l_execution_date.stat.0415

echo "generate model l_expansion_opt"
run_train_x_scut.sh l_expansion_opt > l_expansion_opt.stat.0415

echo "generate model l_expiration_date"
run_train_x_scut.sh l_expiration_date > l_expiration_date.stat.0415

echo "generate model l_generator"
run_train_x_scut.sh l_generator > l_generator.stat.0415

echo "generate model l_go_dark"
run_train_x_scut.sh l_go_dark > l_go_dark.stat.0415

echo "generate model l_govt_compliance"
run_train_x_scut.sh l_govt_compliance > l_govt_compliance.stat.0415

echo "generate model l_guarantor"
run_train_x_scut.sh l_guarantor > l_guarantor.stat.0415

echo "generate model l_hazardous_material"
run_train_x_scut.sh l_hazardous_material > l_hazardous_material.stat.0415

echo "generate model l_holdover"
run_train_x_scut.sh l_holdover > l_holdover.stat.0415

echo "generate model lic_commercialization"
run_train_x_scut.sh lic_commercialization > lic_commercialization.stat.0415

echo "generate model lic_contractor"
run_train_x_scut.sh lic_contractor > lic_contractor.stat.0415

echo "generate model lic_enforcement"
run_train_x_scut.sh lic_enforcement > lic_enforcement.stat.0415

echo "generate model lic_extension"
run_train_x_scut.sh lic_extension > lic_extension.stat.0415

echo "generate model lic_grant_license"
run_train_x_scut.sh lic_grant_license > lic_grant_license.stat.0415

echo "generate model lic_ip"
run_train_x_scut.sh lic_ip > lic_ip.stat.0415

echo "generate model lic_licensee"
run_train_x_scut.sh lic_licensee > lic_licensee.stat.0415

echo "generate model lic_license_fee"
run_train_x_scut.sh lic_license_fee > lic_license_fee.stat.0415

echo "generate model lic_licensor"
run_train_x_scut.sh lic_licensor > lic_licensor.stat.0415

echo "generate model lic_milestone_payments"
run_train_x_scut.sh lic_milestone_payments > lic_milestone_payments.stat.0415

echo "generate model lic_ownership"
run_train_x_scut.sh lic_ownership > lic_ownership.stat.0415

echo "generate model lic_patent"
run_train_x_scut.sh lic_patent > lic_patent.stat.0415

echo "generate model lic_royalties"
run_train_x_scut.sh lic_royalties > lic_royalties.stat.0415

echo "generate model lic_scope"
run_train_x_scut.sh lic_scope > lic_scope.stat.0415

echo "generate model lic_software_license"
run_train_x_scut.sh lic_software_license > lic_software_license.stat.0415

echo "generate model lic_software_upgrade"
run_train_x_scut.sh lic_software_upgrade > lic_software_upgrade.stat.0415

echo "generate model lic_source_code"
run_train_x_scut.sh lic_source_code > lic_source_code.stat.0415

echo "generate model lic_stock_compensation"
run_train_x_scut.sh lic_stock_compensation > lic_stock_compensation.stat.0415

echo "generate model lic_taxes"
run_train_x_scut.sh lic_taxes > lic_taxes.stat.0415

echo "generate model lic_territory"
run_train_x_scut.sh lic_territory > lic_territory.stat.0415

echo "generate model lic_trademark"
run_train_x_scut.sh lic_trademark > lic_trademark.stat.0415

echo "generate model limliability"
run_train_x_scut.sh limliability > limliability.stat.0415

echo "generate model liquidated_damages"
run_train_x_scut.sh liquidated_damages > liquidated_damages.stat.0415

echo "generate model l_landlord_concessions"
run_train_x_scut.sh l_landlord_concessions > l_landlord_concessions.stat.0415

echo "generate model l_landlord_inspection"
run_train_x_scut.sh l_landlord_inspection > l_landlord_inspection.stat.0415

echo "generate model l_landlord_lessor"
run_train_x_scut.sh l_landlord_lessor > l_landlord_lessor.stat.0415

echo "generate model l_landlord_repair"
run_train_x_scut.sh l_landlord_repair > l_landlord_repair.stat.0415

echo "generate model l_landlord_work"
run_train_x_scut.sh l_landlord_work > l_landlord_work.stat.0415

echo "generate model l_lender_cure"
run_train_x_scut.sh l_lender_cure > l_lender_cure.stat.0415

echo "generate model l_lessee_eod"
run_train_x_scut.sh l_lessee_eod > l_lessee_eod.stat.0415

echo "generate model l_lessee_indem"
run_train_x_scut.sh l_lessee_indem > l_lessee_indem.stat.0415

echo "generate model l_lessor_eod"
run_train_x_scut.sh l_lessor_eod > l_lessor_eod.stat.0415

echo "generate model l_lessor_indem"
run_train_x_scut.sh l_lessor_indem > l_lessor_indem.stat.0415

echo "generate model l_light_air_easement"
run_train_x_scut.sh l_light_air_easement > l_light_air_easement.stat.0415

echo "generate model l_mutual_waiver_subrogation"
run_train_x_scut.sh l_mutual_waiver_subrogation > l_mutual_waiver_subrogation.stat.0415

echo "generate model l_no_lien"
run_train_x_scut.sh l_no_lien > l_no_lien.stat.0415

echo "generate model l_no_recordation_lease"
run_train_x_scut.sh l_no_recordation_lease > l_no_recordation_lease.stat.0415

echo "generate model loan_prepay"
run_train_x_scut.sh loan_prepay > loan_prepay.stat.0415

echo "generate model loan_term_ex"
run_train_x_scut.sh loan_term_ex > loan_term_ex.stat.0415

echo "generate model l_operating_escalation"
run_train_x_scut.sh l_operating_escalation > l_operating_escalation.stat.0415

echo "generate model l_parking"
run_train_x_scut.sh l_parking > l_parking.stat.0415

echo "generate model l_percentage_rent"
run_train_x_scut.sh l_percentage_rent > l_percentage_rent.stat.0415

echo "generate model l_permitted_use"
run_train_x_scut.sh l_permitted_use > l_permitted_use.stat.0415

echo "generate model l_porter_wage"
run_train_x_scut.sh l_porter_wage > l_porter_wage.stat.0415

echo "generate model l_premises"
run_train_x_scut.sh l_premises > l_premises.stat.0415

echo "generate model l_prohibited_uses"
run_train_x_scut.sh l_prohibited_uses > l_prohibited_uses.stat.0415

echo "generate model l_purchase_opt"
run_train_x_scut.sh l_purchase_opt > l_purchase_opt.stat.0415

echo "generate model l_quiet_enjoy"
run_train_x_scut.sh l_quiet_enjoy > l_quiet_enjoy.stat.0415

echo "generate model l_relocation"
run_train_x_scut.sh l_relocation > l_relocation.stat.0415

echo "generate model l_renewal_opt"
run_train_x_scut.sh l_renewal_opt > l_renewal_opt.stat.0415

echo "generate model l_renewal_rent"
run_train_x_scut.sh l_renewal_rent > l_renewal_rent.stat.0415

echo "generate model l_rent_abate"
run_train_x_scut.sh l_rent_abate > l_rent_abate.stat.0415

echo "generate model l_rent_adjust"
run_train_x_scut.sh l_rent_adjust > l_rent_adjust.stat.0415

echo "generate model l_rent_concession"
run_train_x_scut.sh l_rent_concession > l_rent_concession.stat.0415

echo "generate model l_rent"
run_train_x_scut.sh l_rent > l_rent.stat.0415

echo "generate model l_re_tax"
run_train_x_scut.sh l_re_tax > l_re_tax.stat.0415

echo "generate model l_right_first_offer"
run_train_x_scut.sh l_right_first_offer > l_right_first_offer.stat.0415

echo "generate model l_rooftop"
run_train_x_scut.sh l_rooftop > l_rooftop.stat.0415

echo "generate model l_rules"
run_train_x_scut.sh l_rules > l_rules.stat.0415

echo "generate model l_sale_premises"
run_train_x_scut.sh l_sale_premises > l_sale_premises.stat.0415

echo "generate model l_security"
run_train_x_scut.sh l_security > l_security.stat.0415

echo "generate model l_signage"
run_train_x_scut.sh l_signage > l_signage.stat.0415

echo "generate model l_snda"
run_train_x_scut.sh l_snda > l_snda.stat.0415

echo "generate model l_square_footage"
run_train_x_scut.sh l_square_footage > l_square_footage.stat.0415

echo "generate model l_substitute_space"
run_train_x_scut.sh l_substitute_space > l_substitute_space.stat.0415

echo "generate model l_surrender_demised_premises"
run_train_x_scut.sh l_surrender_demised_premises > l_surrender_demised_premises.stat.0415

echo "generate model l_tax_indem"
run_train_x_scut.sh l_tax_indem > l_tax_indem.stat.0415

echo "generate model l_tenant_assignment"
run_train_x_scut.sh l_tenant_assignment > l_tenant_assignment.stat.0415

echo "generate model l_tenant_improvements"
run_train_x_scut.sh l_tenant_improvements > l_tenant_improvements.stat.0415

echo "generate model l_tenant_lessee_dba"
run_train_x_scut.sh l_tenant_lessee_dba > l_tenant_lessee_dba.stat.0415

echo "generate model l_tenant_lessee"
run_train_x_scut.sh l_tenant_lessee > l_tenant_lessee.stat.0415

echo "generate model l_tenant_notice"
run_train_x_scut.sh l_tenant_notice > l_tenant_notice.stat.0415

echo "generate model l_tenant_repair"
run_train_x_scut.sh l_tenant_repair > l_tenant_repair.stat.0415

echo "generate model l_term"
run_train_x_scut.sh l_term > l_term.stat.0415

echo "generate model l_terminate_posess"
run_train_x_scut.sh l_terminate_posess > l_terminate_posess.stat.0415

echo "generate model l_time_essence"
run_train_x_scut.sh l_time_essence > l_time_essence.stat.0415

echo "generate model mediation"
run_train_x_scut.sh mediation > mediation.stat.0415

echo "generate model merger_price"
run_train_x_scut.sh merger_price > merger_price.stat.0415

echo "generate model misc_pricing"
run_train_x_scut.sh misc_pricing > misc_pricing.stat.0415

echo "generate model most_favored_nation"
run_train_x_scut.sh most_favored_nation > most_favored_nation.stat.0415

echo "generate model noncompete"
run_train_x_scut.sh noncompete > noncompete.stat.0415

echo "generate model non_disparagement"
run_train_x_scut.sh non_disparagement > non_disparagement.stat.0415

echo "generate model nonsolicit"
run_train_x_scut.sh nonsolicit > nonsolicit.stat.0415

echo "generate model note_preamble"
run_train_x_scut.sh note_preamble > note_preamble.stat.0415

echo "generate model notice"
run_train_x_scut.sh notice > notice.stat.0415

echo "generate model pagenum"
run_train_x_scut.sh pagenum > pagenum.stat.0415

echo "generate model party"
run_train_x_scut.sh party > party.stat.0415

echo "generate model preamble"
run_train_x_scut.sh preamble > preamble.stat.0415

echo "generate model pricing_salary"
run_train_x_scut.sh pricing_salary > pricing_salary.stat.0415

echo "generate model product_price"
run_train_x_scut.sh product_price > product_price.stat.0415

echo "generate model purchase_price"
run_train_x_scut.sh purchase_price > purchase_price.stat.0415

echo "generate model remedy"
run_train_x_scut.sh remedy > remedy.stat.0415

echo "generate model renewal"
run_train_x_scut.sh renewal > renewal.stat.0415

echo "generate model sechead"
run_train_x_scut.sh sechead > sechead.stat.0415

echo "generate model securities_transfer"
run_train_x_scut.sh securities_transfer > securities_transfer.stat.0415

echo "generate model service_price"
run_train_x_scut.sh service_price > service_price.stat.0415

echo "generate model sigdate"
run_train_x_scut.sh sigdate > sigdate.stat.0415

echo "generate model sublet"
run_train_x_scut.sh sublet > sublet.stat.0415

echo "generate model sublicense"
run_train_x_scut.sh sublicense > sublicense.stat.0415

echo "generate model subsechead"
run_train_x_scut.sh subsechead > subsechead.stat.0415

echo "generate model survival"
run_train_x_scut.sh survival > survival.stat.0415

echo "generate model term"
run_train_x_scut.sh term > term.stat.0415

echo "generate model termination"
run_train_x_scut.sh termination > termination.stat.0415

echo "generate model termterm_other"
run_train_x_scut.sh termterm_other > termterm_other.stat.0415

echo "generate model third_party_bene"
run_train_x_scut.sh third_party_bene > third_party_bene.stat.0415

echo "generate model title"
run_train_x_scut.sh title > title.stat.0415

echo "generate model warranty"
run_train_x_scut.sh warranty > warranty.stat.0415

