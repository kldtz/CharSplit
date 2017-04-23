#!/usr/bin/env perl

use strict;
use warnings;

# 'euro borrowers', 'australian reovler borrowers'
my $CORPPAT = "(company|borrowers?|tenant|landlords?|tenent|parent|holders?|licensee|licensor|buyers?|holdings|issuers?|corporation|banks?|lenders?|administrative agents?|merger sub|operating partnership|advisor|advisers?|purchasers?|trustees?|agents?|executive|seller|consultant|employee|partnership|general partner|partner|partnership|contractor|service provider|marketing agent|supplier|lessee|subordination agent|escrow agent|transaction entities|guarantors?|producers?|assignee|swingline lender|liquidity provider|assignor|maker|investors?|lessor|manager|collateral agent|distributors?|servicer|payee|independent director|tenant|sub|op general partner|operating partnership general partner|paying agent|reit|employer|syndication agent|operator|owner|mutual holding company|mid-tier holding company|depositors?|manufacturers?|subsidiaries|loan parties|debtors?|customer|original lender|clients?|sales manager|transferor|originator|collateral custodian|backup servicer|undersigned|subsidiary|affiliates?|credit parties|credit party|representative|indenture trustee|management investors|grantee|sponsor|secured party|director|publisher|payor|class a pass through trustee|arrangers?|documentation agents?|hospital?|grantor|obligors?|noteholders?|master servicer|co-documentation agents?|designated borrower|buyer parent|party b|party a|pledgor|user|selling stockholders?|participants?|you|mortgagee|initial purchaser|syndication agent|subsidiary guarantors?|\\S+ borrowers?|escrow issuers?|companies|\\S+ resolver borrowers?|subscribers?|promissor|parties|holding company|shareholders?|note issuers?|guarantor company|guarantor companies|stockholders?|sub-advisers?|sub-advisors?)";

while (<>) {
    chomp;

    my $line = $_;
    my @cols = split(/\t/, $line);

    my $freq = $cols[0];
    my $st = $cols[1];    
    
    # if (/\(.*[\"\“]${CORPPAT}[\"\”]\s*\)/i) {
    if ($st =~ /\(?.*([\"\“\”](?=the )?${CORPPAT}\.?[\"\“\”]).*\)?/i) {
	my $comp = $2;
	print("ggg\t$line\t$comp\n");
    }
    elsif ($st =~ /\((?=the )?${CORPPAT}\)/i) {
	my $comp = $1;
	print("ggg2\t$line\t$comp\n");
    }
    # the \t should be replaces with ^ in production code
    elsif ($st =~ /^\s*${CORPPAT}:\s*/i) {
	my $comp = $1;
	print("ggg3\t$line\t$comp\n");
    }    
    else {
	print("---\t$line\n");
    }
}
