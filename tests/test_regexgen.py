#!/usr/bin/env python3

import re
import unittest
from typing import Dict, List, Optional, Pattern, Tuple

from kirke.utils import ebsentutils
from kirke.sampleutils import regexgen


def tuv(adict: Dict) -> Dict:
    print('adict= {}'.format(adict))
    out_dict = {'text': adict['text'],
                'value': adict['norm']['value']}
    if adict['norm'].get('unit'):
        out_dict['unit'] = adict['norm']['unit']
    return out_dict


class TestCurrency(unittest.TestCase):

    def test_currency(self):
        "Test CURRENCY_PAT"

        line = "Bob received 33 dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual({'text': '33 dollars',
                          'unit': 'USD',
                          'value': 33},
                         tuv(mat_list[0]))

        """
        line = "Bob received 33. dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33. dollars',
                                       'unit': 'USD',
                                       'value': 33})
        """

        line = "Bob received 33.5 dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.5 dollars',
                                            'unit': 'USD',
                                            'value': 33.5})

        line = "Bob received 33.55 dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual({'text': '33.55 dollars',
                          'unit': 'USD',
                          'value': 33.55},
                         tuv(mat_list[0]))

        # disallow 33B for now
        """
        line = "Bob received 33B dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33B dollars',
                                            'unit': 'USD',
                                            'value': 33000000000})
        """

        # disallow '33 B', '33.3 M' for now
        """
        line = "Bob received 33 B dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33 B dollars',
                                            'unit': 'USD',
                                            'value': 33000000000})

        line = "Bob received 33.3 M dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.3 M dollars',
                                            'unit': 'USD',
                                            'value': 33299999.999999996})
        # 'value': 33300000})

        line = "Bob received 33.33 M dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.33 M dollars',
                                            'unit': 'USD',
                                            'value': 33330000})

        # TODO, this is a little weird
        # we intentionally want to be more inclusive, so
        # didn't check for \b
        line = "Bob received 33.444 M dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.444 M dollars',
                                            'unit': 'USD',
                                            'value': 33444000.000000004})
        # 'value': 33444000}
        """

        line = "Bob received 333,333  dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '333,333  dollars',
                                            'unit': 'USD',
                                            'value': 333333})

        line = "Bob received 333,333.2 million dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '333,333.2 million dollars',
                                            'unit': 'USD',
                                            'value': 333333200000})



        line = "Bob received $333,333.20 from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '$333,333.20',
                                            'unit': 'USD',
                                            'value': 333333.2})

        line = "Bob received €333,333 from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '€333,333',
                                            'unit': 'EUR',
                                            'value': 333333})

        line = "Bob received 333,333 Euros from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '333,333 Euros',
                                            'unit': 'EUR',
                                            'value': 333333})

        line = "Bob received 333,333 € from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '333,333 €',
                                            'unit': 'EUR',
                                            'value': 333333})

        line = "Bob received 333,333€ from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '333,333€',
                                            'unit': 'EUR',
                                            'value': 333333})


        line = "Bob received 333,333 from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 0)

        line = "Bob received -333,333 dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '-333,333 dollars',
                                            'unit': 'USD',
                                            'value': -333333})

        line = 'I have $35 (Thirty-Five Dollars).'
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(2, len(mat_list))
        self.assertEqual({'text': '$35',
                          'unit': 'USD',
                          'value': 35},
                         tuv(mat_list[0]))
        self.assertEqual({'text': 'Thirty-Five Dollars',
                          'unit': 'USD',
                          'value': 35},
                         tuv(mat_list[1]))

        line = 'I have 35 dollars.'
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(1, len(mat_list))
        self.assertEqual({'text': '35 dollars',
                          'unit': 'USD',
                          'value': 35},
                         tuv(mat_list[0]))

        line = "Bob received USD 33 from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'USD 33',
                                            'unit': 'USD',
                                            'value': 33})


        line = "Bob received USD   33 from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'USD   33',
                                            'unit': 'USD',
                                            'value': 33})


        line = "Bob received USD   33.33 from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'USD   33.33',
                                            'unit': 'USD',
                                            'value': 33.33})

        line = "Bob received Rs   3 from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'Rs   3',
                                            'unit': 'INR',
                                            'value': 3})

        line = "Bob received Rs.   3 from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'Rs.   3',
                                            'unit': 'INR',
                                            'value': 3})

        line = "Bob received 3  Rs from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '3  Rs',
                                            'unit': 'INR',
                                            'value': 3})

        line = "Bob received 3 Rs. from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '3 Rs',
                                            'unit': 'INR',
                                            'value': 3})

        line = "Bob received 33.33 Rupees from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.33 Rupees',
                                            'unit': 'INR',
                                            'value': 33.33})

        line = "Bob received 33.33 Rupee from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.33 Rupee',
                                            'unit': 'INR',
                                            'value': 33.33})

        line = "Bob received INR 33.33 from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'INR 33.33',
                                            'unit': 'INR',
                                            'value': 33.33})


        line = "Bob received 33.33  INR from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.33  INR',
                                            'unit': 'INR',
                                            'value': 33.33})

        line = 'Rs.50,000.00 (Rupees Fifty Thousand only)'
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 2)
        self.assertEqual(tuv(mat_list[0]), {'text': 'Rs.50,000.00',
                                            'unit': 'INR',
                                            'value': 50000})
        self.assertEqual(tuv(mat_list[1]), {'text': 'Rupees Fifty Thousand',
                                            'unit': 'INR',
                                            'value': 50000})

        line = '$1,000,000 base'
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '$1,000,000',
                                            'unit': 'USD',
                                            'value': 1000000})

        line = '$11,000,000  of General Liability Insurance ($1,000,000 base + $10,000,00 umbrella)  covering:'
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 3)
        self.assertEqual(tuv(mat_list[0]), {'text': '$11,000,000',
                                            'unit': 'USD',
                                            'value': 11000000})
        self.assertEqual(tuv(mat_list[1]), {'text': '$1,000,000',
                                            'unit': 'USD',
                                            'value': 1000000})
        self.assertEqual({'text': '$10,000,00',
                          'unit': 'USD',
                          'value': 1000000}, tuv(mat_list[2]))

        line = "one and half pound and three and half pound, eight and half dollars three and half million dollars"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 4)
        self.assertEqual(tuv(mat_list[0]), {'text': 'one and half pound',
                                            'unit': 'GBP',
                                            'value': 1.5})
        self.assertEqual(tuv(mat_list[1]), {'text': 'three and half pound',
                                            'unit': 'GBP',
                                            'value': 3.5})
        self.assertEqual(tuv(mat_list[2]), {'text': 'eight and half dollars',
                                            'unit': 'USD',
                                            'value': 8.5})
        self.assertEqual(tuv(mat_list[3]), {'text': 'three and half million dollars',
                                            'unit': 'USD',
                                            'value': 3500000})


    def test_extract_currency_prefer_prefix(self):
        line = """AnnualInstallment

1-12

$

28.25

$

114,104.10

$

1,369,249.25

13-24

$

28.75

$

116,123.65

$"""
        mat_list = regexgen.extract_currencies(line)
        currency_st_list = []
        for mat in mat_list:
            currency_st_list.append(mat['text'])

        target_list = ['$\n\n28.25',
                       '$\n\n114,104.10',
                       '$\n\n1,369,249.25',
                       '$\n\n28.75',
                       '$\n\n116,123.65']

        self.assertEqual(target_list,
                         currency_st_list)


    def test_number(self):
        "Test NUMBER_PAT"

        line = 'I have $35 (Thirty-Five Dollars).'
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(1, len(mat_list))
        self.assertEqual(tuv(mat_list[0]), {'text': 'Thirty-Five',
                                            'value': 35})

        mat_list = regexgen.extract_numbers(line, is_ignore_currency_symbol=True)
        self.assertEqual(2, len(mat_list))
        self.assertEqual(tuv(mat_list[0]), {'text': '35',
                                            'value': 35})
        self.assertEqual(tuv(mat_list[1]), {'text': 'Thirty-Five',
                                            'value': 35})

        line = 'I have 35 dollars.'
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(1, len(mat_list))
        self.assertEqual(tuv(mat_list[0]), {'text': '35',
                                            'value': 35})

        line = "33.3 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.3',
                                            'value': 33.3})

        line = "3.3 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '3.3',
                                            'value': 3.3})

        line = "I got .3 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '.3',
                                            'value': 0.3})

        line = ".3 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '.3',
                                            'value': 0.3})


        line = "0.3 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '0.3',
                                            'value': 0.3})


        line = "-0.3 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '-0.3',
                                            'value': -0.3})

        line = "-333,333.3 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '-333,333.3',
                                            'value': -333333.3})

        line = "-22,333.3 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '-22,333.3',
                                            'value': -22333.3})

        line = "Bob received 33 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33',
                                            'value': 33})

        line = "Bob received 33.3 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.3',
                                            'value': 33.3})

        # TODO
        # this is probably not do right behavior.
        # In particular, this conflict with '33.3M' in
        # the next example
        line = "Bob received 33.3 M dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual({'text': '33.3',
                          'value': 33.3},
                         tuv(mat_list[0]))

        line = "Bob received 33.3M dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual({'text': '33.3M',
                          'value': 33299999.999999996},
                         tuv(mat_list[0]))

        line = "Bob received 33.3802 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.3802',
                                            'value': 33.3802})

        """
        line = "Bob received 33. dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33. dollars',
                                       'value': 33})
        """


        line = "Bob received .3802 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '.3802',
                                            'value': 0.3802})

        line = "Bob received 0.3802 dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '0.3802',
                                            'value': 0.3802})

        line = "Bob received (0.3802) dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '0.3802',
                                            'value': 0.3802})


        line = "Bob received ten dollars from Alice"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'ten',
                                            'value': 10})

        line = "one and half pound and three and half pound, eight and half dollars three and half million dollars"
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(4, len(mat_list))
        self.assertEqual({'text': 'one and half',
                          'value': 1.5}, tuv(mat_list[0]))
        self.assertEqual({'text': 'three and half',
                          'value': 3.5}, tuv(mat_list[1]))
        self.assertEqual({'text': 'eight and half',
                          'value': 8.5}, tuv(mat_list[2]))
        self.assertEqual({'text': 'three and half million',
                          'value': 3500000}, tuv(mat_list[3]))


    def test_extract_number_paren_numbers(self):
        line = 'subsection 3.2 in a case in which Optionee dies within three (3) months after Optionee is Terminated'
        dict_list = regexgen.extract_number_paren_numbers(line)
        self.assertEquals(1, len(dict_list))
        self.assertEqual(3,
                         dict_list[0]['norm']['value'])


    def test_extract_ordinal_numbers(self):
        line = 'subsection 9th months after Optionee is Terminated'
        dict_list = regexgen.extract_ordinal_numbers(line)
        self.assertEquals(1, len(dict_list))
        self.assertEqual(9,
                         dict_list[0]['norm']['value'])

        line = 'subsection twelveth months after Optionee is Terminated'
        dict_list = regexgen.extract_ordinal_numbers(line)
        self.assertEquals(1, len(dict_list))
        self.assertEqual(12,
                         dict_list[0]['norm']['value'])


    def test_extract_numbers_difficult(self):
        line = '$\n\n1,369,249.25\n\n13-24'
        mat_list = regexgen.extract_numbers(line)
        self.assertEqual(1, len(mat_list))
        self.assertEqual({'text': '1,369,249.25',
                          'value': 1369249.25},
                         tuv(mat_list[0]))


    def test_percent(self):
        "Test PERCENT_PAT"

        line = "33.3 percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.3 percent',
                                            'value': 33.3,
                                            'unit': '%'})

        line = "33.3% from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.3%',
                                            'value': 33.3,
                                            'unit': '%'})

        """
        line = "33.3percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.3percent',
                                       'value': 33.3,
                                       'unit': '%'})
        """

        line = "33.3 % from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.3 %',
                                            'value': 33.3,
                                            'unit': '%'})

        line = "3.3 percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '3.3 percent',
                                            'value': 3.3,
                                            'unit': '%'})

        line = ".3 percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '.3 percent',
                                            'value': 0.3,
                                            'unit': '%'})

        line = "0.3 percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '0.3 percent',
                                            'value': 0.3,
                                            'unit': '%'})

        line = "-0.3 percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '-0.3 percent',
                                            'value': -0.3,
                                            'unit': '%'})

        line = "-333,333.3 percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '-333,333.3 percent',
                                            'value': -333333.3,
                                            'unit': '%'})

        line = "-22,333.3 percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '-22,333.3 percent',
                                            'value': -22333.3,
                                            'unit': '%'})

        line = "Bob received 33 percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33 percent',
                                            'value': 33,
                                            'unit': '%'})


        line = "Bob received 33.3 percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33.3 percent',
                                            'value': 33.3,
                                            'unit': '%'})

        """
        line = "Bob received 33.  percent from Alice"
        self.assertEqual(extract_str(percent_pat, line, 2),
                         '33.  percent')
        """

        line = "Bob received .3802 percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '.3802 percent',
                                            'value': 0.3802,
                                            'unit': '%'})

        line = "Bob received 0.3802 percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '0.3802 percent',
                                            'value': 0.3802,
                                            'unit': '%'})

        line = "Bob received ten percent from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'ten percent',
                                            'value': 10,
                                            'unit': '%'})


    def test_fractions(self):
        line = "Bob received 18/60th of the share from Alice"
        mat_list = regexgen.extract_fractions(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '18/60th',
                                            'value': 0.3})


        line = "Bob received 18/60 th of the share from Alice"
        mat_list = regexgen.extract_fractions(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '18/60 th',
                                            'value': 0.3})

        line = 'schedule  one-fourth on April 9, 2015'
        mat_list = regexgen.extract_fractions(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'one-fourth',
                                            'value': 0.25})

    def test_fraction_percent(self):
        line = "Bob received 33 1/3% of the share from Alice"
        mat_list = regexgen.extract_fraction_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33 1/3%',
                                            'value': 33.33,
                                            'unit': '%'})

        line = "Bob received 33⅝% of the share from Alice"
        mat_list = regexgen.extract_fraction_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33⅝%',
                                            'value': 33.625,
                                            'unit': '%'})

    def test_percent_hard(self):
        line = "Bob received 33 1/3% of the share from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33 1/3%',
                                            'value': 33.33,
                                            'unit': '%'})

        line = "Bob received 33⅝% of the share from Alice"
        mat_list = regexgen.extract_percents(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33⅝%',
                                            'value': 33.625,
                                            'unit': '%'})


    def test_time_duration(self):
        line = 'the expiration of 12 months after the date'
        mat_list = regexgen.extract_time_durations(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '12 months',
                                            'value': 12,
                                            'unit': 'month'})

        line = 'the expiration of three months after the date'
        mat_list = regexgen.extract_time_durations(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'three months',
                                            'value': 3,
                                            'unit': 'month'})

        line = 'the expiration of twelve (12) months after the date'
        mat_list = regexgen.extract_time_durations(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'twelve (12) months',
                                            'value': 12,
                                            'unit': 'month'})

        line = 'the expiration of three (3) months after the date'
        mat_list = regexgen.extract_time_durations(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'three (3) months',
                                            'value': 3,
                                            'unit': 'month'})

        line = 'the expiration of three (3 plus) months after the date'
        mat_list = regexgen.extract_time_durations(line)
        self.assertEqual(len(mat_list), 0)


    def test_word_currency(self):
        "Test CURRENCY_PAT"

        line = "Bob received 1 pound from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '1 pound',
                                            'value': 1,
                                            'unit': 'GBP'})

        line = "Bob received one pound from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'one pound',
                                            'value': 1,
                                            'unit': 'GBP'})

        line = "Bob received thirty-three dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': 'thirty-three dollars',
                                            'value': 33,
                                            'unit': 'USD'})

        # disallowed 33M for now
        """
        line = "Bob received 33M dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33M dollars',
                                            'value': 33000000,
                                            'unit': 'USD'})

        line = "Bob received 33 M dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33 M dollars',
                                            'value': 33000000,
                                            'unit': 'USD'})

        line = "Bob received 33B dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33B dollars',
                                            'value': 33000000000,
                                            'unit': 'USD'})

        line = "Bob received 33 B dollars from Alice"
        mat_list = regexgen.extract_currencies(line)
        self.assertEqual(len(mat_list), 1)
        self.assertEqual(tuv(mat_list[0]), {'text': '33 B dollars',
                                            'value': 33000000000,
                                            'unit': 'USD'})
        """