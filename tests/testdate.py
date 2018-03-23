#!/usr/bin/env python3

import unittest

from kirke.ebrules import dates

    
class TestDateUtils(unittest.TestCase):

    def test_parse_date(self):
        "Test parse_date()"

        dnorm = dates.DateNormalizer()

        self.assertEqual(dnorm.parse_date('January 14, 2011'),
                         {'year': 2011, 'month': 1, 'day': 14})
        self.assertEqual(dnorm.parse_date('January 14, 201l'),
                         {'year': 2011, 'month': 1, 'day': 14})
        self.assertEqual(dnorm.parse_date('January 14, 2o1l'),
                         {'year': 2011, 'month': 1, 'day': 14})

        self.assertEqual(dnorm.parse_date('this first day of JUNE, 2011'),
                         {'year': 2011, 'month': 6, 'day': 1})
        self.assertEqual(dnorm.parse_date('the 1st day of September, 2011'),
                         {'year': 2011, 'month': 9, 'day': 1})
        self.assertEqual(dnorm.parse_date('March, 2011'),
                         {'year': 2011, 'month': 3})
        self.assertEqual(dnorm.parse_date('this    day of    , 2004'),
                         {'year': 2004})                        
        self.assertEqual(dnorm.parse_date('the first day of September'),
                         {'month': 9, 'day': 1})
        self.assertEqual(dnorm.parse_date('October ____, 2007'),
                         {'month': 10, 'year': 2007})
        self.assertEqual(dnorm.parse_date('June     , 2010'),
                         {'month': 6, 'year': 2010})
        self.assertEqual(dnorm.parse_date('____, 2007'),
                         {'year': 2007})
        self.assertEqual(dnorm.parse_date('October [___], 2007'),
                         {'month': 10, 'year': 2007})
        self.assertEqual(dnorm.parse_date('the end of December 2009'),
                         {'month': 12, 'year': 2009, 'day': 31})                
        self.assertEqual(dnorm.parse_date('this end of JUNE, 2011'),
                         {'year': 2011, 'month': 6, 'day': 30})

