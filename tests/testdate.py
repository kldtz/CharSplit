#!/usr/bin/env python3

import unittest

from kirke.ebrules import dates


class TestDateUtils(unittest.TestCase):

    def test_extract_dates(self):
        "Test extract_dates()"

        line = 'Amy is born on January 14, 2011 when it rained'
        alist = dates.extract_dates_v2(line, 0)
        start, end, date_st, dtype, norm = alist[0]
        self.assertEqual(norm,
                         '2011-01-14')
        self.assertEqual(date_st,
                         'January 14, 2011')

        line = 'Amy is born on 13.04.14 when it rained'
        alist = dates.extract_dates_v2(line, 0)
        start, end, date_st, dtype, norm = alist[0]
        self.assertEqual(norm,
                         '2014-04-13')
        self.assertEqual(date_st,
                         '13.04.14')

        line = 'Amy is born on 13/04/14 when it rained'
        alist = dates.extract_dates_v2(line, 0)
        start, end, date_st, dtype, norm = alist[0]
        self.assertEqual(norm,
                         '2014-04-13')
        self.assertEqual(date_st,
                         '13/04/14')

        line = 'Amy is born on 13-04-14 when it rained'
        alist = dates.extract_dates_v2(line, 0)
        start, end, date_st, dtype, norm = alist[0]
        self.assertEqual(norm,
                         '2014-04-13')
        self.assertEqual(date_st,
                         '13-04-14')

        line = 'Amy is born on 13-Apr-14 when it rained'
        alist = dates.extract_dates_v2(line, 0)
        start, end, date_st, dtype, norm = alist[0]
        self.assertEqual(norm,
                         '2014-04-13')
        self.assertEqual(date_st,
                         '13-Apr-14')

        line = 'Amy is born on 13Apr2014 when it rained'
        alist = dates.extract_dates_v2(line, 0)
        start, end, date_st, dtype, norm = alist[0]
        self.assertEqual(norm,
                         '2014-04-13')
        self.assertEqual(date_st,
                         '13Apr2014')


    def test_parse_date(self):
        "Test parse_date()"

        dnorm = dates.DateNormalizer()

        self.assertEqual(dnorm.parse_date('January 14, 2011'),
                         # {'year': 2011, 'month': 1, 'day': 14}
                         {'norm': {'date': '2011-01-14'}})
        self.assertEqual(dnorm.parse_date('Jan. 14, 201l'),
                         # {'year': 2011, 'month': 1, 'day': 14}
                         {'norm': {'date': '2011-01-14'}})
        self.assertEqual(dnorm.parse_date('this first day of JUNE, 2011'),
                         # {'year': 2011, 'month': 6, 'day': 1}
                         {'norm': {'date': '2011-06-01'}})
        self.assertEqual(dnorm.parse_date('the 1st day of September, 2011'),
                         # {'year': 2011, 'month': 9, 'day': 1}
                         {'norm': {'date': '2011-09-01'}})
        self.assertEqual(dnorm.parse_date('March, 2011'),
                         # {'year': 2011, 'month': 3}
                         {'norm': {'date': '2011-03-XX'}})
        self.assertEqual(dnorm.parse_date('this    day of    , 2004'),
                         # {'year': 2004}
                         {'norm': {'date': '2004-XX-XX'}})
        self.assertEqual(dnorm.parse_date('the first day of September'),
                         # {'month': 9, 'day': 1}
                         {'norm': {'date': 'XXXX-09-01'}})
        self.assertEqual(dnorm.parse_date('October ____, 2007'),
                         # {'month': 10, 'year': 2007}
                         {'norm': {'date': '2007-10-XX'}})
        self.assertEqual(dnorm.parse_date('June     , 2010'),
                         # {'month': 6, 'year': 2010}
                         {'norm': {'date': '2010-06-XX'}})
        self.assertEqual(dnorm.parse_date('____, 2007'),
                         {'norm': {'date': '2007-XX-XX'}})
        self.assertEqual(dnorm.parse_date('October [___], 2007'),
                         # {'month': 10, 'year': 2007}
                         {'norm': {'date': '2007-10-XX'}})
        self.assertEqual(dnorm.parse_date('the end of December 2009'),
                         # {'month': 12, 'year': 2009, 'day': 31}
                         {'norm': {'date': '2009-12-31'}})
        self.assertEqual(dnorm.parse_date('this end of JUNE, 2011'),
                         # {'year': 2011, 'month': 6, 'day': 30}
                         {'norm': {'date': '2011-06-30'}})

        self.assertEqual(dnorm.parse_date('1st day of May, 2011'),
                         {'norm': {'date': '2011-05-01'}})

        self.assertEqual(dnorm.parse_date('2011-53-20'),
                         None)
        self.assertEqual(dnorm.parse_date('2011-12-53'),
                         None)
        self.assertEqual(dnorm.parse_date('3011-12-01'),
                         None)


    def test_parse_uk_date(self):
        "Test parse_date()"

        dnorm = dates.DateNormalizer()

        self.assertEqual(dnorm.parse_date('13.04.14'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('13-04-14'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('13.04.2014'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('13-04-2014'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('13Apr2014'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('13-Apr-14'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('13-Apr-2014'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('13.Apr.2014'),
                         {'norm': {'date': '2014-04-13'}})

        # self.assertEqual(dnorm.parse_date('05.06.2014'),
        #                 {'norm': {'date': '2014-06-05'}})


    def test_parse_us_date(self):
        "Test parse_date()"

        dnorm = dates.DateNormalizer()

        self.assertEqual(dnorm.parse_date('04.13.14'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('04-13-14'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('04.13.2014'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('04-13-2014'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('Apr. 13, 2014'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('Apr-13-14'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('Apr-13-2014'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('Apr.13.2014'),
                         {'norm': {'date': '2014-04-13'}})

        self.assertEqual(dnorm.parse_date('05.06.2014'),
                         {'norm': {'date': '2014-05-06'}})


    def test_parse_date_to_fail(self):
        "Test parse_date which should fail."

        dnorm = dates.DateNormalizer()

        self.assertEqual(dnorm.parse_date('101020'),
                         None)

        # this returned 2010-10-12
        self.assertEqual(dnorm.parse_date('101012'),
                         None)

        self.assertEqual(dnorm.parse_date('1315'),
                         None)
