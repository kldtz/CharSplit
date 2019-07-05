#!/usr/bin/env python3

from typing import Dict, Tuple

import unittest

from kirke.nlputil import dates_jp, regexcand_jp


def date_flatten(adict: Dict, line: str) -> Tuple[int, int, str, str]:
    start, end = adict['start'], adict['end']
    date_st, norm = adict['text'], adict['norm']['date']
    return start, end, date_st, norm


class TestDateJPUtils(unittest.TestCase):

    def test_extract_dates_jp(self):
        "Test extract_dates_jp()"

        line = '2019年5月22日'
        alist = dates_jp.extract_dates(line)
        start, end, date_st, norm = date_flatten(alist[0], line)
        self.assertEqual(norm,
                         '2019-05-22')
        self.assertEqual(date_st,
                         '2019年5月22日')

        # test double-bytes
        line = '２０１９年５月２２日'
        alist = dates_jp.extract_dates(line, is_norm_dbcs_sbcs=True)
        start, end, date_st, norm = date_flatten(alist[0], line)
        self.assertEqual(norm,
                         '2019-05-22')
        self.assertEqual(date_st,
                         '２０１９年５月２２日')

        line = '令和元年5月22日'
        alist = dates_jp.extract_dates(line)
        start, end, date_st, norm = date_flatten(alist[0], line)
        self.assertEqual(norm,
                         '2019-05-22')
        self.assertEqual(date_st,
                         '令和元年5月22日')

        line = '令和元年５月２２日'
        alist = dates_jp.extract_dates(line, is_norm_dbcs_sbcs=True)
        start, end, date_st, norm = date_flatten(alist[0], line)
        self.assertEqual(norm,
                         '2019-05-22')
        self.assertEqual(date_st,
                         '令和元年５月２２日')


    def test_extract_dates_arabic(self):

        line = '2019/5/22'
        alist = dates_jp.extract_dates(line)
        start, end, date_st, norm = date_flatten(alist[0], line)
        self.assertEqual(norm,
                         '2019-05-22')
        self.assertEqual(date_st,
                         '2019/5/22')

        # double-bytes
        line = '２０１９／５／２２'
        alist = dates_jp.extract_dates(line, is_norm_dbcs_sbcs=True)
        start, end, date_st, norm = date_flatten(alist[0], line)
        self.assertEqual(norm,
                         '2019-05-22')
        self.assertEqual(date_st,
                         '２０１９／５／２２')

class TestCurrencyJPUtils(unittest.TestCase):

    def test_extract_currency_jp(self):
        "Test extract_dates_jp()"

        line = '二千零四￥'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(2004, adict['norm']['value'])
        self.assertEqual('JPY', adict['norm']['unit'])

        line = '2004￥'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(2004, adict['norm']['value'])
        self.assertEqual('JPY', adict['norm']['unit'])

        line = 'JPY 2004'
        alist = regexcand_jp.extract_currencies(line)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(2004, adict['norm']['value'])
        self.assertEqual('JPY', adict['norm']['unit'])

        line = '二千零四ドル'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(2004, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '二千零四㌦'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(2004, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '二千零四$'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(2004, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '二千零四＄'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(2004, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '340円'
        alist = regexcand_jp.extract_currencies(line)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(340, adict['norm']['value'])
        self.assertEqual('JPY', adict['norm']['unit'])

        line = '￥340'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(340, adict['norm']['value'])
        self.assertEqual('JPY', adict['norm']['unit'])

        line = '¥340'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(340, adict['norm']['value'])
        self.assertEqual('JPY', adict['norm']['unit'])

        line = '340YEN'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(340, adict['norm']['value'])
        self.assertEqual('JPY', adict['norm']['unit'])

        line = '３４０ＹＥＮ'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(340, adict['norm']['value'])
        self.assertEqual('JPY', adict['norm']['unit'])

        line = 'JPY340'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(340, adict['norm']['value'])
        self.assertEqual('JPY', adict['norm']['unit'])

        line = 'ＪＰＹ３４０'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(340, adict['norm']['value'])
        self.assertEqual('JPY', adict['norm']['unit'])

        line = '3.40ドル'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '3.40米ドル'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '3.40アメリカドル'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '3.40㌦'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '$3.40'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '＄３．４０'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '3.40弗'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '3.40USD'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '３．４０ＵＳＤ'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '3.40ユーロ'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('EUR', adict['norm']['unit'])

        line = '€3.40'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('EUR', adict['norm']['unit'])

        line = 'EUR3.40'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('EUR', adict['norm']['unit'])

        line = 'ＥＵＲ３．４０'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('EUR', adict['norm']['unit'])

        line = '3.40ポンド'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('GBP', adict['norm']['unit'])

        line = '￡3.40'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('GBP', adict['norm']['unit'])

        line = 'GBP3.40'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('GBP', adict['norm']['unit'])

        line = 'ＧＢＰ３．４０'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('GBP', adict['norm']['unit'])

        line = '3.40人民元'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('CNY', adict['norm']['unit'])

        line = '3.40元'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('CNY', adict['norm']['unit'])

        line = 'CNY3.40'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('CNY', adict['norm']['unit'])

        line = 'ＣＮＹ３．４０'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('CNY', adict['norm']['unit'])

        line = '3.40ルピー'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('INR', adict['norm']['unit'])

        line = '3.40インドルピー'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('INR', adict['norm']['unit'])

        line = '3.40インド・ルピー'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(3.40, adict['norm']['value'])
        self.assertEqual('INR', adict['norm']['unit'])


    def test_extract_currency_with_cents_jp(self):
        line = '二千五百六十二万千二百三十四ドル　五十セント'
        # (25,621,234 dollars 50 cents)
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(25621234.50, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '二千五百六十二万千二百三十四ドル'
        # (25,621,234 dollars 50 cents)
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(25621234, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])

        line = '五十セント'
        alist = regexcand_jp.extract_currencies(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(0.5, adict['norm']['value'])
        self.assertEqual('USD', adict['norm']['unit'])



class TestPercentJPUtils(unittest.TestCase):

    def test_extract_percent_jp(self):
        "Test extract_percent_jp()"

        line = '四Percent'
        alist = regexcand_jp.extract_percents(line)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(4, adict['norm']['value'])
        self.assertEqual('%', adict['norm']['unit'])

        line = '四percent'
        alist = regexcand_jp.extract_percents(line)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(4, adict['norm']['value'])
        self.assertEqual('%', adict['norm']['unit'])

        line = '四 percent'
        alist = regexcand_jp.extract_percents(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(4, adict['norm']['value'])
        self.assertEqual('%', adict['norm']['unit'])

        line = '四パーセント'
        alist = regexcand_jp.extract_percents(line)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(4, adict['norm']['value'])
        self.assertEqual('%', adict['norm']['unit'])

        line = '四 パーセント'
        alist = regexcand_jp.extract_percents(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(4, adict['norm']['value'])
        self.assertEqual('%', adict['norm']['unit'])

        line = '3割8分9厘'
        adict_list = regexcand_jp.extract_percents(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        print('adict')
        print(adict)
        self.assertEqual(38.9, adict['norm']['value'])

        line = '3割'
        adict_list = regexcand_jp.extract_percents(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        print('adict')
        print(adict)
        self.assertEqual(30, adict['norm']['value'])

        line = '8分9厘'
        adict_list = regexcand_jp.extract_percents(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        print('adict')
        print(adict)
        self.assertEqual(8.9, adict['norm']['value'])

        line = '8分'
        adict_list = regexcand_jp.extract_percents(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        print('adict')
        print(adict)
        self.assertEqual(8, adict['norm']['value'])

        line = '9厘'
        adict_list = regexcand_jp.extract_percents(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        print('adict')
        print(adict)
        self.assertEqual(0.9, adict['norm']['value'])

        line = '半分'
        adict_list = regexcand_jp.extract_percents(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        print('adict')
        print(adict)
        self.assertEqual(50, adict['norm']['value'])

        line = '100分の5'
        adict_list = regexcand_jp.extract_percents(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        print('adict')
        print(adict)
        self.assertEqual(5, adict['norm']['value'])

        line = '5/100'
        adict_list = regexcand_jp.extract_percents(line)
        self.assertEqual(len(adict_list), 1)
        adict = adict_list[0]
        print('adict')
        print(adict)
        self.assertEqual(5, adict['norm']['value'])

        # from Japan team's description document
        line = '12.3パーセント'
        alist = regexcand_jp.extract_percents(line)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(12.3, adict['norm']['value'])
        self.assertEqual('%', adict['norm']['unit'])

        line = '12.3%'
        alist = regexcand_jp.extract_percents(line)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(12.3, adict['norm']['value'])
        self.assertEqual('%', adict['norm']['unit'])

        line = '１２．３％'
        alist = regexcand_jp.extract_percents(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(12.3, adict['norm']['value'])
        self.assertEqual('%', adict['norm']['unit'])

        line = '１割２分３厘'
        alist = regexcand_jp.extract_percents(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(12.3, adict['norm']['value'])
        self.assertEqual('%', adict['norm']['unit'])

        line = '五〇・〇四 パーセント'
        # 五〇・〇四 パーセント = 50.04%
        alist = regexcand_jp.extract_percents(line, is_norm_dbcs_sbcs=True)
        self.assertEqual(1, len(alist))
        adict = alist[0]
        self.assertEqual(50.04, adict['norm']['value'])
        self.assertEqual('%', adict['norm']['unit'])

