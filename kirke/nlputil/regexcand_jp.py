import re
from typing import Dict, List, Match

from kirke.nlputil import regexcand_en
from kirke.nlputil import text2int_jp

from kirke.utils import unicodeutils

IS_DEBUG = False

# NUMERIC_REGEX_ST
# pylint: disable=line-too-long
# ((零|一|二|三|四|五|六|七|八|九|〇|壱|弐|参|元|拾|佰|陌|仟|阡|萬|两|亿|十|百|千|万|億|兆)+・(零|一|二|三|四|五|六|七|八|九|〇|壱|弐|参|元|拾|佰|陌|仟|阡|萬|两|亿|十|百|千|万|億|兆)+|([-+]?[0-9,\.]*[0-9]+)|(零|一|二|三|四|五|六|七|八|九|〇|壱|弐|参|元|拾|佰|陌|仟|阡|萬|两|亿|十|百|千|万|億|兆)+|ゼロ)

# Looking at the above regex, there is some, but not significant backtracking issues
# which happened with English Currencies and Percentages.
# The sub-expression, '([-+]?[0-9,\.]*[0-9]+)%' for percentage runs in sub-seconds
# for a 120 digit string (without percentage at the end) is less than 0.3 seconds.
# for 240 digits string, it is less than 2 seconds.  It's unlikely there are such
# astronomical numbers in contract these days.

# pylint: disable=line-too-long
CURRENCY_PAT_ST = r'((((\bUSD|\bEUR|\bGBP|\bCNY|\bJPY|\bINR|\bRupees?|\bRs\.?)|[\$€£円¥₹]) *{})|' \
                  r'({} *(USD|[dD]ollars?|u\.\s*s\.\s*dollars?|ドル|米ドル|㌦|アメリカドル|弗)( *{} *(セント|cents?)))|' \
                  r'({} *((USD|euros?|EUR|GBP|CNY|JPY|INR|Rs|[dD]ollars?|u\.\s*s\.\s*dollars?|ドル|米ドル|㌦|アメリカドル|弗|[eE]uros?|ユーロ|' \
                  r'[pP]ounds?|ポンド|[yY]uans?|人民元|元|[yY]ens?|[rR]upees?|インド・ルピー|インドルピー|ルピー)|' \
                  r'[\$€£円¥₹]))|' \
                  r'({} *(セント|cents?)))'.format(text2int_jp.NUMERIC_REGEX_ST,
                                                text2int_jp.NUMERIC_REGEX_ST,
                                                text2int_jp.NUMERIC_REGEX_ST,
                                                text2int_jp.NUMERIC_REGEX_ST,
                                                text2int_jp.NUMERIC_REGEX_ST)

# CURRENCY_CENT_PAT_ST = r'( *{} *セント)'.format(text2int_jp.NUMERIC_REGEX_ST)


# CURRENCY_PAT_ST = r'{}([\$€£円¥￥₹])'.format(text2int_jp.NUMERIC_REGEX_ST)
# print('currency_pat_st:')
# print(CURRENCY_PAT_ST)

# CURRENCY_PAT_ST = text2int_jp.NUMERIC_REGEX_ST
CURRENCY_PAT = re.compile(CURRENCY_PAT_ST, re.I)

def currency_to_norm_dict(cx_mat: Match, line: str) -> Dict:
    if IS_DEBUG:
        print('  currency cx_mat group: {} {} [{}]'.format(cx_mat.start(),
                                                           cx_mat.end(),
                                                           cx_mat.group()))
        for gidx, unused_group in enumerate(cx_mat.groups(), 1):
            print("    cx_mat.group #{}: [{}]".format(gidx, cx_mat.group(gidx)))
    norm_unit = 'USD'
    norm_value = -1
    if cx_mat.group(5):
        # $35
        norm_unit = regexcand_en.normalize_currency_unit(cx_mat.group(3))
        norm_value = text2int_jp.extract_number(cx_mat.group(5))['norm']['value']
    elif cx_mat.group(10):
        # 二千五百六十二万千二百三十四ドル　五十セント
        # 2000 dollars 34 cents
        norm_unit = regexcand_en.normalize_currency_unit(cx_mat.group(16))
        norm_value = text2int_jp.extract_number(cx_mat.group(11))['norm']['value']
        if cx_mat.group(18):
            norm_cent_value = text2int_jp.extract_number(cx_mat.group(18))['norm']['value']
            norm_value += norm_cent_value / 100
    elif cx_mat.group(25):
        # 二千五百六十二万千二百三十四ドル
        norm_unit = regexcand_en.normalize_currency_unit(cx_mat.group(30))
        norm_value = text2int_jp.extract_number(cx_mat.group(25))['norm']['value']
    elif cx_mat.group(33):
        # 二千五百六十二万千二百三十四ドル
        norm_unit = regexcand_en.normalize_currency_unit(cx_mat.group(38))
        norm_value = text2int_jp.extract_number(cx_mat.group(33))['norm']['value'] / 100
    norm_dict = {'norm': {'unit': norm_unit,
                          'value': norm_value},
                 'text': line[cx_mat.start():cx_mat.end()],
                 'start': cx_mat.start(),
                 'end': cx_mat.end(),
                 'concept': 'currency'}
    return norm_dict

def extract_currencies(line: str, is_norm_dbcs_sbcs=False) -> List[Dict]:
    if is_norm_dbcs_sbcs:
        line = unicodeutils.normalize_dbcs_sbcs(line)
    result = []
    mat_list = CURRENCY_PAT.finditer(line)
    for mat in mat_list:
        # print('mat: [{}]'.format(mat.group()))
        norm_dict = currency_to_norm_dict(mat, line)
        result.append(norm_dict)
    return result


# the '\s*' is just in case there are spaces between the digits
PERCENT_PAT_ST = r'{} *(percent|パーセント|%)|' \
                 r'([0-9]+)/1\s*0\s*0|1\s*0\s*0分の{}|' \
                 r'{}割({}分)?({}厘)?|{}分({}厘)?|{}厘|(半分)'.format(text2int_jp.NUMERIC_REGEX_ST,
                                                              text2int_jp.NUMERIC_REGEX_ST,
                                                              text2int_jp.NUMERIC_REGEX_ST,
                                                              text2int_jp.NUMERIC_REGEX_ST,
                                                              text2int_jp.NUMERIC_REGEX_ST,
                                                              text2int_jp.NUMERIC_REGEX_ST,
                                                              text2int_jp.NUMERIC_REGEX_ST,
                                                              text2int_jp.NUMERIC_REGEX_ST)

# print('PERCENT_PAT_ST:')
# print(PERCENT_PAT_ST)

PERCENT_PAT = re.compile(PERCENT_PAT_ST, re.I)

def percent_to_norm_dict(cx_mat: Match, line: str) -> Dict:
    if IS_DEBUG:
        print('  percent cx_mat group: {} {} [{}]'.format(cx_mat.start(), cx_mat.end(), cx_mat.group()))
        for gidx, unused_group in enumerate(cx_mat.groups(), 1):
            print("    perc cx_mat.group #{}: [{}]".format(gidx, cx_mat.group(gidx)))
    norm_value = -1.0
    if cx_mat.group(1):
        norm_value = text2int_jp.extract_number(cx_mat.group(1))['norm']['value']
    elif cx_mat.group(7):
        # 5/100
        norm_value = text2int_jp.extract_number(cx_mat.group(7))['norm']['value']
    elif cx_mat.group(8):
        # 100分の5
        norm_value = text2int_jp.extract_number(cx_mat.group(8))['norm']['value']
    elif cx_mat.group(13):
        # 3割8分9厘
        norm_value = 10 * text2int_jp.extract_number(cx_mat.group(13))['norm']['value']
        norm_value = round(norm_value, 1)
        if cx_mat.group(19):
            norm_value += text2int_jp.extract_number(cx_mat.group(19))['norm']['value']
            norm_value = round(norm_value, 2)
        if cx_mat.group(25):
            norm_value += 0.1 * text2int_jp.extract_number(cx_mat.group(25))['norm']['value']
            norm_value = round(norm_value, 3)
    elif cx_mat.group(30):
        # 8分9厘
        norm_value = text2int_jp.extract_number(cx_mat.group(30))['norm']['value']
        norm_value = round(norm_value, 2)
        if cx_mat.group(36):
            norm_value += 0.1 * text2int_jp.extract_number(cx_mat.group(36))['norm']['value']
            norm_value = round(norm_value, 3)
    elif cx_mat.group(41):
        # 9厘
        norm_value = 0.1 * text2int_jp.extract_number(cx_mat.group(41))['norm']['value']
        norm_value = round(norm_value, 3)
    elif cx_mat.group(46):
        # 半分
        norm_value = 50

    norm_dict = {'norm': {'unit': '%',
                          'value': norm_value},
                 'text': line[cx_mat.start():cx_mat.end()],
                 'start': cx_mat.start(),
                 'end': cx_mat.end(),
                 'concept': 'percent'}
    return norm_dict


def extract_percents(line: str, is_norm_dbcs_sbcs=False) -> List[Dict]:
    if is_norm_dbcs_sbcs:
        line = unicodeutils.normalize_dbcs_sbcs(line)
    # print('extract_percents({})'.format(line))
    result = []
    mat_list = PERCENT_PAT.finditer(line)
    for mat in mat_list:
        # print('mat: [{}]'.format(mat.group()))
        norm_dict = percent_to_norm_dict(mat, line)
        result.append(norm_dict)
    return result
