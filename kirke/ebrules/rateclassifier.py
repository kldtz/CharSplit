from kirke.utils import strutils
from typing import List

DEBUG_MODE = False

def is_energy_rate_table(st_list: List[str]):
    num_year = 0
    num_list = []
    found_MWh = False
    found_contract_rate = False
    found_payment_rate = False
    found_pricing = False
    found_column_year = False
    found_dollar = False
    found_exhibit = False

    if DEBUG_MODE:
        table_text = ' '.join(st_list)
        print("table_text =[{}]".format(table_text))

    for linenum, line_st in enumerate(st_list):
        words = line_st.split()
        lc_line_st = line_st.lower()

        if 'exhibit' in lc_line_st:
            found_exhibit = True

        if 'MWh' in line_st or 'kWh' in line_st:
            found_MWh = True

        if strutils.are_all_substrs_in_st(['contract', 'rate'], lc_line_st):
            found_contract_rate = True

        if strutils.are_all_substrs_in_st(['payment', 'rate'], lc_line_st):
            found_payment_rate = True

        if 'pricing' in lc_line_st:
            found_pricing = True

        if 'year' in lc_line_st:
            found_column_year = True

        if '$' in lc_line_st:
            found_dollar = True

        # count number of years
        for word in words:
            if strutils.is_all_digits(word):
                intval = int(word)
                if intval > 2000 and intval < 2040:
                    num_year += 1
                num_list.append(intval)

    has_num_seq = find_any_numeric_seq(num_list)

    if DEBUG_MODE:
        print("\nnum_year = {}".format(num_year))
        print("has_num_seq = {}".format(has_num_seq))
        print("found_MWh = {}".format(found_MWh))
        print("found_contract_rate = {}".format(found_contract_rate))
        print("found_payment_rate = {}".format(found_payment_rate))
        print("found_pricing = {}".format(found_pricing))
        print("found_exhibit = {}".format(found_exhibit))
        print("found_column_year = {}".format(found_column_year))
        print("found_dollar = {}".format(found_dollar))

    if (found_exhibit
        and
        (found_contract_rate or
         found_payment_rate or
         found_pricing)
        and
        (found_MWh or num_year > 3 or has_num_seq)):
        return True

    # a relax constraint, because "exhibit is missing"?
    if  ((found_contract_rate or
          found_payment_rate or
          found_pricing)
         and
         found_column_year
         and
         found_dollar):
        return True

    # no exhibit
    if ((found_contract_rate or
         found_payment_rate or
         found_pricing)
        and
        found_MWh
        and
        num_year > 3):
        return True

    return False

def classify_table_list(table_span_list, doc_text):
    ant_list = []
    for table_span in table_span_list:
        startx = table_span[0]
        endx = table_span[1]
        st_text = doc_text[startx:endx]
        # TODO, jshaw, this is outputing a lot of interesting bad tables...
        # print("classify table_span: {}".format(table_span))
        # print("    txt= {}".format(strutils.sub_nltab_with_space(st_text)))
        if is_energy_rate_table(st_text.split('\n')):
            ant_list.append({'end': endx,
                             'label': 'rate_table',
                             'prob': 1.0,
                             'start': startx,
                             'span_list': [{'start': startx,
                                            'end': endx}],
                             'text': strutils.sub_nltab_with_space(st_text)})
    return ant_list


# this is a hack, not really work for any numeric list
def find_any_numeric_seq(num_list):
    if len(num_list) < 3:
        return False
    if 3 in num_list and 4 in num_list and 5 in num_list:
        index3 = num_list.index(3)
        index4 = num_list.index(4)
        index5 = num_list.index(5)

        if index3 < index4 and index4 < index5:
            return True
    return False


def find_any_numeric_seq(num_list):
    if len(num_list) < 3:
        return False
    less_10_set = set([])
    for num in num_list:
        if num <= 10:
            less_10_set.add(num)
    if len(less_10_set) > 4:  # because of mismatching
        return True

    return False
