from typing import List
import re

from kirke.docstruct import lxlineinfo

from kirke.docstruct import sentv2 as lxsentv2

from kirke.utils import strutils, mathutils

from kirke.docstruct import sentv2 as lxsentv2


NOT_TABLE_CATEGORIES = set(['toc-head', 'tocline2', 'pagenum', 'footer', 'tocline', 'toc', 'InTable'])

class EbTable:

    def __init__(self, sentV2_list):
        all_lineinfo_list = []
        for sentV2 in sentV2_list:
            all_lineinfo_list.extend(sentV2.lineinfo_list)
        min_start = all_lineinfo_list[0].start
        max_end = all_lineinfo_list[0].end

        for lineinfo in all_lineinfo_list[1:]:
            if lineinfo.start < min_start:
                min_start = lineinfo.start
            if lineinfo.end < max_end:
                max_end = lineinfo.end

        table_start, table_end = lxlineinfo.find_list_start_end(all_lineinfo_list)
        self.pagenum = sentV2_list[0].pagenum
        self.sentV2_list = sentV2_list
        self.start = table_start
        self.end = table_end


    def append_lineinfo(self, lineinfo):
        self.lineinfo_list.append(lineinfo)

    def __repr__(self):
        return ("EbTable(%d, %d)" % (self.start, self.end))


def find_adjacent_maybes(maybetable_ystart_list, sentV2_list):
    tmp_result = []
    cur_adjacents = []
    prev_ystart = 0
    prev_is_table = False
    
    for (is_maybe_table, ystart), sentV2 in zip(maybetable_ystart_list,
                                                sentV2_list):
        # print("find_adjacent_maybes: {}, {}, {}".format(is_maybe_table, ystart, sentV2.text))
        if is_maybe_table:
            if prev_is_table:
                if ystart - prev_ystart > 60:
                    # even if too small, will remove later
                    cur_adjacents = [sentV2]
                    tmp_result.append(cur_adjacents)
                else: # diff <= 60
                    cur_adjacents.append(sentV2)
            else:  # prev is not a table
                cur_adjacents = [sentV2]  # wipe out any old if too small
                tmp_result.append(cur_adjacents)
        else:  # not in a table
            pass
        prev_is_table = is_maybe_table
        prev_ystart = ystart

    # remove any that are too small
    result = []
    for tmp_sentv2s in tmp_result:
        if len(tmp_sentv2s) >= 3:
            result.append(tmp_sentv2s)
            #print("rrr1xxx --------")
            #for tmpsentv2 in tmp_sentv2s:
            #    print("rrrr {}".format(tmpsentv2.text))
            
    # print("result= {}".format(result))
    return result


def find_table_by_short_num(page_sentV2s_list):
    debug_mode = False
    result = []
    for pagenum, page_sentV2s in enumerate(page_sentV2s_list, 1):

        if not page_sentV2s:  # blank pages
            continue

        found = False

        if debug_mode:
            print("===== find_table_by_short_num, page #{}".format(pagenum))
        num_maybe_table = 0
        status_ystart_list = []
        for sentV2 in page_sentV2s:

            if sentV2.category in NOT_TABLE_CATEGORIES:
                continue

            word_list = []
            for lineinfo in sentV2.lineinfo_list:
                word_list.extend(lineinfo.words)
            num_alpha, num_digit, num_dollar, num_other = (
                strutils.count_word_category(word_list))
            if debug_mode:
                print("word_list: {}".format(word_list))
                print("  num_alpha = %d, digit= %d, dollar = %d, other = %d" %
                      (num_alpha, num_digit, num_dollar, num_other))

            maybe_table = False
            if num_digit > 0 and num_alpha < 2:  # this might be a table
                num_maybe_table += 1
                maybe_table = True

            status_ystart_list.append((maybe_table, sentV2.get_ystart()))

        # print("num_maybe_table = {}".format(num_maybe_table))

        if num_maybe_table >= 3:
            # these are sentV2s
            adjacent_maybe_tables_list = (
                find_adjacent_maybes(status_ystart_list, page_sentV2s))

            if adjacent_maybe_tables_list:
                # print("xxxjjjjjjjjjjjjjjjjjj len = {}".format(len(adjacent_maybe_tables_list)))
                for maybe_tables in adjacent_maybe_tables_list:
                    # print("xxxjjjlen = {}".format(len(maybe_tables)))
                    for row_count, sentV2 in enumerate(maybe_tables, 1):
                        # sentV2.align_list.append("InTTTT-{}".format(row_count))
                        # sentV2.align_list.append('InTable')
                        sentV2.category = 'table'
                    ebtable = EbTable(maybe_tables)
                    result.append(ebtable)


        # if more than 70% of a page is a table, we are probably missing the lables of the table
        # There is header and other stuff
        num_table_row_in_page = 0
        for page_sentV2 in page_sentV2s:
            if page_sentV2.category in set(['table', 'exhibit']):
                num_table_row_in_page += 1
        
        if num_table_row_in_page / len(page_sentV2s) >= 0.7:
            for page_sentV2 in page_sentV2s:
                if page_sentV2.category == '---':
                    page_sentV2.category = 'table'

    return result

"""
# Because there can be blank pages, not all page_sent_v2s_list are present
# in the list.  page_num might be > len(page_sentV2s_list)
def jump_to_page_lineinfos(page_sentV2s_list, page_num):
    fake_max_pagenum = len(page_sentV2s_list)
    if page_num >= 0 and page_num < fake_max_pagenum:
        tmp_sentV2s = page_sentV2s_list[page_num]
        real_pagenum = tmp_sentV2s[0].pagenum
        if real_pagenum == page_num:
            return tmp_sentV2s
        if real_pagenum > page_num:  # shouldn't happen
            pass
        if real_pagenum < page_num:  # shouldn't happen
            while page_num >= tmp_sentV2s[0].pagenum:
                tmp_sentV2s = page_sentV2s_list[page_num)

            
    return []
"""

def find_table_groups(page_sentV2s_list):
    result = []
    for pagenum, page_sentV2s in enumerate(page_sentV2s_list, 1):

        if not page_sentV2s:  # blank pages
            continue

        found = False

        # print("===== find_table_groups, page #{}".format(pagenum))
        intable_sentV2s_list = []
        curr_intable_sentV2s = []
        is_prev_intable = False
        for sent_seq, sentV2 in enumerate(page_sentV2s, 0):

            if sentV2.category in NOT_TABLE_CATEGORIES:
                continue

            if 'InTable' in sentV2.align_list:
                if not is_prev_intable:
                    curr_intable_sentV2s = [sentV2]
                    intable_sentV2s_list.append(curr_intable_sentV2s)
                else:
                    curr_intable_sentV2s.append(sentV2)
                # print("find_table_group: {} {} ({}) [{}]".format(sent_seq, sentV2.category, sentV2.align_list, sentV2.text))
                # for x2, linfo in enumerate(sentV2.lineinfo_list):
                #    print("\t\tlinfo {} x1={}, x2={}, y={} : [{}] ".format(x2, linfo.xStart, linfo.xEnd, linfo.yStart, linfo.text))
                is_prev_intable = True
            else:
                is_prev_intable = False

        if intable_sentV2s_list:
            #for row_count, sentV2 in enumerate(intable_sentV2_list, 1):
            #    sentV2.align_list.append("InTTTB-{}".format(row_count))

            for intable_sentV2s in intable_sentV2s_list:
                ebtable = EbTable(intable_sentV2s)
                result.append(ebtable)

    # print("reser reseult = {}".format(result))

    # mark each line in table as TABLE
    for ebtable in result:
        page_num = ebtable.pagenum
        page_sentV2s = page_sentV2s_list[page_num - 1]
        table_start = ebtable.start
        table_end = ebtable.end

        # print("j3 pagenum = {}".format(page_num))
        # print("page_sentV2s = {}".format(page_sentV2s))

        # there is no more "whole-age"
        #if is_whole_page_segmented(page_sentV2s):
        #    for page_sentV2 in page_sentV2s:
        #        if page_sentV2.category == 'whole-page':
        #            page_sentV2.category = 'table'
        #    continue

        num_table_row_in_page = 0
        for page_sentV2 in page_sentV2s:
            if mathutils.start_end_overlap((table_start, table_end),
                                           (page_sentV2.start, page_sentV2.end)):
                if page_sentV2.category in set(['---', 'graph']):
                    page_sentV2.category = 'table'
                    num_table_row_in_page += 1

        #if page_num == 43:
        #   print("num_table_row_in_page = {}".format(num_table_row_in_page))
        #   print("num(row) = {}".format(len(page_sentV2s)))
        #   print("xxx = {}".format(num_table_row_in_page / len(page_sentV2s)))
           
        # if more than 70% of a page is a table, we are probably missing the lables of the table
        # There is header and other stuff
        if num_table_row_in_page / len(page_sentV2s) >= 0.7:
            for page_sentV2 in page_sentV2s:
                if page_sentV2.category == '---':
                    page_sentV2.category = 'table'

    return result


#def is_whole_page_segmented(page_sentV2s):
#    for page_sentV2 in page_sentV2s:
#        if page_sentV2.category == 'whole-page':
#            return True
#    return False

# the last one, 'o' is a char for itemize
ITEMIZE_PAT = re.compile(r'^\(?([a-zA-Z]|\d+)[\)\.]\s*$')

def is_itemized_pair(s1, s2):
    if s1 == s2:
        s1_linfos = s1.lineinfo_list
        if len(s1_linfos) >= 2:
            return (ITEMIZE_PAT.match(s1_linfos[0].text) and
                    (s1_linfos[1].is_english or len(s1_linfos[1].text) < 20)) 
    else:
        s2_linfos = s2.lineinfo_list
        if s2_linfos:
            return (ITEMIZE_PAT.match(s1.text) and
                    (s2_linfos[0].is_english or len(s2_linfos[0].text) < 20)) 
    return False
              
def find_column_table_by_ystart(page_sentV2s_list):
    debug_mode = False
    
    for pagenum, page_sentV2s in enumerate(page_sentV2s_list, 1):

        if not page_sentV2s:  # blank pages
            continue                

        """
        has_no_style_para = False
        for sentV2 in page_sentV2s:
            if not sentV2.align_list:
                has_no_style_para = True
        if has_no_style_para:
            print("\n\npage {} has no style para".format(i))
            for sentV2 in page_sentV2s:
                if not sentV2.align_list:                
                    print("\n    sentV2: {}".format(sentV2))
        """

        #if i == 53:  # for test1, page 43 has a table
        #    print("hello234")
        #    debug_mode = True

        found = False
        paired_sentV2id_list = []
        
        yStart_list = []
        sentV2_id_list = []
        for s2id, sentV2 in enumerate(page_sentV2s):
            if sentV2.category in NOT_TABLE_CATEGORIES:
                continue
            yStart_list.extend([lineinfo.yStart for lineinfo in sentV2.lineinfo_list])
            # sck, there is a cleaner way to write this
            # [s2id * len(sentV2.lineinfo_list)]??
            sentV2_id_list.extend([s2id for lineinfo in sentV2.lineinfo_list])
            
        y_sentV2id_list = list(zip(yStart_list, sentV2_id_list))
        # print("yxx list:")
        prev_y = 0
        prev_s2id = 0
        for y, sentV2id in sorted(y_sentV2id_list):
            # print("   {}".format(yxx), end='')
            if debug_mode:
                if prev_y != 0:
                    print('  diff= {}'.format(y - prev_y))
                else:
                    print()

            if prev_y != 0:
                diff = y - prev_y
                if diff < lxlineinfo.MAX_Y_DIFF_AS_SAME:
                    found = True
                    if prev_s2id < sentV2id:
                        paired_sentV2id_list.append((prev_s2id, sentV2id))
                    else:
                        paired_sentV2id_list.append((sentV2id, prev_s2id))
                    
            prev_y = y
            prev_s2id = sentV2id

        if found:
            # print("\n\npage {} has table".format(i))
            #for sentV2 in page_sentV2s:
            #    if not sentV2.align_list:                
            #        print("\n    sentV2: {}".format(sentV2))

            # s1 and s2 can belong to the same "paragraph group"
            for s1, s2 in set(paired_sentV2id_list):
                s1_sentv2 = page_sentV2s[s1]
                s2_sentv2 = page_sentV2s[s2]                

                if debug_mode:
                    print()                
                    print("pair sent {} and {}:".format(s1, s2))
                    print("     sentV2 #{}: {}".format(s1, page_sentV2s[s1]))
                    print()
                    print("     sentV2 #{}: {}".format(s2, page_sentV2s[s2]))
                    print()
                    print("is_itemized_pair() = {}".format(is_itemized_pair(s1_sentv2, s2_sentv2)))
                if s1 == s2:
                    if page_sentV2s[s1].category != 'sechead' and not is_itemized_pair(s1_sentv2, s2_sentv2):
                        page_sentV2s[s1].align_list.append("InTable")
                elif page_sentV2s[s1].category != 'sechead' and not is_itemized_pair(s1_sentv2, s2_sentv2):
                    page_sentV2s[s1].align_list.append("InTable")
                    page_sentV2s[s2].align_list.append("InTable")
                    

def is_table_columns(sentV2_list: List[lxsentv2.SentV2]) -> bool:
    return len(sentV2_list) >= 5

def is_lineinfo_column_table(sentV2: lxsentv2.SentV2) -> bool:
    num_linfos = len(sentV2.lineinfo_list)
    if num_linfos <= 5:
        return False
    num_matched_column = 0
    col_count_list = []
    count_maybe_table_cols = 0
    for linfo in sentV2.lineinfo_list:
        # print("     linfo: [{}]".format(linfo.text))
        if 'CN' in linfo.align_label:
            num_words = len(linfo.words)
            if num_words <= 5 and num_words >= 2:
                col_count_list.append(num_words)
                count_maybe_table_cols += 1
            else:
                col_count_list.append(-10)
        else:
            col_count_list.append(-10)
            
    if count_maybe_table_cols < 5:
        # print("returning is_lineinfo_column_table False, easy")            
        return False
    

    num_col_agreement = 0
    prev_column_num = -10
    for num_word in col_count_list:
        if num_word != -10 and prev_column_num != -10:
            if ((num_word - 1 == prev_column_num) or
                (num_word == prev_column_num) or
                (num_word + 1 == prev_column_num)):
                num_col_agreement += 1
            else:
                pass
        else:
            pass
        prev_column_num = num_word

    if num_col_agreement / num_linfos >= 0.8:
        # print("returning is_lineinfo_column_table TRUE *****************x")
        return True
    # print("returning is_lineinfo_column_table False")    
    return False
    
# handle "January 8.74%" or "23 43"                    
def find_column_table_by_num_column(page_sentV2s_list):
    debug_mode = False
    for pagenum, page_sentV2s in enumerate(page_sentV2s_list):

        if not page_sentV2s:  # blank pages
            continue        

        if debug_mode:
            print("----- find_column_table_by_num_column, page {}".format(pagenum))
        col_count_list = []
        for sent_seq, sentV2 in enumerate(page_sentV2s):

            if is_lineinfo_column_table(sentV2):
              sentV2.align_list.append("InTable")  
            elif ('CN' in sentV2.align_list and
                len(sentV2.lineinfo_list) == 1):
                # for optimization, use 'num_words' local var
                num_words = len(sentV2.text.split())
                if num_words <= 5 and num_words >= 2:
                    col_count_list.append(num_words)
                else:
                    col_count_list.append(-10)
            else:
                col_count_list.append(-10)

        cur_column_group = []
        col_group_list = []
        prev_column_num = -10
        for num_word, sentV2 in zip(col_count_list, page_sentV2s):
            if num_word != -10:
                if prev_column_num != -10:
                    if ((num_word - 1 == prev_column_num) or
                        (num_word == prev_column_num) or
                        (num_word + 1 == prev_column_num)):
                        cur_column_group.append(sentV2)  # add to existing
                    else:
                        # create a new column group because col_count too different
                        cur_column_group = [sentV2]
                        col_group_list.append(cur_column_group)
                else:
                    # create a new column group because col_count too different
                    cur_column_group = [sentV2]
                    col_group_list.append(cur_column_group)                    
            prev_column_num = num_word

        for i, col_group in enumerate(col_group_list):
            if debug_mode:
                print("col_group {}".format(i))
                for j, sentV2 in enumerate(col_group):
                    print("\t{}\t{}".format(j, sentV2.text))

            if is_table_columns(col_group):
                for sentV2 in col_group:
                    sentV2.align_list.append('InTable')
    # return nothing now

        

def find_table(page_sentV2s_list):
    find_column_table_by_ystart(page_sentV2s_list)
    find_column_table_by_num_column(page_sentV2s_list)  # these only mark intable
    x1 = find_table_by_short_num(page_sentV2s_list)
    x2 = find_table_groups(page_sentV2s_list)

    # print("x1 = {}".format(x1))
    # print("x2 = {}".format(x2))
    # TODO, jshaw, should merge the tables if they overlap from the above sources
    result = x1
    result.extend(x2)
    return result
