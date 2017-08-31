import re
from typing import List


from kirke.docstruct import lxlineinfo
from kirke.docstruct.doctree import DTreeSegment

from kirke.utils import strutils, engutils

debug_mode = True

class ParaV2:

    def __init__(self, ptype: str, pagenum: int, start: int, end: int, lineinfo_list, text: str):
        self.ptype = ptype
        self.page = pagenum
        self.lineinfo_list = lineinfo_list
        self.start = start
        self.end = end
        self.text = text

    def __str__(self):
        # divider = '\n================================'
        text = self.text
        if self.ptype == 'toc':
            st_list = [linfo.text for linfo in self.lineinfo_list]
            text = '\n'.join(st_list)
        return "\n=====  Parav2(%s, page=%d, start=%d, end=%d)=================\n%s" % \
            (self.ptype, self.page, self.start, self.end, text)

# take the last of tocline and secheadx_list
def init_parav2(ptype, lineinfo_list, text):
    start, end = calc_lineinfos_start_end(lineinfo_list)
    pagenum = lineinfo_list[0].page
    parav2 = ParaV2(ptype, pagenum, start, end, lineinfo_list, text[start:end])
    return parav2


def calc_lineinfos_start_end(lineinfo_list):
    start = lineinfo_list[0].start
    for i, lineinfo in enumerate(lineinfo_list[1:]):
        if lineinfo.start < start:
            start = lineinfo.start
    end = lineinfo_list[-1].end
    for lineinfo in lineinfo_list[:-1]:
        if lineinfo.end > end:
            end = lineinfo.end

    return start, end



def page_lineinfo_list_to_parav2_list(lineinfo_list,
                                      skip_linfo_set,
                                      doc_text) -> List[ParaV2]:
    result = []
    num_not_eng = 0, 0
    cur_ptype = None
    cur_linfo_list = []

    norm_linfo_list = []
    for linfo in lineinfo_list:
        if linfo not in skip_linfo_set:
            norm_linfo_list.append(linfo)

    return norm_linfo_list
    

def is_good_parav2(linfo_list, strict=False):
    num_en, num_not_en = 0, 0

    if len(linfo_list) == 1 and linfo_list[0].is_english:
        return True
    
    if len(linfo_list) == 2:
        return linfo_list[0].is_english  # 2nd one can be either
    
    for linfo in linfo_list:
        if linfo.is_english:
            num_en += 1
        else:
            num_not_en += 1
    if strict:
        return len(linfo_list) >= 3 and num_not_en <= 1
    return len(linfo_list) >= 3 and num_en / (num_en + num_not_en) >= 0.5

"""
def has_consecutive_bad_groups(page_linfo_list, good_linfo_group_list, bad_linfo_group_list):
    good_linfo_set = set(good_linfo_list)

    bad_linfo_list = []
    bad_linfo_seq_list = []    
    for i, linfo in enumerate(page_linfo_list):
        if linfo not in good_linfo_list:
            bad_linfo_list.append(linfo)
            bad_linfo_seq_list.append(i)
"""

def are_consecutive_groups(linfo_group_list, seq_list):
    num_center = 0
    for linfo_group in linfo_group_list:
        for linfo in linfo_group:
            if linfo.is_center():
                num_center += 1
    if num_center / len(seq_list) > 0.8:
        return False    # this is a title page, or section
    
    if len(seq_list) < 4:
        return False
    prev_seq = seq_list[0]
    # can diff up to 2
    for seq in seq_list[1:]:
        if seq - prev_seq > 2:
            return False
        prev_seq = seq
    return True

def group_together(linfo_group_list):
    result = []
    for linfo_group in linfo_group_list:
        for linfo in linfo_group:
            result.append(linfo)
    return result

# take the last of tocline and secheadx_list
def extract_parav2(para_name, lineinfo_list, doc_text):
    start, end = calc_lineinfos_start_end(lineinfo_list)    
    obj = init_parav2(para_name, lineinfo_list, doc_text[start:end])
    return obj, lineinfo_list

def is_gb_status(gb_para_list, status, i, i_len):
    if i < i_len:
        xx_status, para = gb_para_list[i]
        return xx_status == status
    return False

NAME_BY_TITLE_PAT = re.compile(r'^(name|title)', re.IGNORECASE)
ADDRESS_PAT = re.compile(r'^(attention|attn|c\/o|telephon|phone|tel|fax|cel)', re.IGNORECASE)

def find_parav2s(page_lineinfos_list, skip_linfo_set, doc_text):

    result = []

    cur_parav2 = None

    for page_num, page_lineinfos in enumerate(page_lineinfos_list, 1):
        # if page_num == 63 or page_num == 37:
        # if page_num == 91:
        #     print("hellolllll")

        page_linfo_list = list(page_lineinfos)
        page_linfo_list = page_lineinfo_list_to_parav2_list(page_linfo_list, skip_linfo_set, doc_text)

        cur_linfo_list = []
        page_paras = []
        page_bad_paras = []
        num_eng, num_not_eng = 0, 0
        num_line_lt_50_chars = 0
        num_line_lt_30_chars = 0
        num_word_in_page = 0
        page_num_date = 0

        for i, lineinfo in enumerate(page_linfo_list):

            if not lineinfo.is_close_prev_line or i == 0:
                cur_linfo_list = []
                page_paras.append(cur_linfo_list)
            if lineinfo.is_english:
                num_eng += 1
            else:
                num_not_eng += 1
            if len(lineinfo.text) < 50:
                num_line_lt_50_chars += 1
            if len(lineinfo.text) < 30:
                num_line_lt_30_chars += 1
            num_word_in_page += len(lineinfo.words)
            page_num_date += strutils.count_date(lineinfo.text)
            cur_linfo_list.append(lineinfo)

        print("\n\n=== page #{} =====================".format(page_num))

        good_pid_group_list = []
        bad_pid_group_list = []
        num_good_linfos = 0
        num_bad_linfos = 0
        gb_pid_group_list = []
        # to determine if should use whole page
        count_num_consecutive_good_para = 0
        max_num_consecutive_good_para = 0
        is_prev_good_para = False
        for p_i, page_para in enumerate(page_paras, 1):
            print()
            if is_good_parav2(page_para):
                for lineinfo in page_para:
                    print("GG-{}\t{}".format(p_i, lineinfo.tostr2(doc_text)))
                if is_prev_good_para:
                    count_num_consecutive_good_para += 1
                else:
                    count_num_consecutive_good_para = 1
                if count_num_consecutive_good_para > max_num_consecutive_good_para:
                    max_num_consecutive_good_para = count_num_consecutive_good_para
                is_prev_good_para = True

                num_good_linfos += len(page_para)
                good_pid_group_list.append((p_i, page_para))
                gb_pid_group_list.append(('good', page_para))
            else:
                for lineinfo in page_para:
                    print("##-{}\t{}".format(p_i, lineinfo.tostr2(doc_text)))
                num_bad_linfos += len(page_para)
                bad_pid_group_list.append((p_i,page_para))
                gb_pid_group_list.append(('bad', page_para))

                count_num_consecutive_good_para = 0
                is_prev_good_para = False

        bxx_group_list = []
        bxx_group = []
        if bad_pid_group_list:
            # prev_gb_status, gb_para = gb_pid_group_list[0]
            gb_len = len(gb_pid_group_list)
            gb_index = 0
            is_in_good_status = True
            # expand bad group
            while gb_index < gb_len:
                gb_status, gb_para = gb_pid_group_list[gb_index]

                if is_in_good_status:
                    if gb_status != 'good':
                        bxx_group = [gb_para]
                        bxx_group_list.append(bxx_group)
                        is_in_good_status = False
                    # else:  # good
                    #   pass
                else:  # not is_in_good_status
                    # if we have a really big good solid good paragraph, cannot jump over it
                    if gb_status == 'good' and is_good_parav2(gb_para) and len(gb_para) >= 4:
                        is_in_good_status = True 
                    elif gb_status == 'good':
                        if is_gb_status(gb_pid_group_list, 'bad', gb_index + 1, gb_len):
                            bxx_group.append(gb_para)
                            bxx_group.append(gb_pid_group_list[gb_index + 1][1])
                            gb_index += 1
                            # is_in_good_status = False
                        elif is_gb_status(gb_pid_group_list, 'bad', gb_index + 2, gb_len):
                            bxx_group.append(gb_para)
                            bxx_group.append(gb_pid_group_list[gb_index + 1][1])                            
                            bxx_group.append(gb_pid_group_list[gb_index + 2][1])
                            gb_index += 2
                            # is_in_good_status = False
                        else:
                            is_in_good_status = True
                    else:  # gb_status == 'bad'
                        bxx_group.append(gb_para)

                gb_index += 1

            for p_i, bxx_group in enumerate(bxx_group_list, 1):

                num_cn = 0
                num_lineinfo = 0
                num_name_by_title = 0
                num_address = 0
                num_math_equation = 0
                num_word_in_bxx_group = 0
                num_number = 0
                num_date = 0
                bxx_linfo_list = []
                for para in bxx_group:
                    # print("para = {}".format(para))
                    for linfo in para:
                        if linfo.is_center():
                            num_cn += 1
                        num_number += strutils.count_number(linfo.text)
                        num_date += strutils.count_date(linfo.text)
                        if engutils.is_math_equation(linfo.text):
                            num_math_equation += 1
                        if NAME_BY_TITLE_PAT.match(linfo.text):
                            num_name_by_title += 1
                        if ADDRESS_PAT.match(linfo.text):
                            num_address += 1
                        num_word_in_bxx_group += len(linfo.words)
                        bxx_linfo_list.append(linfo)

                    num_lineinfo += len(para)

                table_type = None
                if num_name_by_title >= 1:
                    table_type = 'signature'
                elif num_address >= 2 :
                    table_type = 'address'
                elif num_math_equation >= 1 :
                    table_type = 'equation'                    
                elif (num_word_in_bxx_group and   # not zero
                      num_number / num_word_in_bxx_group > 0.4):
                    table_type = 'number'
                elif num_date >= 2 :
                    table_type = 'has_date'

                if num_lineinfo <= 2:
                    continue
                # sometimes address are centered
                if not table_type and num_cn / num_lineinfo >= 0.5:
                    continue
                if lxlineinfo.is_itemized_list(bxx_linfo_list):
                    continue



                print("page_overlap_perc = {:.2f}".format(num_word_in_bxx_group / num_word_in_page))
                print("avg lineinfo / group = {:.2f}".format(len(page_linfo_list) / len(page_paras)))
                print("num_not_eng / (eng + not_eng) = {:.2f}".format(num_not_eng / (num_eng + num_not_eng)))
                print("num_line_lt_50char / total lines = {:.2f}".format(num_line_lt_50_chars / len(page_linfo_list)))
                print("num_line_lt_30char / total lines = {:.2f}".format(num_line_lt_30_chars / len(page_linfo_list)))
                print("page_num_date = {}, len(page_paras) = {}".format(page_num_date, len(page_paras)))
                if ((num_not_eng / (num_eng + num_not_eng) > 0.8) or
                    (num_line_lt_50_chars / len(page_linfo_list) > 0.8) or
                    (num_line_lt_30_chars / len(page_linfo_list) > 0.5) or
                    (page_num_date >= 4 and len(page_paras) > 10) or
                    (((not max_num_consecutive_good_para >= 3) and
                      ((num_word_in_bxx_group / num_word_in_page > 0.75) or
                       (num_word_in_bxx_group / num_word_in_page > 0.5 and len(page_linfo_list) / len(page_paras) < 2.0) or
                       (num_not_eng / (num_eng + num_not_eng) > 0.64) or
                       (num_line_lt_50_chars / len(page_linfo_list) > 0.7))))):
                    start_bad_para = 0
                    for page_para in page_paras:
                        print("check_is_good_para2 = {}, len(para) = {}".format(is_good_parav2(page_para, strict=True), len(page_para)))
                        if is_good_parav2(page_para, strict=True) and len(page_para) >= 2:
                            start_bad_para += 1
                        else:
                            break
                        
                    for page_para in page_paras[start_bad_para:]:
                        for linfo in page_para:
                            if table_type:
                                print("PG-TABLE-{}-{}\t{}".format(table_type, p_i, linfo.tostr2(doc_text)))
                            else:
                                print("PG-TABLE-X2???????-{}\t{}".format(p_i, linfo.tostr2(doc_text)))
                else:
                    for para in bxx_group:
                        for linfo in para:
                            if table_type:
                                print("---TABLE-{}-{}\t{}".format(table_type, p_i, linfo.tostr2(doc_text)))
                            else:
                                print("---TABLE-X2???????-{}\t{}".format(p_i, linfo.tostr2(doc_text)))
            
            
#        if num_not_eng >= 10 and num_line_lt_50_chars > 5:
#            parav2 = init_parav2('table', page_linfo_list, doc_text)
#            for lineinfo in page_linfo_list:
#                print("??TABLE-X1??????-{}\t{}".format(p_i, lineinfo.tostr2(doc_text)))
#            # to go the next page
#            continue

#        if bad_group_list and are_consecutive_groups(bad_group_list, bad_group_id_list):
#            group2 = group_together(bad_group_list)
#            for lineinfo in group2:
#                print("??TABLE-X2??????-{}\t{}".format(p_i, lineinfo.tostr2(doc_text)))            
            



        


"""        
        group_list = find_adjacent_lines(page_linfo_list, toc_linfo_list, pheader_linfo_list, pfooter_linfo_list)

        # check if the group_list basically covered the whole page
        if group_list and is_word_overlap(group_list, page_linfo_list, skip_lineinfo_set, 0.75):
            # print("checking group overlap, pagenum = {}, {}".format(page_num, group_list))
            segment_obj, whole_lineinfo_list = extract_whole_page(page_linfo_list, text)
            graphspan_results.append(segment_obj)
            graphspan_lineinfo_results.append(whole_lineinfo_list)
            continue

        for lineinfo_group in group_list:
            if not lineinfo_group:
                print("empty group in page_num: {}".format(page_num))
                
            if is_signature(lineinfo_group):
                segment_obj, signature_lineinfo_list = extract_signature(lineinfo_group,
                                                                         text)
                graphspan_results.append(segment_obj)
                graphspan_lineinfo_results.append(signature_lineinfo_list)
            elif is_exhibit(lineinfo_group):
                segment_obj, exhibit_lineinfo_list = extract_exhibit(lineinfo_group,
                                                                     text)
                graphspan_results.append(segment_obj)
                graphspan_lineinfo_results.append(exhibit_lineinfo_list)                
            elif is_graph(lineinfo_group):
                segment_obj, graph_lineinfo_list = extract_graph(lineinfo_group,
                                                                 text)                
                graphspan_results.append(segment_obj)
                graphspan_lineinfo_results.append(graph_lineinfo_list)

                
    return graphspan_results, graphspan_lineinfo_results

"""
